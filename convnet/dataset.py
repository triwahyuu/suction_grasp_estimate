## data loader script
import random
import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class SuctionDataset(Dataset):
    def __init__(self, options, data_path=None, sample_list=None):
        self.path = data_path if data_path != None else options.data_path
        self.output_scale = options.output_scale
        self.img_height = options.img_height
        self.img_width = options.img_width
        
        ## data samples
        sample_path = sample_list if sample_list != None else options.sample_path
        self.sample_list = open(sample_path).read().splitlines()
        self.num_samples = len(self.sample_list)
        self.n_class = 3
        
        self.train_idx = 1
        self.train_epoch_idx = 1
        self.train_epoch_size = self.num_samples

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    def __getitem__(self, index):
        color_img = self.to_tensor(io.imread(self.path + 'color-input/' + self.sample_list[index] + '.png'))
        color_img = self.normalize(color_img)
        
        depth = self.to_tensor(io.imread(self.path + 'depth-input/' + \
            self.sample_list[index] + '.png').astype(np.float32)) * 65536/10000
        depth = depth.clamp(0.0, 1.2)
        depth_img = torch.cat([depth, depth, depth], 0)
        depth_img = self.normalize(depth_img)

        label = self.to_tensor(io.imread(self.path + 'label/' + self.sample_list[index] + '.png'))
        label = torch.round(label*2 + 1)

        return [color_img, depth_img], label

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    options = Struct(\
        data_path = '/home/tri/skripsi/dataset/',
        sample_path = '/home/tri/skripsi/dataset/test-split.txt',
        img_height =  480,
        img_width = 640,
        batch_size = 4,
        n_class = 3,
        output_scale = 8,
        shuffle = True,
        learning_rate = 0.001
    )

    # testing functionality
    suction_dataset = SuctionDataset(options, data_path=options.data_path, sample_list=options.sample_path)
    rgbd, label = suction_dataset[1]

    # create dataloader
    data_loader = DataLoader(suction_dataset, batch_size=options.batch_size, \
        shuffle=options.shuffle)
    
