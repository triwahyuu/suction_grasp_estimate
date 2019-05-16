## data loader script
import random
import os.path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import imgaug as ia
from imgaug import augmenters as iaa

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class SuctionDatasetNew(Dataset):
    def __init__(self, options, data_path=None, sample_list=None, mode='train'):
        self.path = data_path if data_path != None else options.data_path
        self.output_scale = options.output_scale
        self.img_height = options.img_height
        self.img_width = options.img_width
        self.mode = mode
        
        ## data samples
        sample_path = sample_list if sample_list != None else options.sample_path
        self.sample_list = open(sample_path).read().splitlines()
        self.num_samples = len(self.sample_list)
        self.n_class = 3
        
        self.train_idx = 1
        self.train_epoch_idx = 1
        self.train_epoch_size = self.num_samples

        ia.seed(int(time.time()))
        self.aug_seq = iaa.OneOf([
            iaa.Sequential([
                iaa.Fliplr(0.5), 
                iaa.PiecewiseAffine(scale=(0.0, 0.03)), 
                iaa.PerspectiveTransform(scale=(0.0, 0.1))
            ], random_order=True),
            iaa.Sequential([
                iaa.Flipud(0.5),
                iaa.PiecewiseAffine(scale=(0.0, 0.03)), 
                iaa.PerspectiveTransform(scale=(0.0, 0.1))
            ], random_order=True),
            iaa.Sequential([
                iaa.Fliplr(0.25), 
                iaa.Dropout([0.05, 0.15]),
                iaa.PiecewiseAffine(scale=(0.0, 0.03)), 
                iaa.PerspectiveTransform(scale=(0.0, 0.1))
            ], random_order=True)
        ])

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        self.resize_label = transforms.Resize((self.img_height//self.output_scale,
            self.img_width//self.output_scale))

    def __getitem__(self, index):
        seq_det = self.aug_seq.to_deterministic()

        color = Image.open(os.path.join(self.path, 'color-input', self.sample_list[index] + '.png'))
        depth = Image.open(os.path.join(self.path, 'depth-input', self.sample_list[index] + '.png'))
        label = Image.open(os.path.join(self.path, 'label', self.sample_list[index] + '.png'))
        
        if self.mode == 'train':
            color_img = np.asarray(color, dtype=np.float32) / 255
            color_img = np.clip(seq_det.augment_image(color_img), 0.0, 1.0)
            color_img = self.normalize(self.to_tensor(color_img.copy()))

            depth_img = (np.asarray(depth, dtype=np.float64) * 65536/10000).astype(np.float32)
            depth_img = np.array([depth_img, depth_img, depth_img])
            depth_img = np.transpose(depth_img, (1, 2, 0))
            depth_img = seq_det.augment_image(depth_img)
            depth_img = self.normalize(self.to_tensor(depth_img.copy()).clamp(0.0, 1.2))
            
            label = (np.asarray(label, dtype=np.float32) * 2 / 255).astype(np.uint8)
            label_segmap = ia.SegmentationMapOnImage(label, shape=color_img.shape, nb_classes=3)
            label_segmap = seq_det.augment_segmentation_maps([label_segmap])[0]
            label_img = Image.fromarray((label_segmap.get_arr_int() * 255/2).astype(np.uint8))
            label_img = self.to_tensor(self.resize_label(label_img.copy()))
            label_img = torch.round(label_img*2).long()
            label_img = label_img.view(self.img_height//self.output_scale, -1)

        elif self.mode == 'val':
            color_img = self.normalize(self.to_tensor(color))
            
            depth_img = np.asarray(depth, dtype=np.float32)
            depth_img = (self.to_tensor(depth_img) * 65536/10000).clamp(0.0, 1.2)
            depth_img = torch.cat([depth_img, depth_img, depth_img], 0)
            depth_img = self.normalize(depth_img)

            label_img = self.to_tensor(self.resize_label(label))
            label_img = torch.round(label_img*2).long() # set to segmentation map value
            label_img = label_img.view(self.img_height//self.output_scale, -1)
        return [color_img, depth_img], label_img

    def __len__(self):
        return self.num_samples


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
        self.resize_label = transforms.Resize((self.img_height//self.output_scale,
            self.img_width//self.output_scale))

    def __getitem__(self, index):
        color_img = self.to_tensor(Image.open(os.path.join(self.path, 'color-input', self.sample_list[index] + '.png')))
        color_img = self.normalize(color_img)
        
        depth = Image.open(os.path.join(self.path, 'depth-input', self.sample_list[index] + '.png'))
        depth = (self.to_tensor(np.asarray(depth, dtype=np.float32)) * 65536/10000).clamp(0.0, 1.2)
        depth_img = torch.cat([depth, depth, depth], 0)
        depth_img = self.normalize(depth_img)

        label = self.resize_label(Image.open(os.path.join(self.path, 'label', self.sample_list[index] + '.png')))
        label = self.to_tensor(label)
        label = torch.round(label*2).long() # set to label value, then cast to long int
        label = label.view(self.img_height//self.output_scale, -1)

        return [color_img, depth_img], label

    def __len__(self):
        return self.num_samples


## just for testing
if __name__ == "__main__":
    class Options:
        def __init__(self):
            data_path = '/home/tri/skripsi/dataset/'
            sample_path = '/home/tri/skripsi/dataset/test-split.txt'
            img_height =  480
            img_width = 640
            batch_size = 4
            n_class = 3
            output_scale = 8
            shuffle = True
            learning_rate = 0.001

    options = Options()
    data_path = '/home/tri/skripsi/dataset/'
    sample_path = '/home/tri/skripsi/dataset/test-split.txt'

    # testing functionality
    suction_dataset = SuctionDataset(options, data_path=data_path, sample_list=sample_path)
    rgbd, label = suction_dataset[1]
    a, b = suction_dataset[2]
