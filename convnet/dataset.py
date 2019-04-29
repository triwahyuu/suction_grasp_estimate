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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class SuctionDatasetNew(Dataset):
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
        self.iaa_resize = iaa.Resize(
            {"height": self.img_height//self.output_scale, 
            "width": self.img_width//self.output_scale}
        )

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        self.resize_label = transforms.Resize((self.img_height//self.output_scale,
            self.img_width//self.output_scale))

    def __getitem__(self, index):
        seq_det = self.aug_seq.to_deterministic()

        color = Image.open(os.path.join(self.path, 'color-input', self.sample_list[index] + '.png'))
        color_img = np.asarray(color, dtype=np.float32) / 255
        color_img = np.clip(seq_det.augment_image(color_img), 0.0, 1.0)
        color_img = self.normalize(self.to_tensor(color_img))

        depth = Image.open(os.path.join(self.path, 'depth-input', self.sample_list[index] + '.png'))
        depth = (np.asarray(depth, dtype=np.float64) * 65536/10000).astype(np.float32)
        depth_img = np.array([depth, depth, depth])
        depth_img = np.transpose(depth_img, (1, 2, 0))
        depth_img = seq_det.augment_image(depth_img)
        depth_img = self.normalize(self.to_tensor(depth_img).clamp(0.0, 1.2))

        label = Image.open(os.path.join(self.path, 'label', self.sample_list[index] + '.png'))
        label = (np.asarray(label, dtype=np.float32) * 2 / 255).astype(np.uint8)
        label_segmap = ia.SegmentationMapOnImage(label, shape=color_img.shape, nb_classes=3)
        label_segmap = seq_det.augment_segmentation_maps([label_segmap])[0]
        label_img = Image.fromarray((label_segmap.get_arr_int() * 255/2).astype(np.uint8))
        label_img = self.to_tensor(self.resize_label(label_img))
        label_img = torch.round(label_img*2).long()
        label_img = label_img.view(self.img_height//self.output_scale, -1)

        return [color_img, depth_img], label_segmap

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


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(
                device=dali_device, output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0], num_attempts=100)
        else:
            dali_device = "cuda"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(
                device="mixed", output_type=types.RGB, 
                device_memory_padding=211025920, host_memory_padding=140544512,
                random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0], num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
            device="cuda", output_dtype=types.FLOAT, output_layout=types.NCHW,
            crop=(crop, crop), image_type=types.RGB,
            mean=[0.485*255, 0.456*255, 0.406*255],
            std=[0.229*255, 0.224*255, 0.225*255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.cuda(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, random_shuffle=False)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="cuda", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
            device="cuda", output_dtype=types.FLOAT, output_layout=types.NCHW,
            crop=(crop, crop), image_type=types.RGB,
            mean=[0.485*255, 0.456*255, 0.406*255],
            std=[0.229*255, 0.224*255, 0.225*255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

class DatasetIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        random.shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(' ')
            f = open(self.images_dir + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__

## just for testing
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
    a, b = suction_dataset[2]
