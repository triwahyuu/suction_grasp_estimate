## Testing playground

import numpy as np
import torch
from torchvision import transforms
import imgaug as ia
from imgaug import augmenters as iaa
from dataset import SuctionDataset, SuctionDatasetNew
import matplotlib.pyplot as plt
import time


class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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
    suction_dataset = SuctionDatasetNew(options, data_path=options.data_path, sample_list=options.sample_path)

    rgbd, label = suction_dataset[0]
    rgbd1, label1 = suction_dataset[1]

    ia.seed(int(time.time()))

    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.15]),      # drop 5% or 20% of all pixels
        iaa.PiecewiseAffine(scale=(0.0, 0.025)),  # peacewise affine 0.01 to 0.05
        iaa.PerspectiveTransform(scale=(0.0, 0.1))  # perspective transform
    ], random_order=True)

    rgbds_aug = []
    labels_aug = []
    segmap = ia.SegmentationMapOnImage(label, shape=rgbd[0].shape, nb_classes=3)
    for _ in range(5):
        seq_det = seq.to_deterministic() # call on each new batch
        rgbds_aug.append([seq_det.augment_image(rgbd[0]), seq_det.augment_image(rgbd[1])])
        labels_aug.append(seq_det.augment_segmentation_maps([segmap])[0])

    fig, axes = plt.subplots(nrows=5, ncols=5)
    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
    fig.patch.set_visible(False)
    for i, (image_aug, segmap_aug) in enumerate(zip(rgbds_aug, labels_aug)):
        axes[i][0].imshow(rgbd[0])
        axes[i][0].set_axis_off()
        axes[i][1].imshow(image_aug[0])
        axes[i][1].set_axis_off()
        axes[i][2].imshow(label)
        axes[i][2].set_axis_off()
        axes[i][3].imshow(segmap_aug.draw(size=label.shape))
        axes[i][3].set_axis_off()
        axes[i][4].imshow(segmap_aug.draw_on_image(np.asarray(image_aug[0]*255, dtype=np.uint8)))
        axes[i][4].set_axis_off()
        label_aug = segmap_aug.get_arr_int()
    plt.show()

    # transforms = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     transforms.ColorJitter(hue=.05, saturation=.05),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])