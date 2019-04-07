## Testing playground

from skimage import io
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

data_path = '/home/tri/skripsi/dataset/'
df = open(data_path + 'test-split.txt')
data = df.read().splitlines()

transform = transforms.Compose([transforms.ToTensor(), \
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)
fname = '000001-1'
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)

color_img = to_tensor(io.imread(data_path + 'color-input/' + fname + '.png'))
color_img_norm = normalize(color_img)
depth = to_tensor(io.imread(data_path + 'depth-input/' + fname + '.png').astype(np.float32)) * 65536/10000
depth = depth.clamp(0.0, 1.2)
depth_img = torch.cat([depth, depth, depth], 0)
depth_img_norm = normalize(depth_img)

label = to_tensor(io.imread(data_path + 'label/' + fname + '.png'))
label = torch.round(label*2 + 1)
