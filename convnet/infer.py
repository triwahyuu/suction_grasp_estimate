## inference script 
from vis_util import post_process, visualize
import os
from model import SuctionModel18

import torch
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

class Struct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
options = Struct(\
    data_path = '/home/tri/skripsi/dataset/',
    model_path = '/home/tri/skripsi/suction_grasp_estimate/result/20190412_205337/checkpoint.pth.tar',
    img_height =  480,
    img_width = 640,
    output_scale = 8,
    arch = 'resnet18',
    device = 'cuda:0'
)

if __name__ == "__main__":
    ## transforms
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    ## get model
    model = SuctionModel18(options)
    checkpoint = torch.load(options.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(options.device)
    model.eval()

    ## get the image input
    color_img = to_tensor(Image.open(os.path.join(options.data_path, 'test', 'test-image.color.png')))
    color_img = normalize(color_img)
    color_img = color_img.view(1, color_img.size(0), color_img.size(1), color_img.size(2))

    depth = Image.open(os.path.join(options.data_path, 'test', 'test-image.depth.png'))
    depth = (to_tensor(np.asarray(depth, dtype=np.float32)) * 65536/10000).clamp(0.0, 1.2)
    depth_img = torch.cat([depth, depth, depth], 0)
    depth_img = normalize(depth_img)
    depth_img = depth_img.view(1, depth_img.size(0), depth_img.size(1), depth_img.size(2))

    img_input = [color_img.to(options.device), depth_img.to(options.device)]

    ## inference
    print('computing forward pass...')
    output = model(img_input)
    output = output.float().cpu().detach().numpy()
    output = output.reshape(output.shape[1:])
    output = np.transpose(output, (1, 2, 0))
    output = resize(output, (options.img_height, options.img_width), 
        anti_aliasing=True, mode='reflect')

    affordance = output[:,:,1]
    affordance[affordance >= 1] = 0.9999 # normalize
    affordance[affordance < 0] = 0

    ## visualize
    print('post process...')
    # rgb_in = Image.open(os.path.join(options.data_path, 'test', 'test-image.color.png'))
    # rgb_bg = Image.open(os.path.join(options.data_path, 'test', 'test-background.color.png'))
    # depth_in = Image.open(os.path.join(options.data_path, 'test', 'test-image.depth.png'))
    # depth_bg = Image.open(os.path.join(options.data_path, 'test', 'test-background.depth.png'))
    # cam_intrinsic = np.loadtxt(os.path.join(options.data_path, 'test', 'test-camera-intrinsics.txt'))
    rgb_in = Image.open(os.path.join(options.data_path, 'color-input', '000024-0.png'))
    rgb_bg = Image.open(os.path.join(options.data_path, 'color-background', '000024-0.png'))
    depth_in = Image.open(os.path.join(options.data_path, 'depth-input', '000024-0.png'))
    depth_bg = Image.open(os.path.join(options.data_path, 'depth-background', '000024-0.png'))
    cam_intrinsic = np.loadtxt(os.path.join(options.data_path, 'camera-intrinsics', '000024-0.txt'))

    surface_norm, affordance_map = post_process(affordance, rgb_in, rgb_bg,
        depth_in, depth_bg, cam_intrinsic)
    
    print('visualize...')
    visualize(affordance_map, surface_norm, np.array(rgb_in))