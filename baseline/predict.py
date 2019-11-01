from PIL import Image
import numpy as np
import tqdm

import argparse
import os.path as osp
import time

import open3d
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.filters import uniform_filter

import matplotlib.cbook
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

## convert boolean numpy array to image
def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

## local standard deviation filter for 2d rgb (numpy) image
def stdev_filter(im, window_size):
    r,c,_ = im.shape
    res = np.zeros(im.shape)
    for i in range(3):
        x = im[:,:,i] + np.random.rand(r,c)*1e-6
        c1 = uniform_filter(x, window_size, mode='reflect')
        c2 = uniform_filter(x*x, window_size, mode='reflect')
        res[:,:,i] = np.sqrt(c2 - c1*c1)
    return res

def calc_metric(prediction, target, threshold):
    sum_tp = np.sum(np.logical_and((prediction > threshold), (label_np == 128)).astype(np.int))
    sum_fp = np.sum(np.logical_and((prediction > threshold), (label_np == 0)).astype(np.int))
    sum_tn = np.sum(np.logical_and((prediction <= threshold), (label_np == 0)).astype(np.int))
    sum_fn = np.sum(np.logical_and((prediction <= threshold), (label_np == 128)).astype(np.int))
    return [sum_tp, sum_fp, sum_tn, sum_fn]

## predict the affordance map and surface normal map
## take the inputs of PIL image and the camera intrinsic matrix in numpy
## settings: [color_frg_threshold, depth_frg_threshold, normal_radius]
def predict(color_input, color_bg, depth_input, depth_bg, camera_intrinsic, settings=[0.3, 0.02, 0.1]):
    # print(time.time())
    ## scale the images to the proper value
    color_input = np.asarray(color_input, dtype=np.float32) / 255
    color_bg = np.asarray(color_bg, dtype=np.float32) / 255
    depth_input = np.asarray(depth_input, dtype=np.float64) / 10000
    depth_bg = np.asarray(depth_bg, dtype=np.float64) / 10000

    # print(time.time())
    ## get foreground mask
    frg_mask_color = ~(np.sum(abs(color_input-color_bg) < settings[0], axis=2) == 3)
    frg_mask_depth = np.logical_and((abs(depth_input-depth_bg) > settings[1]), (depth_bg != 0))
    foreground_mask = np.logical_or(frg_mask_color, frg_mask_depth)

    # print(time.time())
    ## project depth to camera space
    pix_x, pix_y = np.meshgrid(np.arange(640), np.arange(480))
    cam_x = (pix_x - camera_intrinsic[0][2]) * depth_input/camera_intrinsic[0][0]
    cam_y = (pix_y - camera_intrinsic[1][2]) * depth_input/camera_intrinsic[1][1]
    cam_z = depth_input

    depth_valid = (np.logical_and(foreground_mask, cam_z) != 0)
    input_points = np.array([cam_x[depth_valid], cam_y[depth_valid], cam_z[depth_valid]]).transpose()

    # print(time.time())
    ## get the foreground point cloud
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(input_points)
    open3d.estimate_normals(pcd, search_param=open3d.KDTreeSearchParamHybrid(radius=settings[2], max_nn=100))
    
    # print(time.time())
    ## flip normals to point towards sensor
    open3d.geometry.orient_normals_towards_camera_location(pcd, [0,0,0])
    pcd_normals = np.asarray(pcd.normals)

    # print(time.time())
    ## reproject the normals back to image plane
    pix_x = np.round((input_points[:,0] * camera_intrinsic[0][0] / input_points[:,2] + camera_intrinsic[0][2]))
    pix_y = np.round((input_points[:,1] * camera_intrinsic[1][1] / input_points[:,2] + camera_intrinsic[1][2]))

    surface_normals_map = np.zeros(color_input.shape)
    for n, (x,y) in enumerate(zip(pix_x, pix_y)):
        surface_normals_map[int(y),int(x)] = pcd_normals[n]
    
    # print(time.time())
    ## Compute standard deviation of local normals
    mean_std_norms = np.mean(stdev_filter(surface_normals_map, 25), axis=2)
    affordance_map = 1 - mean_std_norms/mean_std_norms.max()
    affordance_map[~depth_valid] = 0

    return surface_normals_map, affordance_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '--img-input', default='', help='input image index, eg: 00001-1',
    )
    args = parser.parse_args()
    np.random.seed(int(time.time()))

    p = osp.dirname(osp.abspath(__file__)).split('/')
    data_path = args.data_path if args.data_path != '' else osp.join('/'.join(p[:-2]), 'dataset/')
    result_path = osp.join('/'.join(p[:-2]), 'result/baseline/')

    sample_list = open(osp.join(data_path, 'test-split.txt')).read().splitlines()
    input_file = np.random.choice(sample_list, 1)[0] if args.img_input == '' else args.img_input

    ## load the datasets
    rgb_in = Image.open(data_path + 'color-input/' + input_file + '.png')
    rgb_bg = Image.open(data_path + "/color-background/" + input_file + ".png")
    depth_in = Image.open(data_path + "/depth-input/" + input_file + ".png")
    depth_bg = Image.open(data_path + "/depth-background/" + input_file + ".png")
    cam_intrinsic = np.loadtxt(data_path + 'camera-intrinsics/' + input_file + '.txt')

    ## get the suction affordance
    surf_norm, score = predict(rgb_in, rgb_bg, depth_in, depth_bg, cam_intrinsic)
    score_im = Image.fromarray((score*255).astype(np.uint8))
    score_im.save(osp.join(result_path, input_file + '.png'))

    ## Load ground truth manual annotations for suction affordances
    ## 0 - negative, 128 - positive, 255 - neutral (no loss)
    label = Image.open(data_path + 'label/' + input_file + '.png')
    label_np = np.asarray(label, dtype=np.uint8)

    ## Suction affordance threshold
    ## take the top 1% prediction
    # threshold = score.max() - 0.0001
    threshold = np.percentile(score, 99)
    score_norm = (score*255).astype(np.uint8)
    sum_tp = np.sum(np.logical_and((score > threshold), (label_np == 128)).astype(np.int))
    sum_fp = np.sum(np.logical_and((score > threshold), (label_np == 0)).astype(np.int))
    sum_tn = np.sum(np.logical_and((score <= threshold), (label_np == 0)).astype(np.int))
    sum_fn = np.sum(np.logical_and((score <= threshold), (label_np == 128)).astype(np.int))
    
    precision = sum_tp/(sum_tp + sum_fp) if (sum_tp + sum_fp) != 0 else 0
    recall = sum_tp/(sum_tp + sum_fn) if (sum_tp + sum_fp) != 0 else 0
    print("%s\t%.8f\t%.8f" % (input_file, precision, recall))

    ## visualize
    rgb_in_np = np.asarray(rgb_in, dtype=np.uint8)
    plt.subplot(1,3,1)
    plt.imshow(rgb_in_np)
    plt.yticks([]); plt.xticks([])
    plt.subplot(1,3,2)
    plt.imshow(score)
    plt.yticks([]); plt.xticks([])
    plt.subplot(1,3,3)
    plt.imshow(label_np)
    plt.yticks([]); plt.xticks([])
    fig = plt.gcf()
    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
    fig.canvas.set_window_title('Baseline Prediction Result')
    plt.show()
