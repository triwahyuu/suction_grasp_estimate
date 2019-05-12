## utilities code
from __future__ import division

import os
import math
import numpy as np

import open3d
from scipy.ndimage.filters import generic_filter, uniform_filter
from torchvision.transforms import ToTensor, Normalize, Resize
import torch


## prepare image input for inference
def prepare_input(color, depth, device):
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    color_img = normalize(to_tensor(color))
    color_img = color_img.view(1, color_img.size(0), color_img.size(1), color_img.size(2))

    depth = (to_tensor(np.asarray(depth, dtype=np.float32)) * 65536/10000).clamp(0.0, 1.2)
    depth_img = torch.cat([depth, depth, depth], 0)
    depth_img = normalize(depth_img)
    depth_img = depth_img.view(1, depth_img.size(0), depth_img.size(1), depth_img.size(2))

    img_input = [color_img.to(device), depth_img.to(device)]
    return img_input


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


## post process inference result
def post_process(affordance_map, class_pred,
        color_input, color_bg, 
        depth_input, depth_bg, camera_intrinsic):
    ## scale the images to the proper value
    color_input = np.asarray(color_input, dtype=np.float32) / 255
    color_bg = np.asarray(color_bg, dtype=np.float32) / 255
    depth_input = np.asarray(depth_input, dtype=np.float64) / 10000
    depth_bg = np.asarray(depth_bg, dtype=np.float64) / 10000

    ## get foreground mask
    frg_mask_color = ~(np.sum(abs(color_input-color_bg) < 0.3, axis=2) == 3)
    frg_mask_depth = np.logical_and((abs(depth_input-depth_bg) > 0.02), (depth_bg != 0))
    foreground_mask = np.logical_or(frg_mask_color, frg_mask_depth)

    ## project depth to camera space
    pix_x, pix_y = np.meshgrid(np.arange(640), np.arange(480))
    cam_x = (pix_x - camera_intrinsic[0][2]) * depth_input/camera_intrinsic[0][0]
    cam_y = (pix_y - camera_intrinsic[1][2]) * depth_input/camera_intrinsic[1][1]
    cam_z = depth_input

    depth_valid = (np.logical_and(foreground_mask, cam_z) != 0)
    input_points = np.array([cam_x[depth_valid], cam_y[depth_valid], cam_z[depth_valid]]).transpose()

    ## get the foreground point cloud
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(input_points)
    open3d.estimate_normals(pcd, search_param=open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 50))
    pcd_normals = np.asarray(pcd.normals)

    ## flip normals to point towards camera
    center = [0,0,0]
    for k in range(input_points.shape[0]):
        p1 = center - input_points[k][:]
        p2 = pcd_normals[k][:]
        x = np.cross(p1,p2)
        angle = np.arctan2(np.sqrt((x*x).sum()), p1.dot(p2.transpose()))
        if (angle > -np.pi/2 and angle < np.pi/2):
            pcd_normals[k][:] = -pcd_normals[k][:]

    ## reproject the normals back to image plane
    pix_x = np.round((input_points[:,0] * camera_intrinsic[0][0] / input_points[:,2] + camera_intrinsic[0][2]))
    pix_y = np.round((input_points[:,1] * camera_intrinsic[1][1] / input_points[:,2] + camera_intrinsic[1][2]))

    surface_normals_map = np.zeros(color_input.shape)
    n = 0
    for n, (x,y) in enumerate(zip(pix_x, pix_y)):
        x,y = int(x), int(y)
        surface_normals_map[y,x,0] = pcd_normals[n,0]
        surface_normals_map[y,x,1] = pcd_normals[n,1]
        surface_normals_map[y,x,2] = pcd_normals[n,2]
        
    ## Compute standard deviation of local normals (baseline)
    mean_std_norms = np.mean(stdev_filter(surface_normals_map, 25), axis=2)
    baseline_score = 1 - mean_std_norms/mean_std_norms.max()

    ## Set affordance to 0 for regions with high surface normal variance
    affordance_map[baseline_score < 0.1] = 0
    affordance_map[~foreground_mask] = 0
    class_pred[baseline_score < 0.1] = 0
    class_pred[~foreground_mask] = 0

    return surface_normals_map, affordance_map, class_pred


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

## get label acuracy score
## https://github.com/foolwood/deepmask-pytorch/blob/master/utils/
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':
    pass