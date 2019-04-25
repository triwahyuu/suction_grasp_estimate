from PIL import Image
import numpy as np
import open3d
from scipy.ndimage.filters import generic_filter, uniform_filter
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


## predict the affordance map and surface normal map
## take the inputs of PIL image and the camera intrinsic matrix in numpy
def visualize(affordance_map, surface_normals_map, color_input):
    affordance = gaussian_filter(affordance_map, 7)
    surface_normals_map = np.interp(surface_normals_map,
        (surface_normals_map.min(), surface_normals_map.max()), (0.0, 1.0))
    color_input = np.asarray(color_input, dtype=np.float64) / 255

    cmap = cm.get_cmap('jet')
    affordance = cmap(affordance)
    affordance = affordance[:,:,:-1] # ommit last channel (get rgb)
    img = affordance*0.5 + color_input*0.5

    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    ax1.imshow(img)
    ax1.set_axis_off()

    fig2, ax2 = plt.subplots()
    fig2.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    ax2.imshow(surface_normals_map)
    ax2.set_axis_off()

    plt.show()


def post_process(affordance_map, color_input, color_bg, depth_input, depth_bg, camera_intrinsic):
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

    return surface_normals_map, affordance_map


if __name__ == "__main__":
    pass
