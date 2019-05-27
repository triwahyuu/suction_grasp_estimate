from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

import open3d
from scipy.ndimage.filters import generic_filter, uniform_filter


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
    open3d.geometry.estimate_normals(pcd,
        search_param=open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 50)
    )
    ## flip normals to point towards camera
    open3d.geometry.orient_normals_towards_camera_location(pcd, np.array([0.,0.,0.]))
    pcd_normals = np.asarray(pcd.normals)

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


## predict the affordance map and surface normal map
## take the inputs of PIL image and the camera intrinsic matrix in numpy
def visualize(affordance_map, surface_normals_map, class_pred, color_input):
    affordance_filt = gaussian_filter(affordance_map, 4)
    surface_normals_map = np.interp(surface_normals_map,
        (surface_normals_map.min(), surface_normals_map.max()), (0.0, 1.0))
    color_input = np.asarray(color_input, dtype=np.float64) / 255

    cmap = cm.get_cmap('jet')
    affordance = cmap(affordance_filt)[:,:,:-1] # ommit last channel (get rgb)
    img = affordance*0.5 + color_input*0.5

    cmap_cls = cm.get_cmap('Paired')
    cls_img = cmap_cls(class_pred)[:,:,:-1]
    cls_img = cls_img*0.5 + color_input*0.5

    ## best picking point
    max_point = np.argmax(affordance_filt)
    max_point = (max_point//affordance.shape[1], max_point%affordance.shape[1])
    max_circ = patches.Circle(np.flip(max_point), radius=8, fill=False, linewidth=4.0, color='k')

    ## class prediction
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    fig.canvas.set_window_title('Class Prediction')
    ax.imshow(cls_img)
    ax.set_axis_off()

    ## affordance map overlaid
    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    fig1.canvas.set_window_title('Affordance Map')
    ax1.imshow(img)
    ax1.add_patch(max_circ)
    ax1.set_axis_off()

    ## surface normal
    fig2, ax2 = plt.subplots()
    fig2.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    fig2.canvas.set_window_title('Surface Normals')
    ax2.imshow(surface_normals_map)
    ax2.set_axis_off()

    plt.show()


if __name__ == "__main__":
    pass
