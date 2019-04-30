from PIL import Image
import numpy as np
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

## predict the affordance map and surface normal map
## take the inputs of PIL image and the camera intrinsic matrix in numpy
def predict(color_input, color_bg, depth_input, depth_bg, camera_intrinsic):
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

    ## flip normals to point towards sensor
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
    pix_y = np.round((input_points[:,1] * camera_intrinsic[1][1] / input_points[:,2] + cam_intrinsic[1][2]))

    surface_normals_map = np.zeros(color_input.shape)
    for n, (x,y) in enumerate(zip(pix_x, pix_y)):
        x,y = int(x), int(y)
        surface_normals_map[y,x,0] = pcd_normals[n,0]
        surface_normals_map[y,x,1] = pcd_normals[n,1]
        surface_normals_map[y,x,2] = pcd_normals[n,2]
        
    ## Compute standard deviation of local normals
    mean_std_norms = np.mean(stdev_filter(surface_normals_map, 25), axis=2)
    affordance_map = 1 - mean_std_norms/mean_std_norms.max()
    affordance_map[~depth_valid] = 0

    return surface_normals_map, affordance_map

if __name__ == "__main__":
    data_path = '/home/tri/skripsi/dataset/'
    df = open(data_path + 'test-split.txt')
    data = df.read().splitlines()
    
    result = np.zeros((len(data), 4))
    plt.ion()
    plt.show()
    for n,fname in enumerate(data):
        print(fname, end='\t')

        ## load the datasets
        rgb_in = Image.open(data_path + 'color-input/' + fname + '.png')
        rgb_bg = Image.open(data_path + "/color-background/" + fname + ".png")
        depth_in = Image.open(data_path + "/depth-input/" + fname + ".png")
        depth_bg = Image.open(data_path + "/depth-background/" + fname + ".png")
        cam_intrinsic = np.loadtxt(data_path + 'camera-intrinsics/' + fname + '.txt')

        ## get the suction affordance
        surf_norm, score = predict(rgb_in, rgb_bg, depth_in, depth_bg, cam_intrinsic)
        score_im = Image.fromarray((score*255).astype(np.uint8))
        score_im.save(data_path + 'baseline/' + fname + '.png')

        ## Load ground truth manual annotations for suction affordances
        ## 0 - negative, 128 - positive, 255 - neutral (no loss)
        label = Image.open(data_path + 'label/' + fname + '.png')
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
        precision = sum_tp/(sum_tp + sum_fp)
        recall = sum_tp/(sum_tp + sum_fn)
        result[n,:] = [sum_tp, sum_fp, sum_tn, sum_fn]
        print("%.8f\t%.8f" % (precision, recall))

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
        plt.draw()
        plt.pause(0.01)
        
        # fig1, ax1 = plt.subplots()
        # fig1.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
        # ax1.imshow(rgb_in)
        # ax1.set_axis_off()

        # fig2, ax2 = plt.subplots()
        # fig2.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
        # ax2.imshow(score)
        # ax2.set_axis_off()
        # plt.show()
    
    ## save the result and calculate overal precision and recall
    np.savetxt('result.txt', result, fmt='%.10f')
    s = result.sum(axis=0)
    precision = s[0]/(s[0]+s[1])
    recall = s[0]/(s[0]+s[3])
    print(precision, recall)
