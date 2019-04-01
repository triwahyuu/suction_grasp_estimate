from PIL import Image
import numpy as np
from open3d import open3d
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.filters import uniform_filter

## convert boolean numpy array to image
def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

## local standard deviation filter for 2d rgb (numpy) image
def stdev_filter(im, window_size):
    res = np.zeros(im.shape)
    for i in range(3):
        x = im[:,:,i]
        c1 = uniform_filter(x, window_size, mode='reflect')
        c2 = uniform_filter(x*x, window_size, mode='reflect')
        res[:,:,i] = np.sqrt(abs(c2 - c1*c1))
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
    cam_x = (pix_x - cam_intrinsic[0][2]) * depth_input/cam_intrinsic[0][0]
    cam_y = (pix_y - cam_intrinsic[1][2]) * depth_input/cam_intrinsic[1][1]
    cam_z = depth_input

    depth_valid = (np.logical_and(foreground_mask, cam_z) != 0)
    input_points = np.array([cam_x[depth_valid], cam_y[depth_valid], cam_z[depth_valid]]).transpose()

    ## get the foreground point cloud
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(input_points)
    open3d.estimate_normals(pcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 50))
    pcd_normals = np.asarray(pcd.normals)

    ## flip normals to point towards sensor
    center = [0,0,0]
    for k in range(input_points.shape[0]):
        p1 = center - input_points[k][:]
        p2 = pcd_normals[k][:]
        angle = np.arctan2(np.linalg.norm(np.cross(p1,p2)), np.dot(p1, p2.transpose()))
        if (angle > -np.pi/2 and angle < np.pi/2):
            pcd_normals[k][:] = -pcd_normals[k][:]

    ## reproject the normals back to image plane
    pix_x = np.round((input_points[:,0] * cam_intrinsic[0][0] / input_points[:,2] + cam_intrinsic[0][2]))
    pix_y = np.round((input_points[:,1] * cam_intrinsic[1][1] / input_points[:,2] + cam_intrinsic[1][2]))

    surface_normals_map = np.zeros(color_input.shape)
    n = 0
    for x,y in zip(pix_x, pix_y):
        x,y = int(x), int(y)
        surface_normals_map[y,x,0] = pcd_normals[n,0]
        surface_normals_map[y,x,1] = pcd_normals[n,1]
        surface_normals_map[y,x,2] = pcd_normals[n,2]
        n += 1
        
    ## Compute standard deviation of local normals
    mean_std_norms = np.zeros(color_input.shape, dtype=np.float64)
    mean_std_norms = np.mean(stdev_filter(surface_normals_map, 25), axis=2)
    affordance_map = 1 - mean_std_norms/mean_std_norms.max()
    affordance_map[~depth_valid] = 0

    return surface_normals_map, affordance_map

if __name__ == "__main__":
    data_path = '/home/tri/skripsi/dataset/'
    filename = '000001-1'

    rgb_in = Image.open(data_path + "/color-input/" + filename + ".png")
    rgb_bg = Image.open(data_path + "/color-background/" + filename + ".png")
    depth_in = Image.open(data_path + "/depth-input/" + filename + ".png")
    depth_bg = Image.open(data_path + "/depth-background/" + filename + ".png")
    cam_file = open(data_path + 'camera-intrinsics/' + filename + '.txt', 'r')
    cam_intrinsic = [[float(a) for a in line.split('\t')[:3]] for line in cam_file.readlines()[:3]]

    s, a = predict(rgb_in, rgb_bg, depth_in, depth_bg, cam_intrinsic)
    snm = Image.fromarray((s*255).astype(np.uint8))
    snm.show()
    aff_map = Image.fromarray((a*255).astype(np.uint8))
    aff_map.show()

