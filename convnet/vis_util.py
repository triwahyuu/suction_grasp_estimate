from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

## convert boolean numpy array to image
def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


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
