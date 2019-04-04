from PIL import Image
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

data_path = '/home/tri/skripsi/dataset/'
df = open(data_path + 'test-split.txt')
data = df.read().splitlines()

plt.ion()
plt.show()

for fname in data:
    rgb_in = Image.open(data_path + "/color-input/" + fname + ".png")
    rgb_bg = Image.open(data_path + "/color-background/" + fname + ".png")
    rgb_in_np = np.asarray(rgb_in, dtype=np.uint8)
    rgb_bg_np = np.asarray(rgb_bg, dtype=np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(rgb_bg_np)
    plt.subplot(1,2,2)
    plt.imshow(rgb_in_np)
    plt.draw()
    plt.pause(0.01)
    input("Press [enter] to continue.")
