import numpy as np
import matplotlib.pyplot as plt
import os

path = '/home/tri/skripsi/suction_grasp_estimate/result/'
ave_const = 250

list_dir = [a for a in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, a))]
y = np.array([], dtype=np.float64)
y_val = np.array([], dtype=np.float64)
for d in list_dir:
    p = os.path.join(path, d, 'log.csv')
    data = np.genfromtxt(p, delimiter=',', skip_header=1, dtype=np.float)
    y = np.append(y, np.array([a for a in data[:,5].astype(np.float64) if not np.isnan(a)]))
    y_val = np.append(y_val, np.array([b for b in data[:,10] if not np.isnan(b)]))

# get average
y_ave = np.array([np.mean(y[i*ave_const:(i+1)*ave_const]) for i in range(y.shape[0]//ave_const)])
x_ave = np.arange(y_ave.shape[0]).astype(np.int)
x_val = np.arange(y_val.shape[0]).astype(np.int)

# print(y.shape)
# print('minimum train loss %.10f at %d idx' % (y_ave.min(), y_ave.argmin()))
# print('minimum validation loss %.10f at %d idx' % (y_val.min(), y_val.argmin()))

plt.plot(x_ave, y_ave)
plt.title('Training Mean IU')
plt.show()

plt.plot(x_val, y_val)
plt.title('Validation Mean IU')
plt.show()