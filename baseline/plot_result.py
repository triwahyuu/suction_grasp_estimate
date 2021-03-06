import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

p = osp.dirname(osp.abspath(__file__))

radexp_list = open(osp.join(p, "experiment-rad.txt")).read().splitlines()[1:]
radexp_list = list(map(lambda x: x.split(','), radexp_list))
radexp_np = np.asarray(radexp_list)

time_data = list(map(float, radexp_np[:,3]))
prec_rad_data = np.array(list(map(float, radexp_np[:,1])))*100
recall_rad_data = np.array(list(map(float, radexp_np[:,2])))*100
rad = np.array(list(map(float, radexp_np[:,6])))
print(np.average(time_data))


thresexp_list = open(osp.join(p, "experiment-thres.txt")).read().splitlines()[1:]
thresexp_list = list(map(lambda x: x.split(','), thresexp_list))
thresexp_np = np.asarray(thresexp_list)

time_data = list(map(float, thresexp_np[:,3]))
prec_thres_data = np.array(list(map(float, thresexp_np[:,1])))*100
recall_thres_data = np.array(list(map(float, thresexp_np[:,2])))*100
thres_col = np.array(list(map(float, thresexp_np[:,4])))
thres_dep = np.array(list(map(float, thresexp_np[:,5])))
max_idx = np.argmax(prec_thres_data)
print("(%.2f %.2f): %.8f" % (thres_col[max_idx], thres_dep[max_idx], np.max(prec_thres_data)))


plt.plot(rad, prec_rad_data)
plt.title("The Effect of Local Surface Normal Radius on Precision")
plt.xticks(rad)
plt.yticks(np.arange(56.5, 57.2, 0.1))
plt.xlabel("Radius (cm)")
plt.ylabel("Precision (%)")
fig0 = plt.gcf()
fig0.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.1)
# plt.savefig("result/plt.png")


y_prec_dep = prec_thres_data[thres_col == 0.25]
x_thres_dep = thres_dep[thres_col == 0.25]
fig1 = plt.figure()
fig1.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.1)
plt.plot(x_thres_dep, y_prec_dep)
plt.xticks(x_thres_dep)
plt.yticks(np.arange(56.0, 57.4, 0.2))
plt.title("The Effect of Depth Threshold on Precision")
plt.xlabel("Threshold")
plt.ylabel("Precision (%)")


y_prec_col = prec_thres_data[thres_dep == 0.11]
x_thres_col = thres_col[thres_dep == 0.11]
fig2 = plt.figure()
fig2.subplots_adjust(top=0.95, bottom=0.1, right=0.99, left=0.1)
plt.plot(x_thres_col, y_prec_col)
plt.xticks(x_thres_col)
# plt.yticks(np.arange(56.0, 57.4, 0.2))
plt.title("The Effect of Color Threshold on Precision")
plt.xlabel("Threshold")
plt.ylabel("Precision (%)")


fig3 = plt.figure()
fig3.subplots_adjust(top=1.02, bottom=-0.05, right=1.1, left=-0.2)
ax = plt.axes(projection='3d')
ax.set_title("The Effect of Threshold Value on Precision")
for v in np.arange(0.15, 0.50, 0.05):
    v = round(v, 2)
    ax.plot3D(thres_col[thres_col == v], thres_dep[thres_col == v], prec_thres_data[thres_col == v])
ax.set_xlabel('color threshold')
ax.set_ylabel('depth threshold')
ax.set_zlabel('precision')
ax.view_init(elev=15, azim=-76)
# ax.set_aspect('equal')
plt.show()