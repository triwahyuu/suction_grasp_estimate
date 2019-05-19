import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter

points = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])

P = [[2.06498904e-01 , -6.30755443e-07 ,  1.07477548e-03],
 [1.61535574e-06 ,  1.18897198e-01 ,  7.85307721e-06],
 [7.08353661e-02 ,  4.48415767e-06 ,  2.05395893e-01]]

Z = np.zeros((8,3))
for i in range(8): 
    Z[i,:] = np.dot(points[i,:],P)
Z = 10.0*Z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r = [-1,1]

X, Y = np.meshgrid(r, r)
# plot vertices
ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

# list of sides' polygons of figure
verts = [[Z[0],Z[1],Z[2],Z[3]],
 [Z[4],Z[5],Z[6],Z[7]], 
 [Z[0],Z[1],Z[5],Z[4]], 
 [Z[2],Z[3],Z[7],Z[6]], 
 [Z[1],Z[2],Z[6],Z[5]],
 [Z[4],Z[7],Z[3],Z[0]]]

# adding images
ax.set_zlim(-2.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fn = get_sample_data("./lena.png", asfileobj=False)
arr = read_png(fn)
# 10 is equal length of x and y axises of your surface
stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

X1 = np.arange(-5, 5, stepX)
Y1 = np.arange(-5, 5, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
# stride args allows to determine image quality 
# stride = 1 work slow
ax.plot_surface(X1, Y1, -2.01, rstride=1, cstride=1, facecolors=arr)

# plot sides
ax.add_collection3d(Poly3DCollection(verts, facecolors='#A1A8C6', linewidths=1, edgecolors='#7B8CDE'))
ax.view_init(azim=-83, elev=22)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()