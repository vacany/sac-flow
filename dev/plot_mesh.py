import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import scipy as sp
from scipy import spatial as sp_spatial


def icosahedron():
    h = 0.5*(1+np.sqrt(5))
    p1 = np.array([[0, 1, h], [0, 1, -h], [0, -1, h], [0, -1, -h]])
    p2 = p1[:, [1, 2, 0]]
    p3 = p1[:, [2, 0, 1]]
    return np.vstack((p1, p2, p3))


def cube():
    points = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ])
    return points


# points = icosahedron()
# points = cube()
data_path = '/home/patrik/rci/data/kitti_sf/new/000000.npz'
data = np.load(data_path)

depth2 = data['depth2']
pc2 = data['pc2']
points = pc2[:100]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# define the vertices of the triangle
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# create a list of vertices for the face
vertices = [v1, v2, v3]

# create a list of vertex indices for the face
vertex_indices = [[0, 1, 2]]

# create a Poly3DCollection object from the vertices and vertex indices
face = Poly3DCollection([vertices], facecolor='blue', alpha=0.5)
face.set_verts(vertex_indices)

# create a 3D plot and add the face to it
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.add_collection3d(face)

# set the axis limits and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# show the plot
plt.show()
