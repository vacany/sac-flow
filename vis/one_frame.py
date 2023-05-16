from vis.deprecated_vis import *
import sys


if 'pycharm' in sys.argv[0]:
    frame = 162
    path = f'/home/patrik/rci/experiments/scoop_K32_normals/pc_res/{frame:06d}_res.npz'
else:
    frame = int(sys.argv[1])
    path = f'/home/patrik/rci/experiments/scoop_{sys.argv[2]}/pc_res/{frame:06d}_res.npz'

data = np.load(path, allow_pickle=True)

print(data.files)

# visualize_points3D(data['pc1'], np.arange(len(data['pc1'])))

# visualize_flow3d(data['pc1'], data['pc2'], data['est_flow_for_pc1'])

# visualize_KNN_connections(data['pc1'], data['KNN'])

import mayavi.mlab as mlab

pc1 = data['pc1']
pc2 = data['pc2']
flow = data['est_flow_for_pc1']
normals = data['normals'][0]



# import matplotlib.pyplot as plt
# plt.scatter(pc1[:,0], pc1[:, 2], c=pc1[:, 2])
# plt.show()

mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0,0,1), scale_factor=0.02)
mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(1,0,0), scale_factor=0.02)
mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], flow[:, 0], flow[:, 1], flow[:, 2], color=(0,1,0), line_width=0.1, mode='arrow', scale_factor=1)
mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], color=(1,1,1), line_width=0.1, mode='arrow', scale_factor=1)
# mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], corrected_normals[:, 0], corrected_normals[:, 1], corrected_normals[:, 2], color=(1,1,1), line_width=0.1, mode='arrow', scale_factor=1)
# mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], direction_to_origin[:, 0], direction_to_origin[:, 1], direction_to_origin[:, 2], color=(0.5,0.5,0.5), line_width=0.1, mode='arrow', scale_factor=1)


knn = data['KNN']


mlab.show()
