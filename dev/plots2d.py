import mayavi.mlab
import torch
import numpy as np
from data.PATHS import KITTI_SF_PATH, DATA_PATH
from loss.flow import *

import matplotlib.pyplot as plt
def load_frame(frame_id):

    data_path = f"{KITTI_SF_PATH}/all_data_format/{frame_id:06d}.npz"
    data = np.load(data_path, allow_pickle=True)

    mask = np.ones(data['pc1'].shape[0], dtype=bool)

    valid_mask = data['valid_mask']
    image_indices = np.stack(valid_mask.nonzero()).T
    # image_indices = image_indices[pc_0_orig_valid_ma]

    data_dict = {'pc1': data['pc1'], 'pc2': data['pc2'], 'gt_flow': data['flow'], 'gt_mask': mask,
                 'inst_pc1' : data['inst_pc1'], 'inst_pc2' : data['inst_pc2'],
                 'depth1' : data['depth1'], 'depth2' : data['depth2'], 'image_indices': image_indices,
                 'frame_id': frame_id}

    return data_dict


# todo create lidar "depth" image

if torch.cuda.is_available():
    device = torch.device(6)
else:
    device = torch.device('cpu')

data = load_frame(0)

# params
K = 8

orig_pc1 = data['pc1']
depth1 = torch.from_numpy(data['depth1'])
image_indices = data['image_indices']
# pc1 = torch.from_numpy(data['pc1']).unsqueeze(0)

argo_data = np.load(DATA_PATH + '/argoverse/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/315968385323560000_315968385423756000.npz')
pc1 = argo_data['pc1']
pc2 = argo_data['pc2']
gt_flow = argo_data['flow']
h = 58.0 * 0.0254
pc1[:, 2] += h
# pc1 = data['pc1']

depth_pc = np.linalg.norm(pc1, axis=1)

# Dillatation/erosion?
# produce images 2D matplotlib
# - flow, GT flow, synchronization, Instance Segmentation, which flow is Hist? Subplots, later, maybe for Valeo/Void meeting PCA vectors on instance masks?
fig, axs = plt.subplots()
axs.plot(pc1[:,0], pc1[:,1], 'bo', markersize=0.1, alpha=0.6)
axs.plot(pc2[:,0], pc2[:,1], 'ro', markersize=0.1, alpha=0.2)
axs.quiver(pc1[:,0], pc1[:,1], gt_flow[:,0], gt_flow[:,1], color='g', alpha=0.2, scale=1, scale_units='xy')


# axs[0,0].plot(pc1[:,0], pc1[:,1], 'bo', markersize=0.1, alpha=0.6)
# axs[0,0].plot(pc2[:,0], pc2[:,1], 'ro', markersize=0.1, alpha=0.2)
# axs[0,0].quiver(pc1[:,0], pc1[:,1], gt_flow[:,0], gt_flow[:,1], color='g', alpha=0.2, scale=1, scale_units='xy')

# axs[0,1].plot(pc1[:,0], pc1[:,1], 'bo', markersize=0.1, alpha=0.7)
# axs[0,1].quiver(pc1[:,0], pc1[:,1], gt_flow[:,0], gt_flow[:,1], color='g', alpha=0.2, scale=1, scale_units='xy')

axs.set_title('ARGOVERSE FLOW')
# plt.show()

# from vis.deprecated_vis import *
import mayavi.mlab as mlab


fig = mlab.figure(1)
mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.1, color=(0, 0, 1), figure=fig)
mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], gt_flow[:, 0], gt_flow[:, 1], gt_flow[:, 2], scale_factor=1, color=(0, 1, 0), figure=fig)
mlab.show()

# todo connect horizontal 0 and 2048 together - two range images next to each other? shifted by half

# from vis.deprecated_vis import visualize_points3D
# visualize_points3D(pc1, H_coords, lookat=(0,0,h))
# visualize_points3D(pc1, V_coords)


### Losses
# pc1 = pc1[:, :100, :]
# est_flow = torch.rand(1, pc1.shape[1], 3)
#
# nn_dist, nn_ind, _ = knn_points(pc1, pc1, K=K)
#
# normals1 = estimate_pointcloud_normals(pc1, neighborhood_size=K)
#
# KNN_image_indices = torch.from_numpy(image_indices[nn_ind[0]])
# smooth_loss, smooot_per_point = smoothness_loss(est_flow, NN_idx=nn_ind)
#
# visibility_aware_smoothness_loss(est_flow, KNN_image_indices=KNN_image_indices, depth=depth1, NN_idx=nn_ind,  margin=1,)
#
# chamf_dist = chamfer_distance_loss(pc1 + est_flow, pc1)


#todo yev
