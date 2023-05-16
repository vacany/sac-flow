import mayavi.mlab
import torch
import numpy as np
from data.PATHS import KITTI_SF_PATH, DATA_PATH
from loss.flow import *

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
h = 58.0 * 0.0254
pc1[:, 2] += h
# pc1 = data['pc1']

depth_pc = np.linalg.norm(pc1, axis=1)

HOF, VOF = 2048, 64
yaw = -np.arctan2(pc1[:,1], pc1[:,0])
pitch = np.arcsin(pc1[:,2] / (depth_pc + 1e-8))

delta_yaw = (yaw.max() - yaw.min()) / HOF
delta_pitch = (pitch.max() - pitch.min()) / VOF

H_coords = np.floor((yaw - yaw.min()) / delta_yaw).astype(int)
V_coords = np.floor((pitch - pitch.min()) / delta_pitch).astype(int)

H_coords = torch.from_numpy(H_coords).to(device)
V_coords = torch.from_numpy(V_coords).to(device)

torch.clip(H_coords, 0, HOF-1, out=H_coords)
torch.clip(V_coords, 0, VOF-1, out=V_coords)

depth = torch.zeros((VOF, HOF), device=device)

depth[V_coords, H_coords] = torch.from_numpy(depth_pc).to(device)

pc1 = torch.from_numpy(pc1).unsqueeze(0).to(device)
import socket

if socket.gethostname() != 'Patrik':

    from pytorch3d.ops.knn import knn_points
    from ops.visibility2D import KNN_visibility_solver, substitute_NN_by_mask

    # indices of KKN in depth image



    dist_nn, nn_ind, _ = knn_points(pc1, pc1, K=K)

    np.save('old_nn_ind.npy', nn_ind[0].detach().cpu().numpy())

    KNN_img_indices = torch.stack((V_coords[nn_ind[0]], H_coords[nn_ind[0]]), dim=2)
    valid_KNN = KNN_visibility_solver(KNN_img_indices, depth, margin=10)

    new_nn_ind = substitute_NN_by_mask(nn_ind[0], valid_KNN)

    # vis NN in mayavi
    np.save('new_nn_ind.npy', new_nn_ind.detach().cpu().numpy())


else:

    new_nn_ind = np.load('/home/patrik/cmp/pcflow/new_nn_ind.npy')
    old_nn_ind = np.load('/home/patrik/cmp/pcflow/old_nn_ind.npy')

    # from vis.deprecated_vis import *
    import mayavi
    # mayavi.mlab.options.offscreen = True
    pc_NN = pc1[0, new_nn_ind[..., 1:], :].detach().cpu()
    old_pc_NN = pc1[0, old_nn_ind[..., 1:], :].detach().cpu()

    fig = mayavi.mlab.figure(1)
    mayavi.mlab.points3d(pc1[0, :, 0], pc1[0, :, 1], pc1[0, :, 2], scale_factor=0.1, color=(0, 0, 1), figure=fig)



    for n in range(new_nn_ind.shape[1] - 1):
        mayavi.mlab.quiver3d(pc1[0, :, 0], pc1[0, :, 1], pc1[0, :, 2], old_pc_NN[:, n, 0] - pc1[0, :, 0], old_pc_NN[:, n, 1] - pc1[0, :, 1], old_pc_NN[:, n, 2]  - pc1[0, :, 2], scale_factor=0.9, color=(0, 1, 0), figure=fig)

    for n in range(new_nn_ind.shape[1] - 1):
        m = (pc_NN[:, n] == pc1[0]).all(1)

        mayavi.mlab.quiver3d(pc1[0, m, 0], pc1[0, m, 1], pc1[0, m, 2], old_pc_NN[m, n, 0] - pc1[0, m, 0], old_pc_NN[m, n, 1] - pc1[0, m, 1], old_pc_NN[m, n, 2]  - pc1[0, m, 2], scale_factor=0.9, color=(1, 0, 0), figure=fig)

    mayavi.mlab.show()



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
