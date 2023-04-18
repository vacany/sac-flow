import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from vis.deprecated_vis import *


from data.gpu_utils import get_device
from pytorch3d.ops.knn import knn_points
from data.gpu_utils import get_device, print_gpu_memory

# data_path = '/home/patrik/rci/data/kitti_sf/new/000000.npz'
data_path = os.path.expanduser("~") + '/pcflow/toy_samples/000000.npz'
data = np.load(data_path)

depth2 = data['depth2']
pc2 = data['pc2']
valid_mask = data['valid_mask']
pc2_depth = depth2[valid_mask]



# image Neighboors without invalid
K = 32
NN_by_pts = np.zeros((pc2.shape[0], K), dtype=np.int32)

image_indices = np.stack(valid_mask.nonzero()).T
vis = depth2.copy()

device = torch.device("cuda:4")

    # get_device()

torch_pc2 = torch.from_numpy(pc2).to(device).unsqueeze(0)

out_nn = knn_points(torch_pc2, torch_pc2, K=K, return_nn=True)
nn_dist, nn_ind = out_nn[0][0], out_nn[1][0]

nn_ind = nn_ind.cpu().numpy()
nn_dist = nn_dist.cpu().numpy()

KNN_image_indices = image_indices[nn_ind]

chosen_NN = 2132


chos_ind = KNN_image_indices[chosen_NN]
vis[chos_ind[:,0], chos_ind[:,1]] = 100




from ops.rays import create_connecting_pts


chos_mask = np.zeros_like(nn_ind[:, 0])
chos_mask[nn_ind[chosen_NN]] = 1

# # vectorized version
direction_vector = KNN_image_indices - KNN_image_indices[:,0:1]
max_distance = np.max(direction_vector)
KNN_image_indices = torch.from_numpy(KNN_image_indices).to(device)
direction_vector = torch.from_numpy(direction_vector).to(device)
depth2 = torch.from_numpy(depth2).to(device)


valid_KNN = torch.ones(KNN_image_indices.shape[:2], dtype=bool, device=device)
# todo batch multiplication


margin = 0.05

vis_image = np.zeros_like(depth2.detach().cpu().numpy())

indices_list = []
depth_list = []
pts_list = []
chosen_K = 3

# main visibility solver
for increment in tqdm(torch.linspace(0, 1, max_distance, device=device)):

    parts = KNN_image_indices - direction_vector * increment

    parts = parts.to(torch.long)



    parts_idx = parts.view(-1, 2)
    curr_connection_depth = depth2[parts_idx[:,0], parts_idx[:,1]]
    origins_depth = depth2[parts[:,0,0], parts[:,0,1]]
    end_points_depth = depth2[parts[:,:,0], parts[:,:,1]]

    # podminka s koncovym bodem, jako linearni interpolace
    curr_linear_depth = origins_depth + (end_points_depth.permute(1,0) - origins_depth) * increment
    curr_linear_depth = curr_linear_depth.permute(1,0)

    # valid connection is the one where linear connection is before the point in image, i.e.
    # depth of connection is smaller than depth of point in image as it comes before
    valid_connection = curr_connection_depth.reshape(-1, K) <= curr_linear_depth + margin

    valid_KNN *= valid_connection

    # visuals

    indices_list.append(parts[chosen_NN, chosen_K].detach().cpu().numpy())
    depth_list.append(curr_connection_depth.reshape(-1,K)[chosen_NN, chosen_K].detach().cpu().numpy())


# one KNN visual in depth image
origin_pt = KNN_image_indices[chosen_NN, 0]
knn_pts = KNN_image_indices[chosen_NN, 1:]

# connect to origin
connections = []

for k in range(0, K - 1):

    px = torch.linspace(origin_pt[0], knn_pts[k,0], 300, device=device)
    py = torch.linspace(origin_pt[1], knn_pts[k,1], 300, device=device)

    # linear depth
    lin_d = depth2[origin_pt[0], origin_pt[1]] + (depth2[knn_pts[k,0], knn_pts[k,1]] - depth2[origin_pt[0], origin_pt[1]]) * torch.linspace(0, 1, 300, device=device)
    orig_d = depth2[px.to(torch.long), py.to(torch.long)]
    logic_d = lin_d <= orig_d
    px = px.to(torch.long)
    py = py.to(torch.long)
    connections.append(torch.stack([px, py, lin_d, orig_d, logic_d], dim=1))


fig, ax = plt.subplots(3,1, figsize=(10,10), dpi=200)
# vis_im = torch.zeros(depth2.shape, device=device)
vis_im = depth2.clone()
vis_im1 = depth2.clone()
vis_im2 = depth2.clone()

for con in connections:
    vis_im[con[:,0].long(), con[:,1].long()] = con[:,2]
    vis_im1[con[:,0].long(), con[:,1].long()] = con[:,3]
    vis_im2[con[:,0].long(), con[:,1].long()] = con[:,4] * 100

    vis_im[origin_pt[0], origin_pt[1]] = 100
    vis_im[knn_pts[:,0], knn_pts[:,1]] = 75

    vis_im1[origin_pt[0], origin_pt[1]] = 100
    vis_im1[knn_pts[:, 0], knn_pts[:, 1]] = 75

    vis_im2[origin_pt[0], origin_pt[1]] = 100
    vis_im2[knn_pts[:, 0], knn_pts[:, 1]] = 75

ax[0].imshow(vis_im.detach().cpu().numpy())
ax[1].imshow(vis_im1.detach().cpu().numpy())
ax[2].imshow(vis_im2.detach().cpu().numpy())
fig.savefig('toy_samples/out.png')



# visuals raycasting

from ops.rays import raycast_NN
r_, ind_ = raycast_NN(pc2, nn_ind, fill_pts=10)

# substitune NN
indentity_matrix = np.tile(np.arange(pc2.shape[0]), (K,1)).T
new_nn_ind = nn_ind.copy()

new_nn_ind[valid_KNN.detach().cpu().numpy() == False] = indentity_matrix[valid_KNN.detach().cpu().numpy() == False]

after_r_, after_ind_ = raycast_NN(pc2, new_nn_ind, fill_pts=10)


visualize_points3D(r_, ind_[:,1], lookat=[0,0,0])
visualize_points3D(after_r_, after_ind_[:,1])

from ops.metric import KNN_precision
instance_mask = data['inst_pc2']


KNN_precision(nn_ind, instance_mask=instance_mask)
KNN_precision(new_nn_ind, instance_mask=instance_mask)



# todo put functions to see it from local in matplotlib
