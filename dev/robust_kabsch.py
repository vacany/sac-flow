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

class InstanceSegRefiner(torch.nn.Module):

    def __init__(self, x, max_instances=30):
        self.pred = torch.rand(x.shape[0], x.shape[1], max_instances, device=x.device, requires_grad=True)
        # self.pred = torch.zeros(x.shape[0], x.shape[1], max_instances, device=x.device)
        # self.pred[:, :, 0] = 1.
        # self.pred.requires_grad = True

        super().__init__()
        self.pred = torch.nn.Parameter(self.pred)

    def forward(self):
        return self.pred

class InstanceSmoothnessLoss(torch.nn.Module):

    def __init__(self, pc, K=8, max_radius=1):
        super().__init__()
        self.K = K
        self.max_radius = max_radius

        self.dist, self.nn_ind, _ = knn_points(pc, pc, K=K)
        tmp_idx = self.nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(self.nn_ind.device)
        self.nn_ind[self.dist > max_radius] = tmp_idx[self.dist > max_radius]

    def forward(self, mask):

        out = mask[0][self.nn_ind[0]]
        out = out.permute(0, 2, 1)
        out = out.unsqueeze(0)

        # norm for each of N separately
        per_point_smooth_loss = (mask.unsqueeze(3) - out).norm(p=loss_norm, dim=2)
        smooth_loss = per_point_smooth_loss.mean()

        return smooth_loss, per_point_smooth_loss

def find_robust_weighted_rigid_alignment(A, B, weights, use_epsilon_on_weights=False):
    """
    Calculates the weighted rigid transformation that aligns two sets of points.
    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.
        weights (torch.Tensor): A tensor of shaep (batch_size, num_points) containing weights.
        use_epsilon_on_weights (bool): A condition if to use eps for weights.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """
    # assert (weights >= 0.0).all(), "Negative weights found"
    # if use_epsilon_on_weights:
    #     weights += torch.finfo(weights.dtype).eps
    #     count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
    #     not_enough_points = count_nonzero_weighted_points < 3
    # else:
    #     # Add eps if not enough points with weight over zero
    #     count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
    #     not_enough_points = count_nonzero_weighted_points < 3
    #     eps = not_enough_points.float() * torch.finfo(weights.dtype).eps
    #     weights += eps.unsqueeze(-1)
    # assert not not_enough_points, f"pcl0 shape {A.shape}, pcl1 shape {B.shape}, points {count_nonzero_weighted_points}"

    weights = weights.unsqueeze(-1)
    sum_weights = torch.sum(weights, dim=1)

    A_weighted = A * weights
    B_weighted = B * weights

    a_mean = A_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)
    b_mean = B_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)

    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = ((A_c * weights).transpose(1, 2) @ B_c) / sum_weights
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(1, 2)
    # Translation vector
    t = b_mean.transpose(1, 2) - (R @ a_mean.transpose(1, 2))

    T = torch.cat((R, t), dim=2)
    T = torch.cat((T, torch.tensor([[[0,0,0,1]]], device=A.device)), dim=1)
    return T


if torch.cuda.is_available():
    device = torch.device(6)
else:
    device = torch.device('cpu')

data = load_frame(0)

# params
K = 12
max_radius = 1  # this is important for dummy case

orig_pc1 = data['pc1']
depth1 = torch.from_numpy(data['depth1'])
image_indices = data['image_indices']
# pc1 = torch.from_numpy(data['pc1']).unsqueeze(0)

argo_data = np.load(DATA_PATH + '/argoverse/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/315968385323560000_315968385423756000.npz')
pc1 = argo_data['pc1']
pc2 = argo_data['pc2']
gt_flow = argo_data['flow']


pc1 = torch.from_numpy(pc1).to(device).unsqueeze(0)
mask = pc1.norm(dim=2) < 35
pc1 = pc1[mask].unsqueeze(0)
gt_flow = torch.from_numpy(gt_flow).to(device).unsqueeze(0)[mask].unsqueeze(0)
# w = torch.rand(pc1.shape[1], device=device).unsqueeze(0)
# w.requires_grad = True


model = InstanceSegRefiner(pc1, max_instances=100)
mask = model.pred
min_nbr_pts = 5
loss_norm = 1
# optimizer = torch.optim.Adam([mask], lr=0.1)

# ~ diffentiable DBSCAN
# min distance from clusters

    # still not batch-wise


# w = mask[:, :, 0]
# optimizer = torch.optim.Adam([mask], lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# exp 1: just smoothness loss
# IS_smooth_loss = InstanceSmoothnessLoss(pc1, K=K, max_radius=max_radius)

# exp 2: flow in smoothness loss

IS_smooth_loss = InstanceSmoothnessLoss(torch.cat([pc1, gt_flow], dim=2), K=K, max_radius=max_radius)
# When RMSD is zero, it should be static (no motion). Pseudo label

# flow do instance seg?

for e in range(500):

    mask = model()

    # w_hard = w.max(dim=1, keepdim=True)[1]

    # ret = w_hard - w.detach() + w


    mask = torch.softmax(mask, dim=2)
    # mask = torch.sigmoid(mask)
    w = mask[:, :, 0]  # how to init to calculate Kabsch from all in the beginning?

    w_hard = (mask.max(dim=2, keepdim=True)[1] == 0).to(torch.float)[:,:,0]
    kabsch_w = w_hard - w.detach() + w

    # print(kabsch_w.mean())

    transform = find_robust_weighted_rigid_alignment(pc1, pc1+gt_flow, weights=kabsch_w)

    to_transform_pc1 = torch.cat((pc1, torch.ones_like(pc1[..., :1])), dim=2)
    # FAKIN Transformace!!!
    sync_pc1 = torch.bmm(to_transform_pc1, transform.transpose(2,1))[..., :3]


    difference = sync_pc1 - (pc1+gt_flow)
    rmsd = difference.norm(dim=2)

    # pseudo_label
    static_pseudo_label = rmsd < 0.05

    # Smoothness IS
    smooth_loss, per_point_smooth_loss = IS_smooth_loss(mask)

    loss = rmsd + smooth_loss

    loss.mean().backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Iter: {e:03d} \t Loss: {loss.mean().item():.4f} \t RMSD: {rmsd.mean().item():.4f} \t Smoothness: {smooth_loss.mean().item():.4f} \t Kabsch_w: {kabsch_w.mean().item():.4f}")



from vis.deprecated_vis import *




visualize_multiple_pcls(*[pc1[0].detach().cpu().numpy(), sync_pc1[0].detach().cpu().numpy(), pc2])

# verification of transform visual
# vis_pc1 = np.insert(pc1[0].detach().cpu().numpy(), 3, 1, axis=1)
# vis_trans = transform[0].detach().cpu().numpy()
# vis_sync_pc1 = (vis_pc1 @ vis_trans.T)[:,:3]
# visualize_multiple_pcls(*[pc1[0].detach().cpu().numpy(), vis_sync_pc1, pc2])

instance_classes = torch.argmax(mask, dim=2).squeeze(0).detach().cpu().numpy()
visualize_points3D(pc1[0].detach().cpu().numpy(), instance_classes)
visualize_points3D(pc1[0].detach().cpu().numpy(), instance_classes != 0)



# visualize_points3D(pc1[0].detach().cpu().numpy(), instance_classes == 0)
# visualize_points3D(pc1[0].detach().cpu().numpy(), w[0].detach().cpu().numpy())

# visualize_points3D(pc1[0].detach().cpu().numpy(), w[0].detach().cpu().numpy() < 3)
visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2, gt_flow[0].detach().cpu().numpy())
# import mayavi.mlab as mlab
# fig = mlab.figure(1)
# mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], scale_factor=0.1, color=(0, 0, 1), figure=fig)
# mlab.quiver3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], gt_flow[:, 0], gt_flow[:, 1], gt_flow[:, 2], scale_factor=1, color=(0, 1, 0), figure=fig)
# mlab.show()

