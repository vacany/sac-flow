import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from pytorch3d.ops.knn import knn_points
from torch_scatter import scatter

# todo iterative This is nice!
# todo time in dbscan - extend dimension in voxel for time
# todo cuml dbscan


def center_rigidity_loss(pc1, flow, cluster_ids):
    '''
    For batch size of 1
    :param pc1:
    :param flow:
    :param cluster_ids:
    :return:
    '''
    pts_centers = scatter(pc1, cluster_ids, dim=1, reduce='mean')
    flow_centers = scatter(pc1 + flow, cluster_ids, dim=1, reduce='mean')

    pt_dist_to_center = (pc1 - pts_centers[0, cluster_ids[0]].unsqueeze(0))  # .norm(dim=-1, p=1)
    flow_dist_to_center = ((pc1 + flow) - flow_centers[0, cluster_ids[0]].unsqueeze(0))  # .norm(dim=-1, p=1)

    center_displacement = pt_dist_to_center - flow_dist_to_center

    rigidity_loss = center_displacement.norm(dim=-1).mean()

    return rigidity_loss


# construct flow rays

def construct_ray_flows(flow, max_magnitude=3, eps=0.15):
    if max_magnitude == None:
        max_magnitude = flow.norm(dim=-1).max()

    else:
        max_magnitude = torch.tensor(max_magnitude, device=flow.device)

    in_between = torch.ceil(max_magnitude / eps).long()

    ray_flow = flow.repeat(in_between, 1, 1)

    indices = torch.arange(0, end=in_between, step=1, device=flow.device).unsqueeze(-1).unsqueeze(-1) / in_between
    ray_flow_points = ray_flow * indices

    return ray_flow_points


def downsampled_clustering(deformed_pts, eps=0.15):
    pc_to_downsample = deformed_pts.view(-1, 3)

    cell_size = torch.tensor(eps, device=deformed_pts.device)

    # this is fixed anyway
    max_range = deformed_pts.view(-1, 3).max(dim=0)[0]
    min_range = deformed_pts.view(-1, 3).min(dim=0)[0]

    size = ((max_range - min_range) / cell_size).long() + 2
    # origin_coors = (- min_range / cell_size).long()
    voxel_grid = torch.zeros((size[0], size[1], size[2]), dtype=torch.long, device=pc_to_downsample.device)
    index_grid = voxel_grid.clone()

    grid_coors = ((pc_to_downsample - min_range) / cell_size).long()

    voxel_grid[grid_coors[:, 0], grid_coors[:, 1], grid_coors[:, 2]] = 1

    upsampled_pts = voxel_grid.nonzero()

    upsample_ids = DBSCAN(eps=eps, min_samples=1).fit_predict((upsampled_pts * (eps - 0.01)).detach().cpu().numpy())

    index_grid[upsampled_pts[:, 0], upsampled_pts[:, 1], upsampled_pts[:, 2]] = torch.from_numpy(upsample_ids).to(
        index_grid.device).long()

    upsampled_pts_ids = index_grid[grid_coors[:, 0], grid_coors[:, 1], grid_coors[:, 2]]

    return upsampled_pts_ids


def gather_flow_ids(pc_with_flow, flow, eps):
    ray_flow_points = construct_ray_flows(flow, max_magnitude=3, eps=eps)
    deformed_pts = ray_flow_points + pc_with_flow

    deformed_ids = downsampled_clustering(deformed_pts, eps=eps)
    flow_ids = deformed_ids.view(-1, flow.shape[1])[:1]  # to get back from shifting, bad practice, but works

    return flow_ids


def smooth_cluster_ids(flow, flow_ids):
    # todo check it
    if len(flow) != 1:
        print('Batch Size not yet implemented!')
    mean_id_flow = scatter(flow[:, :], flow_ids, dim=1, reduce='mean')
    smooth_flow_loss = (flow[0, :] - mean_id_flow[0][flow_ids][0]).norm(dim=-1)

    return smooth_flow_loss


def fit_motion_svd_batch(pc1, pc2, mask=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R
        t_base[valid_batches] = t

    return R_base, t_base


class InstanceSmoothnessLoss(torch.nn.Module):

    def __init__(self, pc, K=8, max_radius=1, loss_norm=1):
        super().__init__()
        self.K = K
        self.max_radius = max_radius
        self.loss_norm = loss_norm

        self.dist, self.nn_ind, _ = knn_points(pc, pc, K=K)
        tmp_idx = self.nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(self.nn_ind.device)
        self.nn_ind[self.dist > max_radius] = tmp_idx[self.dist > max_radius]

    def forward(self, mask):

        out = mask[0][self.nn_ind[0]]
        out = out.permute(0, 2, 1)
        out = out.unsqueeze(0)

        # norm for each of N separately
        per_point_smooth_loss = (mask.unsqueeze(3) - out).norm(p=self.loss_norm, dim=2)
        smooth_loss = per_point_smooth_loss.mean()

        return smooth_loss, per_point_smooth_loss
class DynamicLoss(nn.Module):
    """
    Enforce the rigid transformation estimated from object masks to explain the per-point flow.
    """
    def __init__(self, loss_norm=2):
        super().__init__()
        self.loss_norm = loss_norm

    def forward(self, pc, mask, flow):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param mask: (B, N, K) torch.Tensor.
        :param flow: (B, N, 3) torch.Tensor.
        :return:
            loss: () torch.Tensor.
        """
        n_batch, n_point, n_object = mask.size()
        pc2 = pc + flow
        mask = mask.transpose(1, 2).reshape(n_batch * n_object, n_point)
        pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
        pc2_rep = pc2.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

        # Estimate the rigid transformation
        object_R, object_t = fit_motion_svd_batch(pc_rep, pc2_rep, mask)

        # Apply the estimated rigid transformation onto point cloud
        pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3).detach()
        mask = mask.reshape(n_batch, n_object, n_point)

        # Measure the discrepancy of per-point flow
        mask = mask.unsqueeze(-1)
        pc_transformed = (mask * pc_transformed).sum(1)
        loss = (pc_transformed - pc2).norm(p=self.loss_norm, dim=-1)
        return loss.mean()
