import torch
import sys
import argparse
import importlib
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals

from data.range_image import VisibilityScene
# import FastGeodis

from .visibility import KNN_visibility_solver, substitute_NN_by_mask, strip_KNN_with_vis


def chamfer_distance_loss(x, y, x_lengths=None, y_lengths=None, both_ways=False, normals_K=0, loss_norm=1):
    '''
    Unique Nearest Neightboors?
    :param x:
    :param y:
    :param x_lengths:
    :param y_lengths:
    :param reduction:
    :return:
    '''
    if normals_K >= 3:
        normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_K)
        normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_K)

        x = torch.cat([x, normals1], dim=-1)
        y = torch.cat([y, normals2], dim=-1)


    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=loss_norm)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    x_nearest_to_y = x_nn[1]

    if both_ways:
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=loss_norm)
        cham_y = y_nn.dists[..., 0]  # (N, P2)
        # y_nearest_to_x = y_nn[1]

        nn_loss = (cham_x.mean() + cham_y.mean() ) / 2 # different shapes

    else:
        nn_loss = cham_x.mean()

    return nn_loss, cham_x, x_nearest_to_y




class DT:
    def __init__(self, pc1, pc2, grid_factor=10):
        ''' works for batch size 1 only - modification to FNSFP'''
        self.grid_factor = grid_factor
        pts = pc2[0]

        pc1_min = torch.min(pc1.squeeze(0), 0)[0]
        pc2_min = torch.min(pc2.squeeze(0), 0)[0]
        pc1_max = torch.max(pc1.squeeze(0), 0)[0]
        pc2_max = torch.max(pc2.squeeze(0), 0)[0]

        xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min < pc2_min, pc1_min, pc2_min) * 10 - 1) / 10
        xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max > pc2_max, pc1_max, pc2_max) * 10 + 1) / 10

        pmin = (xmin_int, ymin_int, zmin_int)
        pmax = (xmax_int, ymax_int, zmax_int)

        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2

        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=pts.device)[:-1] / grid_factor + pmin[0]
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=pts.device)[:-1] / grid_factor + pmin[1]
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=pts.device)[:-1] / grid_factor + pmin[2]

        # NOTE: build a binary image first, with 0-value occuppied points
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=pts.device)
        self.pts_sample_idx_x = ((pts[:,0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:,1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:,2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.

        iterations = 1
        image_pts = torch.zeros(H, W, D, device=pts.device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
        self.D = FastGeodis.generalised_geodesic3d(
            image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, 0.0, iterations
        ).squeeze()

    def torch_bilinear_distance(self, pc_deformed):

        pc_deformed = pc_deformed.squeeze(0)

        H, W, D = self.D.size()
        target = self.D[None, None, ...]

        sample_x = ((pc_deformed[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((pc_deformed[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((pc_deformed[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)

        sample = torch.cat([sample_x, sample_y, sample_z], -1)

        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1

        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)

        # NOTE: reshape to match 5D volumetric input
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)


        return dist.mean(), dist



def _smoothness_loss(est_flow, NN_idx, loss_norm=1, mask=None):

    bs, n, c = est_flow.shape

    if bs > 1:
        print("Smoothness Maybe not working, needs testing!")
    K = NN_idx.shape[2]

    est_flow_neigh = est_flow.view(bs * n, c)
    est_flow_neigh = est_flow_neigh[NN_idx.view(bs * n, K)]

    est_flow_neigh = est_flow_neigh[:, 1:K + 1, :]
    flow_diff = est_flow.view(bs * n, c) - est_flow_neigh.permute(1, 0, 2)

    flow_diff = (flow_diff).norm(p=loss_norm, dim=2)
    smooth_flow_loss = flow_diff.mean()
    smooth_flow_per_point = flow_diff.mean(dim=0).view(bs, n)

    return smooth_flow_loss, smooth_flow_per_point



def mask_NN_by_dist(dist, nn_ind, max_radius):
    # todo refactor to loss utils
    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)
    nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]

    return nn_ind
def _forward_flow_loss(pc1, pc2, est_flow):
    ''' not yet for K > 1 or BS > 1
    Smooth flow on same NN from pc2 '''
    _, forward_nn, _ = knn_points(pc1 + est_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)

    a = est_flow[0] # magnitude

    ind = forward_nn[0] # more than one?

    #if pc1 is bigger than pc2, then skip?
    if pc1.shape[1] < pc2.shape[1]:
        shape_diff = pc2.shape[1] - ind.shape[0] + 1 # one for dummy    # what if pc1 is bigger than pc2?
        a = F.pad(a, (0,0,0, shape_diff), mode='constant', value=0)
        a.retain_grad() # padding does not retain grad, need to do it manually. Check it

        ind = F.pad(ind, (0,0,0, shape_diff), mode='constant', value=pc2.shape[1])  # pad with dummy not in orig

    # storage of same points
    vec = torch.zeros(ind.shape[0], 3, device=pc1.device)

    vec.scatter_reduce_(0, ind.repeat(1,3), a, reduce='mean', include_self=False)  # will do forward flow

    # loss
    forward_loss = torch.nn.functional.mse_loss(vec[ind[:,0]], a, reduction='none').mean(dim=-1)

    return forward_loss.mean(), forward_loss[:est_flow.shape[1]]


def _forward_smoothness(pc1, pc2, est_flow, NN_pc2=None, K=2, include_pc2_smoothness=True):

    if NN_pc2 is None and include_pc2_smoothness:
        # Compute NN_pc2
        print('Computing NN_pc2')
        _, NN_pc2, _ = knn_points(pc2, pc2, lengths1=None, lengths2=None, K=K, norm=1)


    _, forward_nn, _ = knn_points(pc1 + est_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)

    a = est_flow[0] # magnitude

    ind = forward_nn[0] # more than one?

    if pc1.shape[1] < pc2.shape[1]:
        shape_diff = pc2.shape[1] - ind.shape[0] + 1 # one for dummy    # what if pc1 is bigger than pc2?
        a = F.pad(a, (0,0,0, shape_diff), mode='constant', value=0)
        a.retain_grad() # padding does not retain grad, need to do it manually. Check it

        ind = F.pad(ind, (0,0,0, shape_diff), mode='constant', value=pc2.shape[1])  # pad with dummy not in orig

    # storage of same points
    vec = torch.zeros(ind.shape[0], 3, device=pc1.device)

    # this is forward flow withnout NN_pc2 smoothness
    vec = vec.scatter_reduce_(0, ind.repeat(1,3), a, reduce='mean', include_self=False)

    forward_flow_loss = torch.nn.functional.mse_loss(vec[ind[:,0]], a, reduction='none').mean(dim=-1)

    if include_pc2_smoothness:
        # rest is pc2 smoothness with pre-computed NN
        keep_ind = ind[ind[:,0] != pc2.shape[1] ,0]

        # znamena, ze est flow body maji tyhle indexy pro body v pc2 a ty indexy maji mit stejne flow.
        n = NN_pc2[0, keep_ind, :]

        # beware of zeros!!!
        connected_flow = vec[n] # N x KSmooth x 3 (fx, fy, fz)

        prep_flow = est_flow[0].unsqueeze(1).repeat_interleave(repeats=K, dim=1) # correct

        # smooth it, should be fine
        flow_diff = prep_flow - connected_flow  # correct operation, but zeros makes problem

        occupied_mask = connected_flow.all(dim=2).repeat(3,1,1).permute(1,2,0)

        # occupied_mask
        per_flow_dim_diff = torch.masked_select(flow_diff, occupied_mask)

        # per_point_loss = per_flow_dim_diff.norm(dim=-1).mean()
        NN_pc2_loss = (per_flow_dim_diff ** 2).mean()    # powered to 2 because norm will sum it directly

    else:
        NN_pc2_loss = torch.tensor(0.)

    return forward_flow_loss.mean(), forward_flow_loss, NN_pc2_loss





class SmoothnessLoss(torch.nn.Module):

    # use normals to calculate smoothness loss
    def __init__(self, pc1, pc2=None, K=12, sm_normals_K=0, smooth_weight=1., VA=False, max_radius=2, loss_norm=1, forward_weight=0., pc2_smooth=False, **kwargs):

        super().__init__()
        self.K = K
        self.max_radius = max_radius
        self.pc1 = pc1
        self.pc2 = pc2
        self.normals_K = sm_normals_K
        self.loss_norm = loss_norm
        self.smooth_weight = smooth_weight

        self.Visibility_pc1 = VisibilityScene(dataset=kwargs['dataset'], pc_scene=pc1[0])
        self.Visibility_pc2 = VisibilityScene(dataset=kwargs['dataset'], pc_scene=pc2[0])


        # normal Smoothness
        if K > 0:

            if self.normals_K > 3:
                self.dist1, self.NN_pc1, _ = self.KNN_with_normals(pc1)
            else:
                self.dist1, self.NN_pc1, _ = knn_points(self.pc1, self.pc1, K=self.K)

            self.NN_pc1 = mask_NN_by_dist(self.dist1, self.NN_pc1, max_radius)


        # vis-aware - jenom skrtam, muze byt po radiusu
        self.VA = VA
        if VA:
            self.NN_pc1 = self.Visibility_pc1.visibility_aware_smoothness_KNN(self.NN_pc1).unsqueeze(0)

        # ff
        self.forward_weight = forward_weight

        # ff with NNpc2
        self.pc2_smooth = pc2_smooth
        self.NN_pc2 = None

        if pc2_smooth and K > 0:

            if self.normals_K > 3:
                self.dist2, self.NN_pc2, _ = self.KNN_with_normals(pc2)
            else:
                self.dist2, self.NN_pc2, _ = knn_points(self.pc2, self.pc2, K=self.K)

            self.NN_pc2 = mask_NN_by_dist(self.dist2, self.NN_pc2, max_radius)

            if VA:
                self.NN_pc2 = self.Visibility_pc2.visibility_aware_smoothness_KNN(self.NN_pc2).unsqueeze(0)

    def forward(self, pc1, est_flow, pc2):

        loss = torch.tensor(0, dtype=torch.float32, device=pc1.device)

        if self.smooth_weight > 0:

            smooth_loss, pp_smooth_loss = self.smoothness_loss(est_flow, self.NN_pc1, self.loss_norm)

            loss += self.smooth_weight * smooth_loss

        if self.forward_weight > 0:
            forward_loss, pp_forward_loss = self.forward_smoothness(pc1, est_flow, pc2)

            loss += self.forward_weight * forward_loss

        return loss

    def KNN_with_normals(self, pc):

        normals = estimate_pointcloud_normals(pc, neighborhood_size=self.normals_K)
        pc_with_norms = torch.cat([pc, normals], dim=-1)

        return knn_points(pc_with_norms, pc_with_norms, K=self.K)

    def smoothness_loss(self, est_flow, NN_idx, loss_norm=1, mask=None):

        bs, n, c = est_flow.shape

        if bs > 1:
            print("Smoothness Maybe not working for bs>1, needs testing!")
        K = NN_idx.shape[2]

        est_flow_neigh = est_flow.view(bs * n, c)
        est_flow_neigh = est_flow_neigh[NN_idx.view(bs * n, K)]

        est_flow_neigh = est_flow_neigh[:, 1:K + 1, :]
        flow_diff = est_flow.view(bs * n, c) - est_flow_neigh.permute(1, 0, 2)

        flow_diff = (flow_diff).norm(p=loss_norm, dim=2)
        smooth_flow_loss = flow_diff.mean()
        smooth_flow_per_point = flow_diff.mean(dim=0).view(bs, n)

        return smooth_flow_loss, smooth_flow_per_point


    def forward_smoothness(self, pc1, est_flow, pc2):


        _, forward_nn, _ = knn_points(pc1 + est_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)

        a = est_flow[0]

        ind = forward_nn[0] # more than one?

        if pc1.shape[1] < pc2.shape[1]:
            shape_diff = pc2.shape[1] - ind.shape[0] + 1 # one for dummy    # what if pc1 is bigger than pc2?
            a = torch.nn.functional.pad(a, (0,0,0, shape_diff), mode='constant', value=0)
            a.retain_grad() # padding does not retain grad, need to do it manually. Check it

            ind = torch.nn.functional.pad(ind, (0,0,0, shape_diff), mode='constant', value=pc2.shape[1])  # pad with dummy not in orig

        # storage of same points
        vec = torch.zeros(ind.shape[0], 3, device=pc1.device)

        # this is forward flow withnout NN_pc2 smoothness
        vec = vec.scatter_reduce_(0, ind.repeat(1,3), a, reduce='mean', include_self=False)

        forward_flow_loss = torch.nn.functional.mse_loss(vec[ind[:,0]], a, reduction='none').mean(dim=-1)

        if self.pc2_smooth:
            # rest is pc2 smoothness with pre-computed NN
            keep_ind = ind[ind[:,0] != pc2.shape[1] ,0]

            # znamena, ze est flow body maji tyhle indexy pro body v pc2 a ty indexy maji mit stejne flow.
            n = self.NN_pc2[0, keep_ind, :]

            # beware of zeros!!!
            connected_flow = vec[n] # N x KSmooth x 3 (fx, fy, fz)

            prep_flow = est_flow[0].unsqueeze(1).repeat_interleave(repeats=self.K, dim=1) # correct

            # smooth it, should be fine
            flow_diff = prep_flow - connected_flow  # correct operation, but zeros makes problem

            occupied_mask = connected_flow.all(dim=2).repeat(3,1,1).permute(1,2,0)

            # occupied_mask
            per_flow_dim_diff = torch.masked_select(flow_diff, occupied_mask)

            # per_point_loss = per_flow_dim_diff.norm(dim=-1).mean()
            NN_pc2_loss = (per_flow_dim_diff ** 2).mean()    # powered to 2 because norm will sum it directly

        else:
            NN_pc2_loss = torch.tensor(0.)

        forward_loss = forward_flow_loss.mean() + NN_pc2_loss

        return forward_loss, forward_flow_loss

class VAChamferLoss(torch.nn.Module):

    def __init__(self, pc2, fov_up, fov_down, H, W, max_range, pc_scene=None, nn_weight=1, max_radius=2, both_ways=False, free_weight=0, margin=0.001, ch_normals_K=0, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.pc2 = pc2
        self.pc_scene = pc_scene if pc_scene is not None else pc2

        self.Visibility = VisibilityScene(dataset=self.kwargs['dataset'], pc_scene=pc_scene[0])


        # todo option of "pushing" points out of the freespace
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.H = H
        self.W = W
        self.max_range = max_range
        self.margin = margin
        self.free_weight = free_weight

        # NN component
        self.normals_K = ch_normals_K
        self.nn_weight = nn_weight
        self.nn_max_radius = max_radius
        self.both_ways = both_ways

        # torch.use_deterministic_algorithms(mode=True, warn_only=False)  # this ...
        # pc2_depth, idx_w, idx_h, inside_range_img = range_image_coords(pc2[0], fov_up, fov_down, proj_H, proj_W)

        # self.range_depth = create_depth_img(pc2_depth, idx_w, idx_h, proj_H, proj_W, inside_range_img)
        # torch.use_deterministic_algorithms(mode=False, warn_only=False)  # this ...

    def forward(self, pc1, est_flow, pc2=None):
        '''

        Args:
            pc1:
            est_flow:

        Returns:
        mask whether the deformed point cloud is in freespace visibility area
        '''
        # dynamic

        # assign Kabsch to lonely points or just push them out of freespace?
        # precompute chamfer, radius
        chamf_x, chamf_y = self.chamfer_distance_loss(pc1 + est_flow, self.pc2, both_ways=self.both_ways, normals_K=self.normals_K)

        if self.free_weight > 0:
            freespace_loss = self.flow_freespace_loss(pc1, est_flow, chamf_x)

        else:
            freespace_loss = torch.zeros_like(chamf_x, dtype=torch.float32, device=chamf_x.device)

        chamf_loss = self.nn_weight * (chamf_x.mean() + chamf_y.mean()) + self.free_weight * freespace_loss.mean()

        return chamf_loss, freespace_loss


    def chamfer_distance_loss(self, x, y, x_lengths=None, y_lengths=None, both_ways=False, normals_K=0, loss_norm=1):
        '''
        Unique Nearest Neighboors?
        :param x:
        :param y:
        :param x_lengths:
        :param y_lengths:
        :param reduction:
        :return:
        '''
        if normals_K >= 3:
            normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_K)
            normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_K)

            x = torch.cat([x, normals1], dim=-1)
            y = torch.cat([y, normals2], dim=-1)


        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=loss_norm)
        cham_x = x_nn.dists[..., 0]  # (N, P1)
        # x_nearest_to_y = x_nn[1]

        if both_ways:
            y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=loss_norm)
            cham_y = y_nn.dists[..., 0]  # (N, P2)
            # y_nearest_to_x = y_nn[1]
        else:

            cham_y = torch.tensor(0, dtype=torch.float32, device=x.device)

        return cham_x, cham_y

    def flow_freespace_loss(self, pc1, est_flow, chamf_x):

        # flow_depth, flow_w, flow_h, flow_inside = self.Visibility.generate_range_coors(pc1 + est_flow)
        pc2_image_depth = self.Visibility.assign_depth_to_flow((pc1 + est_flow)[0])
        flow_depth = ((pc1+est_flow)[0]).norm(dim=-1)

            # use it only for flow inside the image
        # masked_pc2_depth = self.range_depth[flow_h[flow_inside], flow_w[flow_inside]]
        compared_depth = pc2_image_depth - flow_depth



        # if flow point before the visible point from pc2, then it is in freespace
        # margin is just little number to not push points already close to visible point
        flow_in_freespace = compared_depth > 0 + self.margin


        # Indexing flow in freespace
        # freespace_mask = torch.zeros_like(chamf_x, dtype=torch.bool)[0]
        # freespace_mask = flow_in_freespace
        # if repel:
        freespace_loss = - est_flow[0, flow_in_freespace].norm(dim=-1).mean()

        # freespace_loss = flow_in_freespace * chamf_x

        return freespace_loss


# datainfo
class LossModule(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        if kwargs['dataset'] in ['kitti_t', 'kitti_o']:
            # range function
            pass

    def update(self, pc1, pc2):
        if hasattr(self, 'pc_scene'):
            self.VAChamfer_loss = VAChamferLoss(pc2=pc2, pc_scene=self.pc_scene, **self.kwargs)
        else:
            self.VAChamfer_loss = VAChamferLoss(pc2=pc2, **self.kwargs)


        self.Smoothness_loss = SmoothnessLoss(pc1, pc2, **self.kwargs)

    def forward(self, pc1, est_flow, pc2):

        chamf_loss, pp_freespace_loss = self.VAChamfer_loss(pc1, est_flow)
        smooth_loss = self.Smoothness_loss(pc1, est_flow, pc2)

        loss = smooth_loss + chamf_loss
        # todo save losses
        return loss

