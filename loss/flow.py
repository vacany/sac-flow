import torch
import sys
import argparse
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals

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
        y_nearest_to_x = y_nn[1]

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

# def rigid_cycle_loss(p_i, fw_trans, bw_trans, reduction='none'):
#
#     trans_p_i = torch.cat((p_i, torch.ones((len(p_i), p_i.shape[1], 1), device=p_i.device)), dim=2)
#     bw_fw_trans = bw_trans @ fw_trans - torch.eye(4, device=fw_trans.device)
#     # todo check this in visualization, if the points are transformed as in numpy
#     res_trans = torch.matmul(bw_fw_trans, trans_p_i.permute(0, 2, 1)).norm(dim=1)
#
#     rigid_loss = res_trans.mean()
#
#     return rigid_loss

def smoothness_loss(est_flow, NN_idx, loss_norm=1, mask=None):

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

def visibility_aware_smoothness_loss(est_flow, KNN_image_indices, depth, NN_idx, margin=5, loss_norm=1, reduction='mean'):

    valid_KNN_mask = KNN_visibility_solver(KNN_image_indices, depth, margin=margin)
    vis_aware_KNN = substitute_NN_by_mask(NN_idx, valid_KNN_mask)

    smooth_flow_loss, smooth_flow_per_point = smoothness_loss(est_flow, vis_aware_KNN, loss_norm=loss_norm)

    if reduction == 'mean':
        return smooth_flow_loss

    elif reduction == 'sum':
        return smooth_flow_per_point.sum()

    else:
        return smooth_flow_per_point

def mask_NN_by_dist(dist, nn_ind, max_radius):
    # todo refactor to loss utils
    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)
    nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]

    return nn_ind
def forward_flow_loss(pc1, pc2, est_flow):
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


def forward_smoothness(pc1, pc2, est_flow, NN_pc2=None, K=2, include_pc2_smoothness=True):

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

# class FlowSmoothLoss(torch.nn.Module):
#     # use normals to calculate smoothness loss
#     def __init__(self, pc, K=12, weight=1., max_radius=1, loss_norm=1):
#         super().__init__()
#         self.K = K
#         self.max_radius = max_radius
#         self.pc = pc
#         self.loss_norm = loss_norm
#         self.weight = weight
#
#         self.dist, self.nn_ind, _ = knn_points(self.pc, self.pc, K=self.K)
#         tmp_idx = self.nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(self.nn_ind.device)
#         self.nn_ind[self.dist > max_radius] = tmp_idx[self.dist > max_radius]
#
#     def forward(self, pred_flow):
#
#         smooth_loss, per_point_smooth_loss = smoothness_loss(pred_flow, self.nn_ind, loss_norm=self.loss_norm)
#
#         return smooth_loss * self.weight, per_point_smooth_loss * self.weight

class VisibilitySmoothnessLoss(torch.nn.Module):
    # Maybe inheritance next time?
    def __init__(self, pc, K, VOF, HOF, max_radius=1.5, margin=3, loss_norm=1):
        super(VisibilitySmoothnessLoss, self).__init__()
        self.margin = margin
        self.pc = pc
        self.K = K
        self.VOF = VOF
        self.HOV = HOF
        self.max_radius = max_radius
        self.margin = margin
        self.loss_norm = loss_norm


        self.dist, self.nn_ind, _ = knn_points(self.pc, self.pc, K=K)

        self.nn_ind = strip_KNN_with_vis(self.pc[0], self.nn_ind[0], self.VOF, self.HOV, margin=self.margin).unsqueeze(0)
        # apply radius
        tmp_idx = self.nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(self.nn_ind.device)
        self.nn_ind[self.dist > max_radius] = tmp_idx[self.dist > max_radius]

    def forward(self, pred_flow):

        smooth_loss, per_point_smooth_loss = smoothness_loss(pred_flow, self.nn_ind, loss_norm=self.loss_norm)

        return smooth_loss, per_point_smooth_loss

# class loss, weights 0 means if applied at all and rest values are weights

# separate smoothness, chamfer
class UnsupervisedFlowLosses(torch.nn.Module):
    # update samples
    def __init__(self, args : argparse.Namespace):
        super().__init__()
        self.args = args

        # store outputs of losses if better?
        self.current_loss = 10000

        # init losses
        self.loss_functions = []

        # vertical and horizontal field of view
        self.VOF, self.HOF = 64, 1028

        # kostka pro vyber NN / okoli - pondelimozna

    # todo rewrite parameters
    def update(self, pc1, pc2):
        args = self.args

        self.NN_pc1 = None
        self.NN_pc2 = None

        if args.l_dt > 0:
            self.dt = DT(pc1, pc2, grid_factor=10)

        if args.normals_K >= 3:
            normals1 = estimate_pointcloud_normals(pc1, neighborhood_size=args.normals_K)
            pc_with_norms = torch.cat([pc1, normals1], dim=-1)

            # This is hence forward!
            pc1 = pc_with_norms

        # Smoothness
        if args.l_sm > 0:

            dist, self.NN_pc1, _ = knn_points(pc1, pc1, lengths1=None, lengths2=None, K=args.smooth_K, norm=1)

            if args.l_vsm > 0:

                mask_NN_by_dist(dist, self.NN_pc1, args.l.NN_max_radius)
                self.NN_pc1 = strip_KNN_with_vis(pc1[0], self.NN_pc1[0], VOF=self.VOF, HOF=self.HOF, margin=3).unsqueeze(0)

        if args.l_ff > 0.:
            _, self.NN_pc2, _ = knn_points(pc2, pc2, lengths1=None, lengths2=None, K=args.smooth_K, norm=1)


    def forward(self, pc1, pc2, est_flow):
        args = self.args

        loss_dict = {}
        loss = 0

        if args.l_sm > 0 or args.l_vsm > 0:
            smooth_loss, smooth_per_point = smoothness_loss(est_flow, self.NN_pc1, loss_norm=1, mask=None)

            loss_dict['smooth_loss'] = smooth_loss
            loss_dict['smooth_per_point'] = smooth_per_point

            loss += args.l_sm * smooth_loss

        if args.l_ch > 0 and args.l_dt == 0:
            chamfer_loss, per_point_chamfer, x_to_y = chamfer_distance_loss(pc1 + est_flow, pc2,
                                                        both_ways=args.l_ch_bothways, normals_K=args.normals_K, loss_norm=1)

            loss_dict['chamfer_loss'] = chamfer_loss
            loss_dict['chamfer_per_point'] = per_point_chamfer
            loss_dict['chamfer_x_to_y'] = x_to_y

            loss += args.l_ch * chamfer_loss

        if args.l_dt > 0:
            dt_loss, per_point_dt = self.dt.torch_bilinear_distance(pc1 + est_flow)

            loss_dict['dt_loss'] = dt_loss
            loss_dict['dt_per_point'] = per_point_dt

            loss += args.l_dt * dt_loss

        if args.l_ff > 0:

            # FF_loss, per_point_FF = forward_flow_loss(pc1, pc2, est_flow)
            ff_loss, per_point_ff_loss, pc2_smooth_loss = forward_smoothness(pc1, pc2, est_flow, NN_pc2=self.NN_pc2,
                                                            K=args.smooth_K, include_pc2_smoothness=args.pc2_smoothness)

            loss_dict['ff_loss'] = ff_loss
            loss_dict['ff_per_point'] = per_point_ff_loss
            loss_dict['ff_pc2_smooth_loss'] = pc2_smooth_loss

            loss += args.l_ff * ff_loss + args.l_ff * pc2_smooth_loss

        loss_dict['loss'] = loss

        return loss_dict

if __name__ == "__main__":
    pass

    # KNN for normals estimation
    # KNN smoothness
    # KNN max radius
