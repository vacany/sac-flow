import torch
import sys

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals

from .visibility import KNN_visibility_solver, substitute_NN_by_mask, strip_KNN_with_vis

def KNN_with_normals(pc1, normals1=None, K=3, normals_K=3):

    if normals1 is None:
        normals1 = estimate_pointcloud_normals(pc1, neighborhood_size=normals_K)

    pc1_n = torch.cat([pc1, normals1], dim=2)

    nn_normal_dist, nn_normal_ind, _ = knn_points(pc1_n, pc1_n, K=K)

    return nn_normal_ind

def chamfer_distance_loss(x, y, x_lengths=None, y_lengths=None, use_normals=False, normals_neighboor=3, reduction='mean'):
    '''
    Unique Nearest Neightboors?
    :param x:
    :param y:
    :param x_lengths:
    :param y_lengths:
    :param reduction:
    :return:
    '''
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=2)
    # y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=1)

    # if use_normals:
    #     normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_neighboor)
    #     normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_neighboor)


    cham_x = x_nn.dists[..., 0]  # (N, P1)
    # cham_y = y_nn.dists[..., 0]  # (N, P2)

    nearest_to_y = x_nn[1]
    # breakpoint()
    # this way the NN loss is calculated only on one point of nn?
    # print(x, y)
    # print(cham_x)

    # breakpoint()
    # else:
    #     cham_x = x_nn.dists[:N_pts_x, 0]  # (N, P1)
        # cham_y = y_nn.dists[:N_pts_x, 0]  # (N, P2)
        #
        # nearest_to_y = x_nn[1][:,N_pts_x]

    # TODO rozmyslet, jestli potrebujeme two-way chamfer distance

    # nn_loss = x - y[nearest_to_y]
    # print(y[:, nearest_to_y, :].shape)
    nn_loss = cham_x
    # nn_loss = (cham_x + cham_y) / 2

    if reduction == 'mean':
        nn_loss = nn_loss.mean()
    elif reduction == 'sum':
        nn_loss = nn_loss.sum()
    elif reduction == 'none':
        nn_loss = nn_loss
    else:
        raise NotImplementedError

    # breakpoint()
    return nn_loss#, nearest_to_y



def rigid_cycle_loss(p_i, fw_trans, bw_trans, reduction='none'):

    trans_p_i = torch.cat((p_i, torch.ones((len(p_i), p_i.shape[1], 1), device=p_i.device)), dim=2)
    bw_fw_trans = bw_trans @ fw_trans - torch.eye(4, device=fw_trans.device)
    # todo check this in visualization, if the points are transformed as in numpy
    res_trans = torch.matmul(bw_fw_trans, trans_p_i.permute(0, 2, 1)).norm(dim=1)

    rigid_loss = res_trans.mean()

    return rigid_loss

def smoothness_loss(est_flow, NN_idx, loss_norm=1, mask=None):

    bs, n, c = est_flow.shape

    if bs > 1:
        print("Smoothness Maybe not working, needs testing!")
    K = NN_idx.shape[2]

    # breakpoint()
    # valid_mask = torch.ones((bs * n, K), device=est_flow.device)
    # valid_mask[NN_idx.view(bs * n, K) == 0] = 0
    # valid_mask = valid_mask.permute(1, 0).unsqueeze(2).repeat(1, 1, c)

    # format prepared for subtraction. For fever operations can be calculated outside the loss
    # current approach is okay, because loss is zeroed and I take mean from all points, not from neighboors?

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



class FlowSmoothLoss(torch.nn.Module):
    # use normals to calculate smoothness loss
    def __init__(self, pc, K=12, weight=1., max_radius=1, loss_norm=1):
        super().__init__()
        self.K = K
        self.max_radius = max_radius
        self.pc = pc
        self.loss_norm = loss_norm
        self.weight = weight

        self.dist, self.nn_ind, _ = knn_points(self.pc, self.pc, K=self.K)
        tmp_idx = self.nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(self.nn_ind.device)
        self.nn_ind[self.dist > max_radius] = tmp_idx[self.dist > max_radius]

    def forward(self, pred_flow):

        smooth_loss, per_point_smooth_loss = smoothness_loss(pred_flow, self.nn_ind, loss_norm=self.loss_norm)

        return smooth_loss * self.weight, per_point_smooth_loss * self.weight

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

if __name__ == "__main__":
    x = torch.rand(1, 120, 3, requires_grad=True)
    y = torch.rand(1, 100, 3, requires_grad=True)



    x_lengths = torch.ones(len(x), dtype=torch.long) * x.shape[1]
    y_lengths = torch.ones(len(y), dtype=torch.long) * x.shape[1]




    nn_loss, nn_x_to_y = chamfer_distance_loss(x, y, x_lengths, y_lengths)
