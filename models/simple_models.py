import torch
import torch.nn as nn


class Weights_model(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        self.output_dim = output_dim
        # self.init_weight = torch.nn.Parameter(torch.rand(1, 3, output_dim))

    def update(self, pc1, pc2=None):

        self.weights = torch.nn.Parameter(torch.randn(1, pc1.shape[1], self.output_dim, requires_grad=True))

    def forward(self, pc1, pc2=None):

        flow = self.weights[..., :3]
        mask = self.weights[..., 3:].softmax(dim=2)

        return mask, flow

class FlowSegPrior(torch.nn.Module):

    def __init__(self, input_size=1000, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, output_feat=False):
        super().__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_feat = output_feat

        self.nn_layers = nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x + 64, filter_size)))    # here change
            if act_fn == 'relu':
                self.nn_layers.append(nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(nn.Sigmoid())

            for _ in range(layer_size - 1):
                self.nn_layers.append(nn.Sequential(nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(nn.Sigmoid())

            self.nn_layers.append(nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(
                nn.Sequential(nn.Linear(dim_x, dim_x)))  # todo + bias dela flow model jako weights?

        self.per_sample_init = True

    def update(self, pc1, pc2=None):
        pass
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    # maybe arguments init weights every time? better
    def initialize(self):

        if self.per_sample_init:

            self.apply(self.init_weights)

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        if self.output_feat:
            feat = []

        for layer in self.nn_layers:
            x = layer(x)
            if self.output_feat and layer == nn.Linear:
                feat.append(x)

        mask = x.softmax(dim=-1)
        # breakpoint()
        if self.output_feat:
            return mask, feat
        else:
            return mask

class InstanceSegModule(nn.Module):
    def __init__(self):
        super().__init__()
        from models.OGC.models.segnet_kitti import MaskFormer3D
        self.net = MaskFormer3D(n_slot=10, use_xyz=True, n_transformer_layer=2, transformer_embed_dim=128, transformer_input_pos_enc=False)#.to(device)

    def update(self, pc1, pc2=None):
        pass

    def forward(self, pc):
        mask = self.net(pc, pc)

        return mask

class JointFlowInst(nn.Module):
    def __init__(self):
        super().__init__()
        from models.OGC.models.segnet_kitti import MaskFormer3D
        self.net = MaskFormer3D(n_slot=10, use_xyz=True, n_transformer_layer=2, transformer_embed_dim=128, transformer_input_pos_enc=False)#.to(device)
        self.flow_model = FlowSegPrior(input_size=1000, dim_x=3, filter_size=128, act_fn='relu', layer_size=8)

    def update(self, pc1, pc2=None):
        pass

    def forward(self, pc):

        mask, pointnet_feats = self.net(pc, pc)
        flow_features = torch.cat([pc, pointnet_feats], dim=2)
        # print(flow_features.shape)
        pred_flow = self.flow_model(flow_features)

        return mask, pred_flow


def fit_flow():
    ''' c_pc is center point cloud, calculate flow around it '''
    forth_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
    back_flow = torch.zeros(c_pc.shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([forth_flow, back_flow], lr=0.008)
    # Losses
    SM_loss = SmoothnessLoss(pc1=c_pc, K=16, max_radius=1)
    b_DT = DT(c_pc, b_pc)
    f_DT = DT(c_pc, f_pc)
    loss_list = []

    # init dbscan - test against smoothness
    init_clusters = DBSCAN(eps=eps, min_samples=1).fit_predict(c_pc[0, :, :2].detach().cpu().numpy())
    # init_clusters = torch.from_numpy(init_clusters).to(device=pc1.device).long() + 1

    My_metric = SceneFlowMetric()

    last_loss = 100000
    for e in range(max_flow_epoch):

        # forth_dist, forth_nn, _ = knn_points(c_pc + forth_flow, f_pc, K=1, return_nn=True)
        # back_dist, back_nn, _ = knn_points(c_pc + back_flow, b_pc, K=1, return_nn=True)
        forth_dist, _ = f_DT.torch_bilinear_distance(c_pc + forth_flow)
        back_dist, _ = b_DT.torch_bilinear_distance(c_pc + back_flow)

        dist_loss = (forth_dist + back_dist).mean()
        smooth_loss = SM_loss(c_pc, forth_flow, f_pc) + SM_loss(c_pc, back_flow, f_pc)
        # smooth_loss = smooth_cluster_ids(forth_flow, init_clusters) + smooth_cluster_ids(back_flow, init_clusters)
        time_smooth = (forth_flow - (-back_flow)).norm(dim=2, p=1).mean()  # maybe magnitude of flow?

        loss = dist_loss + 0.5 * smooth_loss.mean() + time_smooth

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            print(e, loss.item(), time_smooth.item())

        if e >= early_patience:
            if last_loss - loss < loss_diff_stop:
                break
            else:
                last_loss = loss.item()

        loss_list.append(loss.item())

    # After flow
    print("---------- After Flow -------------")
    plain_NN_dist, _ = f_DT.torch_bilinear_distance(c_pc)  # must synchronized with pose
    mos = np.logical_or((forth_flow[0].norm(dim=1, p=1) > motion_metric).detach().cpu().numpy(),
                        plain_NN_dist.detach().cpu().numpy() > motion_metric)  # mask of static and dynamic
