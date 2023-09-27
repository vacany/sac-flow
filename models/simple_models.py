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
