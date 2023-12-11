from time import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from loss.flow import DT, SC2_KNN #, SmoothnessLoss, FastNN
from loss.instance import smooth_cluster_ids, gather_flow_ids, center_rigidity_loss
# from lietorch import SE3
# from pytorch3d.transforms import euler_angles_to_matrix
# from ops.transform import find_weighted_rigid_alignment
from models.neuralpriors import PoseTransform




def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class ModelTemplate(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.store_init_params(locals())
        self.model_cfg = dir(self)
        # self.initialize()

    def forward(self, data):
        st = time.time()

        eval_time = time.time() - st
        return data

    def model_forward(self, data):
        return data

    def initialize(self):
        self.apply(init_weights)

    def store_init_params(self, local_variables):
        for key, value in local_variables.items():
            if key != 'self':
                setattr(self, key, value)
            if key == 'kwargs':
                for k, v in value.items():
                    setattr(self, k, v)
            if key == 'args':
                setattr(self, 'args', value)

            # if key.startswith('_'):
            #     continue


class STNeuralPrior(torch.nn.Module):
    '''
    Neural Prior with Rigid Transformation, takes only point cloud t=1 on input and returns flow and rigid flow (ego-motion flow)
    '''

    def __init__(self, eps=0.15, lr=0.008, early_stop=30, cluster_every_iter=100, refine=False, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        self.RigidTransform = PoseTransform()

        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

        self.eps = eps
        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose
        self.refine = refine
        self.cluster_every_iter = cluster_every_iter
    def forward(self, data):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """
        st = time()
        pc1, pc2 = data['pc1'][:1], data['pc2'][:1]
        flow_ids = - torch.ones(pc1.shape[0], pc1.shape[1], device=pc1.device, dtype=torch.long)
        # Smooth_layer = SmoothnessLoss(pc1, pc2, K=4, sm_normals_K=0, smooth_weight=1, VA=False, max_radius=2, forward_weight=0, pc2_smooth=False, dataset='argoverse')
        DT_layer = DT(pc1, pc2)
        KNNRigidity_layer = SC2_KNN(pc1, K=16, d_thre=0.03)
        last_loss = torch.inf

        # Iteration of losses
        for e in range(500):

            deformed_pc = self.RigidTransform(pc1)
            rigid_flow = deformed_pc - pc1

            x = self.nn_layers[0](pc1)

            for layer in self.nn_layers[1:]:
                x = layer(x)

            # Sum of flows
            pred_flow = x + rigid_flow

            _, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)
            sc_loss = KNNRigidity_layer(pred_flow)

            # truncated at two meters
            dt_loss = per_point_dt_loss[per_point_dt_loss < 2].mean()
            # rigid_dt_loss = per_point_rigid_loss[per_point_rigid_loss < 2].mean()

            loss = dt_loss + sc_loss #+ rigid_dt_loss  # + smooth_loss

            ###########################################
            ##### SpatioTemporal Clustering Module ####
            ###########################################
            if e % self.cluster_every_iter == 0 and e > 0:   # start after flow is initialized
                # print('Performing clustering in Iter: ', e)
                flow_ids = gather_flow_ids(pc1[:1], pred_flow[:], eps=self.eps)

            if e > self.cluster_every_iter:
                loss += smooth_cluster_ids(pred_flow, flow_ids).mean()

            if torch.abs(last_loss - loss) < self.loss_diff and e > self.early_stop:
                break
            else:
                last_loss = loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.verbose:
                print(f"Neural Prior Module Iter: {e:03d}, NN Loss: {dt_loss.item():.4f}, SC Loss: {sc_loss.item():.4f}, Total Loss: {loss.item():.4f}")


        ##############################
        ##### Refinement Module ######
        ##############################

        if self.refine:
            # pred_flow = output_data['pred_flow'].detach().requires_grad_(True)
            pred_flow = pred_flow.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([pred_flow], lr=0.008)

            flow_ids = gather_flow_ids(pc1[:1], pred_flow[:], eps=self.eps)

            for i in range(100):
                # pokus some indexing wrong
                # forth_dist, _ = f_DT.torch_bilinear_distance(pc1[:1] + pred_flow)
                smooth_flow_loss = smooth_cluster_ids(pred_flow, flow_ids)
                loss = smooth_flow_loss.mean() #+ forth_dist.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.verbose:
                    print("Refinement Module Iter: ", i, '\tInstanceSmoothFlow: ', smooth_flow_loss.mean().item())

        data['pred_flow'] = pred_flow
        data['rigid_flow'] = rigid_flow
        data['pred_inst'] = flow_ids
        data['eval_time'] = time() - st
        return data

    def initialize(self):
        self.apply(init_weights)


class MultiRigidNeuralPrior(torch.nn.Module):
    '''
    Neural Prior with Rigid Transformation, takes only point cloud t=1 on input and returns flow and rigid flow (ego-motion flow)
    '''

    def __init__(self, eps=0.15, lr=0.008, early_stop=30, cluster_every_iter=100, refine=False, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        self.RigidTransform = PoseTransform()

        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        self.initialize()

        self.eps = eps
        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose
        self.refine = refine
        self.cluster_every_iter = cluster_every_iter
    def forward(self, data):
        """ Points -> Flow
            [B, N, 3] -> [B, N, 3]
        """
        st = time()
        pc1, pc2 = data['pc1'][:1], data['pc2'][:1]
        flow_ids = - torch.ones(pc1.shape[0], pc1.shape[1], device=pc1.device, dtype=torch.long)
        # Smooth_layer = SmoothnessLoss(pc1, pc2, K=4, sm_normals_K=0, smooth_weight=1, VA=False, max_radius=2, forward_weight=0, pc2_smooth=False, dataset='argoverse')
        DT_layer = DT(pc1, pc2)


        # KNNRigidity_layer = SC2_KNN(pc1, K=16, d_thre=0.03)
        last_loss = torch.inf

        # Using GT mask
        # print('Using ground truth id mask')
        # flow_ids = data['id_mask1'][0].long() + 1

        # cluster time
        # clu_st = time()
        cluster_ids = DBSCAN(eps=0.8, min_samples=30).fit_predict(pc1[0].cpu().numpy()) + 1
        flow_ids = torch.from_numpy(cluster_ids).to(pc1.device).long()  # slow shift to gpu, cupy and cuml might solve this
        # print('Clustering time: ', time() - clu_st)
        # Iteration of losses
        for e in range(500):

            # deformed_pc = self.RigidTransform(pc1)
            # rigid_flow = deformed_pc - pc1

            x = self.nn_layers[0](pc1)

            for layer in self.nn_layers[1:]:
                x = layer(x)

            # Sum of flows
            pred_flow = x #+ rigid_flow

            _, per_point_dt_loss = DT_layer.torch_bilinear_distance(pc1 + pred_flow)
            # sc_loss = KNNRigidity_layer(pred_flow)


            # todo do it as distance not vectors
            rigid_loss = center_rigidity_loss(pc1, pred_flow, flow_ids)
            # truncated at two meters
            dt_loss = per_point_dt_loss[per_point_dt_loss < 2].mean()
            # rigid_dt_loss = per_point_rigid_loss[per_point_rigid_loss < 2].mean()

            loss = dt_loss + rigid_loss #+ rigid_dt_loss  # + smooth_loss

            ###########################################
            ##### SpatioTemporal Clustering Module ####
            ###########################################
            # if e % self.cluster_every_iter == 0 and e > 0:   # start after flow is initialized
                # print('Performing clustering in Iter: ', e)
                # flow_ids = gather_flow_ids(pc1[:1], pred_flow[:], eps=self.eps)

            # if e > self.cluster_every_iter:
            #     loss += smooth_cluster_ids(pred_flow, flow_ids).mean()

            if torch.abs(last_loss - loss) < self.loss_diff and e > self.early_stop:
                break
            else:
                last_loss = loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.verbose:
                print(f"Neural Prior Module Iter: {e:03d}, NN Loss: {dt_loss.item():.4f}, Rigid Loss: {rigid_loss.item():.4f}, Total Loss: {loss.item():.4f}")


        ##############################
        ##### Refinement Module ######
        ##############################

        if self.refine:
            # pred_flow = output_data['pred_flow'].detach().requires_grad_(True)
            pred_flow = pred_flow.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([pred_flow], lr=0.008)

            flow_ids = gather_flow_ids(pc1[:1], pred_flow[:], eps=self.eps)

            for i in range(100):
                # pokus some indexing wrong
                # forth_dist, _ = f_DT.torch_bilinear_distance(pc1[:1] + pred_flow)
                smooth_flow_loss = smooth_cluster_ids(pred_flow, flow_ids)
                loss = smooth_flow_loss.mean() #+ forth_dist.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.verbose:
                    print("Refinement Module Iter: ", i, '\tInstanceSmoothFlow: ', smooth_flow_loss.mean().item())

        data['pred_flow'] = pred_flow
        # data['rigid_flow'] = rigid_flow
        data['pred_inst'] = flow_ids
        data['eval_time'] = time() - st
        return data

    def initialize(self):
        self.apply(init_weights)
