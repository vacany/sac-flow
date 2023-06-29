import os
import uuid
import torch
import torch.nn as nn
import pandas as pd
import argparse
import copy
import itertools
import glob
import numpy as np
import time
import datetime
from tqdm import tqdm

from data.PATHS import DATA_PATH, TMP_VIS_PATH, EXP_PATH

from loss.flow import UnsupervisedFlowLosses
from ops.metric import scene_flow_metrics
from models.FastNSF.optimization import Neural_Prior, Timers, EarlyStopping, init_weights


def preprocess_args(args):
    '''
    args: argparse object where values can be in list. Then this function return permutations of the arguments in list,
    so each value in list is tried out.
    return: list of argparse objects
    '''
    args_list = []
    iterable_names = []
    for name in args.__dict__:
        # print(name, getattr(args, name))
        if type(getattr(args, name)) == list:
            is_ablation = True
            iterable_names.append(name)

    if len(iterable_names) == 0:
        args_list.append(args)

    else:
        dict_pairs = {}

        for iter_name in iterable_names:
            dict_pairs.update({iter_name: getattr(args, iter_name)})

            keys, values = zip(*dict_pairs.items())


        variants = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for exp_var in variants:

            args_copy = copy.deepcopy(args)

            args_copy.__dict__.update(exp_var)
            # print(args_copy)

            args_list.append(args_copy)

    return args_list

def build_dataloader(args):

    from data.NSF_data import NSF_dataset
    dataset = NSF_dataset(dataset_type=args.dataset)

    return dataset

# def init?
def build_model(args):

    net = Neural_Prior()

    return net

def build_loss(args):

    Loss_Function = UnsupervisedFlowLosses(args)

    return Loss_Function


# solver
def solver(
        pc1: torch.Tensor,
        pc2: torch.Tensor,
        gt_flow: torch.Tensor,
        net: nn.Module,
        Loss_Function: UnsupervisedFlowLosses,
        args: argparse.Namespace,
):

    timers = Timers()
    timers.tic("solver_timer")

    pre_compute_st = time.time()
    solver_time = 0.


    if args.per_sample_init:
        net.apply(init_weights)

    for param in net.parameters():
        param.requires_grad = True

    params = net.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0)

    total_losses = []
    total_acc_strit = []
    total_iter_time = []

    early_stopping = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)


    pc1 = pc1.contiguous()
    pc2 = pc2.contiguous()
    gt_flow = gt_flow.contiguous()

    Loss_Function.update(pc1, pc2)

    pre_compute_time = time.time() - pre_compute_st
    solver_time = solver_time + pre_compute_time

    # ANCHOR: initialize best metrics
    best_loss_1 = 1e10
    best_flow_1 = None
    best_epe3d_1 = 1.
    best_acc3d_strict_1 = 0.
    best_acc3d_relax_1 = 0.
    best_angle_error_1 = 1.
    best_outliers_1 = 1.
    best_epoch = 0
    # kdtree_query_time = 0.
    net_time = 0.
    net_backward_time = 0.
    dt_query_time = 0.

    for epoch in range(args.iters):
        iter_time_init = time.time()

        optimizer.zero_grad()

        net_time_st = time.time()
        flow_pred_1 = net(pc1)
        net_time = net_time + time.time() - net_time_st
        pc1_deformed = pc1 + flow_pred_1

        # loss = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
        loss_dict = Loss_Function(pc1, pc2, flow_pred_1)
        loss = loss_dict['loss']

        net_backward_st = time.time()
        loss.backward()
        optimizer.step()
        net_backward_time = net_backward_time + time.time() - net_backward_st


        if early_stopping.step(loss):
            break

        iter_time = time.time() - iter_time_init
        solver_time = solver_time + iter_time

        flow_pred_1_final = pc1_deformed - pc1
        flow_metrics = gt_flow.clone()
        EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(flow_pred_1_final,
                                                                                              flow_metrics)

        # ANCHOR: get best metrics
        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_flow_1 = flow_pred_1_final
            best_epe3d_1 = EPE3D_1
            best_acc3d_strict_1 = acc3d_strict_1
            best_acc3d_relax_1 = acc3d_relax_1
            best_angle_error_1 = angle_error_1
            best_outliers_1 = outlier_1
            best_epoch = epoch

        total_losses.append(loss.item())
        total_acc_strit.append(acc3d_strict_1)
        total_iter_time.append(time.time() - iter_time_init)


    timers.toc("solver_timer")
    time_avg = timers.get_avg("solver_timer")


    # ANCHOR: get the best metrics
    info_dict = {
        'final_flow': best_flow_1,
        'loss': best_loss_1,
        'loss_dict' : loss_dict,
        'EPE3D_1': best_epe3d_1,
        'acc3d_strict_1': best_acc3d_strict_1,
        'acc3d_relax_1': best_acc3d_relax_1,
        'angle_error_1': best_angle_error_1,
        'outlier_1': best_outliers_1,
        'time': time_avg,
        'epoch': best_epoch,
        'solver_time': solver_time,
        'pre_compute_time': pre_compute_time,
    }


    info_dict['dt_query_time'] = dt_query_time
    info_dict['avg_dt_query_time'] = dt_query_time / epoch

    info_dict['network_time'] = net_time
    info_dict['avg_net_time'] = net_time / epoch
    info_dict['net_backward_time'] = net_backward_time
    info_dict['avg_net_backward_time'] = net_backward_time / epoch

    return info_dict


class SceneFlowSolver():

    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.gpu}'

        self.dataloader = build_dataloader(args)
        # This can be done in a better way perhaps?
        self.model = build_model(args)
        self.model.to(self.device)
        self.Loss_Function = build_loss(args)



        name = datetime.datetime.utcnow().isoformat(sep='-', timespec='milliseconds')
        name = name.replace(':', '-')
        name = name.replace('.', '-')

        # self.exp_uuid = uuid.uuid4().hex
        self.exp_dir = EXP_PATH + args.exp_name + '/' + name
        os.makedirs(self.exp_dir, exist_ok=True)

    def optimize_sceneflow(self):

        self.outputs = []

        for batch_id, data in tqdm(enumerate(self.dataloader)):

            if batch_id != 1: continue


            pc1, pc2, gt_flow = data

            pc1 = pc1.to(self.device)
            pc2 = pc2.to(self.device)
            gt_flow = gt_flow.to(self.device)

            # Benchmark area, nsf is not doing that, but still baseline exp is done consistently with others
            radius_mask = pc1.norm(dim=-1) < self.args.max_range
            radius_mask2 = pc2.norm(dim=-1) < self.args.max_range

            pc1 = pc1[:,radius_mask[0]]
            pc2 = pc2[:,radius_mask2[0]]
            gt_flow = gt_flow[:,radius_mask[0]]

            # todo opt weights, scoop


            pred_dict = solver(pc1, pc2, gt_flow, self.model, self.Loss_Function, self.args)


            pred_flow = pred_dict['final_flow']


            # EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error = scene_flow_metrics(gt_flow, )

            store_dict = {'loss' : pred_dict['loss'],
                          'EPE3D' : pred_dict['EPE3D_1'],
                          'acc3d_strict' : pred_dict['acc3d_strict_1'],
                          'acc3d_relax' : pred_dict['acc3d_relax_1'],
                          'angle_error' : pred_dict['angle_error_1'],
                          'outlier' : pred_dict['outlier_1'],
                          'avg_solver_time' : pred_dict['solver_time'],}

            self.outputs.append(dict(list(store_dict.items())[1:]))

            if args.store_inference:
                self.store_inference(pc1, pc2, pred_flow, gt_flow, batch_id)

            if args.dev:
                print(args)
                if args.vis:
                    from vis.deprecated_vis import visualize_flow3d
                    visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), gt_flow[0].detach().cpu().numpy())
                    visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), pred_flow[0].detach().cpu().numpy())
                break   # for development




        self.store_exp()

    def store_inference(self, pc1, pc2, pred_flow, gt_flow, batch_id):

        pc1 = pc1[0].detach().cpu().numpy()
        pc2 = pc2[0].detach().cpu().numpy()
        flow_to_store = pred_flow[0].detach().cpu().numpy()
        gt_to_store = gt_flow[0].detach().cpu().numpy()


        os.makedirs(self.exp_dir + '/inference', exist_ok=True)
        np.savez(self.exp_dir + f'/inference/{batch_id:06d}.npz', pc1=pc1, pc2=pc2, est_flow=flow_to_store,
                 gt_flow=gt_to_store)

    def store_exp(self):
        # metric
        np.save(self.exp_dir + '/args.npy', self.args)
        # print(np.load('test.npy', allow_pickle=True))
        df = pd.DataFrame(self.outputs)
        df.loc['mean'] = df.mean()

        log_file = open(f'{self.exp_dir}/logfile', 'w')

        log_file.writelines(f'---------- {self.exp_dir} ---------- \n')
        log_file.writelines(*[' \n'.join(f'{k}={v}' for k, v in vars(args).items())])
        log_file.writelines(f'\n{df.mean()}\n')
        log_file.writelines(f'---------- end of exp ---------- \n \n')
        log_file.close()

        # print(' \n'.join(f'{k}={v}' for k, v in vars(args).items()))

        print(f'{df.mean()}')

    def optimize_sceneflow_with_hyperparams(self, args):

        raise NotImplemented

# pro ted vyhodit DT kvuli FASTGEODIS?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### General
    parser.add_argument('--dataset', type=str, nargs='+', default='argoverse', help='argo or not yet implemented')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='not yet') #?
    parser.add_argument('--exp_name', type=str, default='dev', help='name of the experiment') #?
    parser.add_argument('--baseline', type=str, default='NSF', help='choose the baseline experiment') #?
    parser.add_argument('--gpu', type=str, default='0', help='choose one gpu resource') #?
    parser.add_argument('--verbose', type=int, default=0, help='print more information') #?
    parser.add_argument('--max_range', type=float, nargs='+', default=35, help='maximum range of LiDAR points') #?
    parser.add_argument('--per_sample_init', type=int, nargs='+', default=1, help='initialize model each time - optimization') #?
    parser.add_argument('--store_inference', type=int, default=0, help="store flow to exp folder")

    # Running
    parser.add_argument('--runs', type=int, default=1, help='start multiple runs for experiment variance') #?
    parser.add_argument('--dev', type=int, default=0, help='try fit to one sample')  # ?
    parser.add_argument('--vis', type=int, default=0, help='visualize one sample')  # ?

    ### LOSS
    # weights
    parser.add_argument('--l_ch', type=float, nargs='+', default=1.) #?
    parser.add_argument('--l_ff', type=float, nargs='+', default=1)  # ?
    parser.add_argument('--l_dt', type=float, nargs='+', default=0)  # ?
    parser.add_argument('--l_sm', type=float, nargs='+', default=1)  # ?
    parser.add_argument('--l_vsm', type=float, nargs='+', default=0)  # ?
    # parser.add_argument('--loss_chamfer_use_visibility', type=bool, default=1) # Freespace?


    # KNN
    parser.add_argument('--normals_K', type=int, nargs='+', default=4)  # ?
    parser.add_argument('--KNN_max_radius', type=float, nargs='+', default=1.5)  # ?
    parser.add_argument('--smooth_K', type=int, nargs='+', default=4)
    parser.add_argument('--pc2_smoothness', type=int, nargs='+', default=0)
    # pcsmoothness weight instead
    # NN dist
    parser.add_argument('--l_ch_bothways', type=bool, nargs='+', default=1)  # ?
    parser.add_argument('--l_dt_gridfactor', type=int, nargs='+', default=10) #?

    # Smoothness
    # Visibility-aware smoothness
    # Forward flow smoothness


    ### Hyperparameters
    parser.add_argument('--lr', type=float, nargs='+', default=0.008, help='learning rate') #?
    parser.add_argument('--iters', type=int, nargs='+', default=5000, help='number of iterations')  # ?
    parser.add_argument('--early_patience', type=int, nargs='+', default=10, help='when to consider convergence') #?
    parser.add_argument('--early_min_delta', type=float, nargs='+', default=0.001, help='convergence difference') #?

    ### Purpose
    parser.add_argument('--use_case', type=str, default='local minima', help='what problems and use-cases it should solve')  # ?
    parser.add_argument('--expectation', type=str, default='beat baseline', help='what do you expect from the experiment')  # ?
    parser.add_argument('--hypothesis', type=str, default='beat baseline', help='what should happen and why it should be better')  # ?

    args = parser.parse_args()

    # different args for 8192 and full point cloud ...

    # maybe in the class experiment?
    argument_list = preprocess_args(args)

    for run in range(args.runs):

        for args in argument_list:  # drop this for now?

            Experiment = SceneFlowSolver(args)
            Experiment.optimize_sceneflow()
