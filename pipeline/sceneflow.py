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
from data.seed import seed_everything
from loss.flow import LossModule
from ops.metric import scene_flow_metrics
from models.FastNSF.optimization import Timers, EarlyStopping, init_weights




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

    for add_arg in ['lidar_pose', 'fov_up', 'fov_down', 'H', 'W']:
        setattr(args, add_arg, dataset.data_config[add_arg])

    return dataset, args

def build_model(args):

    if args.model == 'NeuralPrior':
        from models.FastNSF.optimization import Neural_Prior
        net = Neural_Prior()

    elif args.model == 'SCOOP':
        from models.scoopy.get_model import PretrainedSCOOP
        net = PretrainedSCOOP()

    return net

def build_loss(args):

    kwargs = vars(args)

    Loss_Function = LossModule(**kwargs)

    return Loss_Function


# solver
def solver(
        pc1: torch.Tensor,
        pc2: torch.Tensor,
        gt_flow: torch.Tensor,
        net: nn.Module,
        Loss_Function: LossModule,
        args: argparse.Namespace,

):

    timers = Timers()
    timers.tic("solver_timer")

    pre_compute_st = time.time()
    solver_time = 0.

    if hasattr(net, 'update'):
        net = net.eval()
        net.update(pc1, pc2)
        net = net.train()

    if args.per_sample_init:
        net.apply(init_weights)

    for param in net.parameters():
        param.requires_grad = True

    params = net.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0)

    total_losses = []
    total_acc_strit = []
    total_iter_time = []

    epe_all = []
    loss_all = []

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
        # print(net.refinement)   # uci se
        net_time_st = time.time()
        flow_pred_1 = net(pc1)
        net_time = net_time + time.time() - net_time_st
        pc1_deformed = pc1 + flow_pred_1

        # loss = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
        loss = Loss_Function(pc1, flow_pred_1, pc2) # this is loss
        # loss = loss_dict['loss']

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

        loss_all.append(loss.item())
        epe_all.append(EPE3D_1)

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
        # 'loss_dict' : loss_dic,
        'EPE3D_1': best_epe3d_1,
        'acc3d_strict_1': best_acc3d_strict_1,
        'acc3d_relax_1': best_acc3d_relax_1,
        'angle_error_1': best_angle_error_1,
        'outlier_1': best_outliers_1,
        'time': time_avg,
        'epoch': best_epoch,
        'solver_time': solver_time,
        'pre_compute_time': pre_compute_time,
        'loss_all' : loss_all,
        'epe_all' : epe_all
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

        self.device = f'cuda:{args.gpu}'

        self.dataloader, args = build_dataloader(args)
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

        args.exp_dir = self.exp_dir
        self.args = args

    def optimize_sceneflow(self):

        self.outputs = []

        for batch_id, data in tqdm(enumerate(self.dataloader)):


            pc1, pc2, gt_flow, pc_scene = data

            pc1 = pc1.to(self.device)
            pc2 = pc2.to(self.device)
            gt_flow = gt_flow.to(self.device)
            pc_scene = pc_scene.to(self.device)

            # JESUS!!!
            # # Benchmark area, nsf is not doing that, but still baseline exp is done consistently with others, nsf also perform worse here for some reason
            # radius_mask = pc1.norm(dim=-1) < self.args.max_range
            # radius_mask2 = pc2.norm(dim=-1) < self.args.max_range
            #
            # pc1 = pc1[:,radius_mask[0]]
            # pc2 = pc2[:,radius_mask2[0]]
            # gt_flow = gt_flow[:,radius_mask[0]]





            # update loss function
            self.Loss_Function.pc_scene = pc_scene

            pred_dict = solver(pc1, pc2, gt_flow, self.model, self.Loss_Function, self.args)
            print('fuck')

            pred_flow = pred_dict['final_flow']


            store_dict = {'loss' : pred_dict['loss'],
                          'EPE3D' : pred_dict['EPE3D_1'],
                          'acc3d_strict' : pred_dict['acc3d_strict_1'],
                          'acc3d_relax' : pred_dict['acc3d_relax_1'],
                          'angle_error' : pred_dict['angle_error_1'],
                          'outlier' : pred_dict['outlier_1'],
                          'avg_solver_time' : pred_dict['solver_time'],
                          'loss_all': pred_dict['loss_all'],
                          'epe_all': pred_dict['epe_all'],
                          }

            self.outputs.append(dict(list(store_dict.items())[1:]))

            if args.store_inference:
                # store optimization
                self.store_inference(pc1, pc2, pred_flow, gt_flow, batch_id, store_dict)

            if args.dev:

                if args.vis:

                    from vis.deprecated_vis import visualize_flow3d
                    # visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), gt_flow[0].detach().cpu().numpy())
                    visualize_flow3d(pc1[0].detach().cpu().numpy(), pc2[0].detach().cpu().numpy(), pred_flow[0].detach().cpu().numpy())
                break   # for development




        self.store_exp()

    def store_inference(self, pc1, pc2, pred_flow, gt_flow, batch_id, store_dict):

        pc1 = pc1[0].detach().cpu().numpy()
        pc2 = pc2[0].detach().cpu().numpy()
        flow_to_store = pred_flow[0].detach().cpu().numpy()
        gt_to_store = gt_flow[0].detach().cpu().numpy()


        os.makedirs(self.exp_dir + '/inference', exist_ok=True)
        np.savez(self.exp_dir + f'/inference/{batch_id:06d}.npz', pc1=pc1, pc2=pc2, est_flow=flow_to_store,
                 gt_flow=gt_to_store, **store_dict)

    def store_exp(self):
        # metric
        # np.save(self.exp_dir + '/args.npy', self.args)
        # print(np.load('test.npy', allow_pickle=True))
        df = pd.DataFrame(self.outputs, columns=self.outputs[0].keys())
        metric_types = df.columns.tolist()

        final_metric = {}


        for metric_type in metric_types:
            if metric_type in ['loss_all', 'epe_all']: continue
            # print(df[metric_type])

            final_metric[metric_type] = df[metric_type].mean()


        final_df = pd.DataFrame.from_dict(final_metric, orient='index')
        final_df.to_csv(f'{self.exp_dir}/metric.csv', header=False)

        # if self.args.dev:

        print(final_df)

        log_file = open(f'{self.exp_dir}/logfile', 'w')

        log_file.writelines(f'---------- {self.exp_dir} ---------- \n')
        log_file.writelines(*[' \n'.join(f'{k}={v}' for k, v in vars(args).items())])
        log_file.writelines(f'\n{final_df}\n')
        log_file.writelines(f'---------- end of exp ---------- \n \n')
        log_file.close()

        df = pd.DataFrame.from_dict(vars(self.args), orient='index')
        df.to_csv(f'{self.exp_dir}/args.csv', header=False)
        # print(' \n'.join(f'{k}={v}' for k, v in vars(args).items()))


    def optimize_sceneflow_with_hyperparams(self, args):

        raise NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### General
    parser.add_argument('--dataset', type=str, nargs='+', default='kitti_t', help='argo or not yet implemented')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='not yet') #?
    parser.add_argument('--exp_name', type=str, default='dev', help='name of the experiment') #?
    parser.add_argument('--affiliation', type=str, default='none', help='name of the experiment') #?


    parser.add_argument('--gpu', type=str, default='0', help='choose one gpu resource') #?
    parser.add_argument('--verbose', type=int, default=0, help='print more information') #?
    parser.add_argument('--max_range', type=float, nargs='+', default=35, help='maximum range of LiDAR points') #?
    parser.add_argument('--per_sample_init', type=int, nargs='+', default=1, help='initialize model each time - optimization') #?
    parser.add_argument('--store_inference', type=int, default=0, help="store flow to exp folder")

    parser.add_argument('--model', type=str, default='NeuralPrior', help='choose network')  # ?
    # Running
    parser.add_argument('--runs', type=int, default=1, help='start multiple runs for experiment variance') #?
    parser.add_argument('--dev', type=int, default=0, help='try fit to one sample')  # ?
    parser.add_argument('--vis', type=int, default=0, help='visualize one sample')  # ?

    ### LOSS
    # weights
    parser.add_argument('--nn_weight', type=float, nargs='+', default=1., help='Chamfer distance loss weight') #?
    parser.add_argument('--smooth_weight', type=float, nargs='+', default=0)  # ?
    parser.add_argument('--forward_weight', type=float, nargs='+', default=0)  # ?
    parser.add_argument('--free_weight', type=float, nargs='+', default=0)  # ?
    parser.add_argument('--VA', type=float, nargs='+', default=0)  # ?


    # KNN
    parser.add_argument('--K', type=int, nargs='+', default=4)  # ?
    parser.add_argument('--sm_normals_K', type=int, nargs='+', default=0)  # ?
    parser.add_argument('--max_radius', type=float, nargs='+', default=2.5)  # is for chamfer as well!

    parser.add_argument('--pc2_smooth', type=int, nargs='+', default=0)

    # Chamfer
    parser.add_argument('--both_ways', type=bool, nargs='+', default=True)  # ?
    parser.add_argument('--ch_normals_K', type=int, nargs='+', default=0)  # ?


    # parser.add_argument('--l_dt_gridfactor', type=int, nargs='+', default=10) #?

    ### Hyperparameters
    parser.add_argument('--lr', type=float, nargs='+', default=0.008, help='learning rate') #?
    parser.add_argument('--iters', type=int, nargs='+', default=250, help='number of iterations')  # ?
    parser.add_argument('--early_patience', type=int, nargs='+', default=10, help='when to consider convergence') #?
    parser.add_argument('--early_min_delta', type=float, nargs='+', default=0.001, help='convergence difference') #?

    # learning rate scoop for refinement - 0.2


    args = parser.parse_args()
    # SEED
    # seed = seed_everything(seed=42)


    argument_list = preprocess_args(args)

    for run in range(args.runs):

        for args in argument_list:  # drop this for now?

            # print(args)
            Experiment = SceneFlowSolver(args)
            Experiment.optimize_sceneflow()
