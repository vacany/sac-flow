import os
import uuid
import numpy as np
import torch
import pandas as pd
import argparse
import copy
import itertools
import glob
from tqdm import tqdm

from vis.deprecated_vis import *
from data.PATHS import DATA_PATH, TMP_VIS_PATH, EXP_PATH

# optimization
from ops.metric import scene_flow_metrics
from models.FastNSF.optimization import solver_no_opt, Neural_Prior


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

    raise NotImplemented


# optimizer, loss function, solver ...
# todo class experiment directory to handle saving and all? metric jupyter
class SceneFlowSolver():

    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.gpu}'

        self.dataloader = build_dataloader(args)
        # This can be done in a better way perhaps?
        self.model = build_model(args)
        self.model.to(self.device)

        self.exp_dir = EXP_PATH + args.exp_name #+ str(args.iters)
        self.exp_uuid = uuid.uuid4().hex
        os.makedirs(self.exp_dir, exist_ok=True)
    def optimize_sceneflow(self):


        outputs = []

        for batch_id, data in tqdm(enumerate(self.dataloader)):

            # if batch_id == 1: break

            pc1, pc2, gt_flow = data

            pc1 = pc1.to(self.device)
            pc2 = pc2.to(self.device)
            gt_flow = gt_flow.to(self.device)

            # Benchmark area, nsf is not doing that, but still baseline exp is done consistently with others
            radius_mask = pc1.norm(dim=-1) < self.args.max_radius
            radius_mask2 = pc2.norm(dim=-1) < self.args.max_radius

            pc1 = pc1[:,radius_mask[0]]
            pc2 = pc2[:,radius_mask2[0]]
            gt_flow = gt_flow[:,radius_mask[0]]

            # todo opt weights, scoop
            # todo loss

            # init (model), solver(model, loss, iter)

            pred_dict = solver_no_opt(pc1, pc2, gt_flow, self.model,
                                      max_iters=self.args.iters,
                                      use_smoothness=self.args.use_smoothness,
                                      use_visibility_smoothness=self.args.use_visibility_smoothness,
                                      use_forward_flow_smoothness=self.args.use_forward_flow_smoothness,
                                      use_reverse_nn=self.args.use_reverse_nn,
                                      )

            pred_flow = pred_dict['final_flow']


            # EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error = scene_flow_metrics(gt_flow, )

            store_dict = {'loss' : pred_dict['loss'],
                          'EPE3D' : pred_dict['EPE3D_1'],
                          'acc3d_strict' : pred_dict['acc3d_strict_1'],
                          'acc3d_relax' : pred_dict['acc3d_relax_1'],
                          'angle_error' : pred_dict['angle_error_1'],
                          'outlier' : pred_dict['outlier_1'],
                          'avg_solver_time' : pred_dict['solver_time'],}

            outputs.append(dict(list(store_dict.items())[1:]))

            break   # for development

        # metric
        df = pd.DataFrame(outputs)
        df.loc['mean'] = df.mean()


        log_file = open(f'{self.exp_dir}/logfile-{self.exp_uuid}', 'w')

        log_file.writelines(f'---------- {self.exp_dir}/logfile-{self.exp_uuid} ---------- \n')
        log_file.writelines(*[' \n'.join(f'{k}={v}' for k, v in vars(args).items())])
        log_file.writelines(f'\n{df.mean()}\n')
        log_file.writelines(f'---------- end of exp ---------- \n \n')
        log_file.close()

        # print(' \n'.join(f'{k}={v}' for k, v in vars(args).items()))

        print(self.exp_dir, f'{df.mean()}')

    def optimize_sceneflow_with_hyperparams(self, args):

        raise NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### General
    parser.add_argument('--dataset', type=str, nargs='+', default='argoverse', help='argo or not yet implemented')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='not yet') #?
    parser.add_argument('--exp_name', type=str, default='baseline', help='name of the experiment') #?
    parser.add_argument('--baseline', type=str, default='NSF', help='choose the baseline experiment') #?
    parser.add_argument('--gpu', type=str, default='0', help='choose one gpu resource') #?
    parser.add_argument('--verbose', type=int, default=0, help='print more information') #?
    parser.add_argument('--max_radius', type=float, default=35, help='maximum range of LiDAR points') #?

    ### LOSS
    # Smoothness
    parser.add_argument('--use_smoothness', type=int, default=0) #?
    parser.add_argument('--loss_smooth_weight', type=float, default=0) #?
    parser.add_argument('--loss_vis_weight', type=float, default=0) #?

    parser.add_argument('--KNN_vis', type=int, nargs='+', default=8)
    parser.add_argument('--use_visibility_smoothness', type=int, default=0) #?

    parser.add_argument('--loss_forwardflow_weight', type=float, default=0) #?
    parser.add_argument('--loss_reverseNN_weight', type=float, default=0)  # ?
    parser.add_argument('--use_forward_flow_smoothness', type=int, default=0) #?
    parser.add_argument('--use_reverse_nn', type=int, default=0) #?')
    ### Hyperparameters
    parser.add_argument('--iters', type=int, nargs='+', default=1000, help='number of iterations')  # ?
    # KNN, weights of losses etc.

    ### Purpose
    parser.add_argument('--use_case', type=str, default='local minima', help='what problems and use-cases it should solve')  # ?
    parser.add_argument('--expectation', type=str, default='beat baseline', help='what do you expect from the experiment')  # ?
    parser.add_argument('--hypothesis', type=str, default='beat baseline', help='what should happen and why it should be better')  # ?

    args = parser.parse_args()

    # different args for 8192 and full point cloud ...

    # maybe in the class experiment?
    argument_list = preprocess_args(args)

    for args in argument_list:

        Experiment = SceneFlowSolver(args)
        Experiment.optimize_sceneflow()
