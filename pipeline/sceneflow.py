import os

import numpy as np
import torch
import pandas as pd
import argparse
import glob

from vis.deprecated_vis import *
from data.PATHS import DATA_PATH, TMP_VIS_PATH, EXP_PATH

# optimization
from ops.metric import scene_flow_metrics
from models.FastNSF.optimization import solver_no_opt, Neural_Prior



# build model, dataloader, optimizer, loss function, solver ...
# todo class experiment directory to handle saving and all?
class SceneFlowSolver():

    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.gpu}'

        # This can be done in a better way perhaps?
        self.baseline_net_prior = Neural_Prior()
        self.baseline_net_prior.to(self.device)

        from data.NSF_data import Argo1_NSF
        self.dataloader = Argo1_NSF()

        self.exp_dir = EXP_PATH + args.exp_name
        os.makedirs(self.exp_dir, exist_ok=True)
    def optimize_sceneflow(self):

        # print(args)

        # for development# design loop here, then refactor to a function later --- nice
        # Not yet complete or even estabilished
        outputs = []

        for batch_id, data in enumerate(self.dataloader):

            # if batch_id == 1: break

            pc1, pc2, gt_flow = data

            pc1 = pc1.to(self.device)
            pc2 = pc2.to(self.device)
            gt_flow = gt_flow.to(self.device)

            # Benchmark area
            radius_mask = pc1.norm(dim=-1) < 35
            radius_mask2 = pc2.norm(dim=-1) < 35

            pc1 = pc1[:,radius_mask[0]]
            pc2 = pc2[:,radius_mask2[0]]
            gt_flow = gt_flow[:,radius_mask[0]]

            # todo do not use prior?
            pred_dict = solver_no_opt(pc1, pc2, gt_flow, self.baseline_net_prior,
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
                          'solver_time' : pred_dict['solver_time'],}

            outputs.append(dict(list(store_dict.items())[1:]))

        # metric
        df = pd.DataFrame(outputs)
        df.loc['mean'] = df.mean()

        log_file = open(f'{self.exp_dir}/logfile', 'w')

        log_file.writelines(*[' \n'.join(f'{k}={v}' for k, v in vars(args).items())])
        log_file.writelines(f'\n{df.mean()}')
        log_file.close()

        # print(' \n'.join(f'{k}={v}' for k, v in vars(args).items()))

        print(self.exp_dir, f'{df.mean()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### General
    parser.add_argument('--dataset', type=str, default='argoverse', help='argo or not yet implemented')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='not yet') #?
    parser.add_argument('--exp_name', type=str, default='baseline', help='name of the experiment') #?
    parser.add_argument('--baseline', type=str, default='NSF', help='choose the baseline experiment') #?
    parser.add_argument('--gpu', type=str, default='0', help='choose one gpu resource') #?
    parser.add_argument('--verbose', type=int, default=0, help='print more information') #?

    ### LOSS
    # Smoothness
    parser.add_argument('--use_smoothness', type=int, default=0) #?
    parser.add_argument('--use_visibility_smoothness', type=int, default=0) #?
    parser.add_argument('--use_forward_flow_smoothness', type=int, default=0) #?
    parser.add_argument('--use_reverse_nn', type=int, default=0) #?')
    ### Hyperparameters
    parser.add_argument('--iters', type=int, default=1000, help='number of iterations')  # ?
    # KNN, weights of losses etc.

    ### Purpose
    parser.add_argument('--use_case', type=str, default='local minima', help='what problems and use-cases it should solve')  # ?
    parser.add_argument('--expectation', type=str, default='beat baseline', help='what do you expect from the experiment')  # ?
    parser.add_argument('--hypothesis', type=str, default='beat baseline', help='what should happen and why it should be better')  # ?

    args = parser.parse_args()

    Experiment = SceneFlowSolver(args)
    Experiment.optimize_sceneflow()
