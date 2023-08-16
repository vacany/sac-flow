import os.path

import pandas as pd
import sys
import numpy as np
import socket
from pipeline.run_utils import run_experiment

# rci job array, allocation will take time probably
# split to rci and (later) cmp

if __name__ == "__main__":

    # config_path = str(sys.argv[1])

    config_path = f'{os.path.expanduser("~")}/pcflow/configs/experiments/normals_0.csv'
    exps = pd.read_csv(config_path, index_col=False)
    exp_nbr = int(sys.argv[1])
    # gpu_mask = np.load(config_path + '.npy')

    # run_idx = np.where(gpu_mask == gpu)[0]
    # cfg_list = [exps.iloc[i].to_dict() for i in range(len(exps))]
    # cfg_list = [cfg_list[i] for i in run_idx]

    cfg = exps.iloc[exp_nbr].to_dict()

    # if socket.gethostname().startswith() == 'g':
    cfg['gpu'] = 0 # int(sys.argv[1]) boruvka
    cfg['exp_name'] = os.path.basename(config_path).split('.')[0] #+ f'_{exp_nbr}'  # separate datetime from main function?

    print(cfg)

    run_experiment(cfg, DETACH=False)

    ### test
    # for dataset in ['kitti_t', 'kitti_o', 'argoverse', 'nuscenes', 'waymo']:
    #     cfg['dataset'] = dataset
    #     cfg['dev'] = 1
    #     cfg['iter'] = 5
    #     run_experiment(cfg, DETACH=False)

