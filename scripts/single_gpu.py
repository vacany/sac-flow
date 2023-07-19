import pandas as pd
import sys
import numpy as np
from pipeline.run_utils import run_experiment


if __name__ == "__main__":

    config_path = str(sys.argv[1])
    gpu = int(sys.argv[2])

    exps = pd.read_csv(config_path, index_col=False)
    gpu_mask = np.load(config_path + '.npy')

    run_idx = np.where(gpu_mask == gpu)[0]
    cfg_list = [exps.iloc[i].to_dict() for i in range(len(exps))]
    cfg_list = [cfg_list[i] for i in run_idx]

    for cfg in cfg_list:
        cfg['gpu'] = gpu
        cfg['exp_name'] = 'dump_dev'
        cfg['dev'] = 1
        # print(cfg)

        run_experiment(cfg, DETACH=False)
