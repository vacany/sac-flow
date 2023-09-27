import pandas as pd
import numpy as np
import time
import sys
import os
from data.gpu_utils import wait_for_gpu, get_free_gpu_indices
from pipeline.run_utils import run_experiment


if __name__ == "__main__":

    print("ONLY ON CMP GRID!!!!! \nBe sure to run this with nohup &")

    config_path = str(sys.argv[1])

    exps = pd.read_csv(config_path, index_col=False)

    cfg_list = [exps.iloc[i].to_dict() for i in range(len(exps))]

    total_exp = len(cfg_list)
    # all_gpus = get_free_gpu_indices()
    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]


    # if len(sys.argv) > 1:
    #     print('using gpu ', sys.argv[1])

    # else:
    #     raise ValueError('Please specify a gpu')

    # split list to evenly sized-chunks
    split_nbr = len(cfg_list) / len(all_gpus)


    def chunks(xs, n):
        n = max(1, n)
        return (xs[i:i + n] for i in range(0, len(xs), n))

    # tohle do toho nezapada
    cfg_split_list = [d for d in chunks(cfg_list, int(split_nbr))]

    # assign gpu to each chunk

    for i, cfg_split in enumerate(cfg_split_list):
        for cfg in cfg_split:
            # cfg['gpu'] = all_gpus[i]
            cfg['dev'] = 1  #tmp
            cfg['exp_name'] = 'dump_dev'

    idx_range = np.arange(0, total_exp, 1)
    idx_split = np.array_split(idx_range, len(all_gpus))

    gpu_mask = - np.ones(total_exp, dtype=int)
    for i, idx in enumerate(idx_split):
        gpu_mask[idx] = i

    print(gpu_mask)
    print("Running experiments...")


    np.save(f'{os.path.dirname(config_path)}/{os.path.basename(config_path)}.npy', gpu_mask)



    # def run_one(cfg_split):
    #     gpu = int(sys.argv[1])
    #     for cfg in cfg_split:
    #         cfg['gpu'] = gpu
    #         print(f'Running gpu: {cfg["gpu"]}')
    #
    #         run_experiment(cfg, DETACH=False)

    # run_one(cfg_split_list[gpu])


