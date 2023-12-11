# Imports
import os

os.chdir('/home/vacekpa2/4D-RNSFP')
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.dataloader import SFDataset4D, NSF_dataset
from models.neuralpriors import RigidNeuralPrior, NeuralPrior
from loss.flow import *
from ops.metric import SceneFlowMetric
from vis.deprecated_vis import imshow, visualize_flow3d

from time import time

from benchmark.exp_config import exp_list
# import builtins
# from IPython.lib import deepreload

from models.neuralpriors import *  # RigidNeuralPrior, NeuralPrior, FreespaceRigidNeuralPrior

# todo setup everything with ground truth info to know if that helps []
# todo sceneflow metric for multiple runs []
device = torch.device('cuda:0')



# todo train data?
# todo store config csv
runs = 5
# valid cfg list
cfg_list = []

for early_stop in range(10,210,10):
# for early_stop in range(20,21):
    for run in range(runs):
        for exp_config in exp_list:
            cfg_list.append({**exp_config, 'run': run, 'early_stop': early_stop})

import pandas as pd

cfg_df = pd.DataFrame(cfg_list, columns=cfg_list[0].keys())

print(cfg_df)

exp_nbr = int(sys.argv[1])

exp_config = cfg_list[exp_nbr]

# for exp_config in cfg_list:
# for model_name in ['RigidNeuralPrior', 'NeuralPrior']:  # , 'RigidNeuralPrior']:
model_name = exp_config['model_name']

model = getattr(sys.modules[__name__], model_name)(lr=0.008, early_stop=exp_config['early_stop'], loss_diff=0.001, dim_x=3,
                                                   filter_size=128, layer_size=8).to(device)


run = exp_config['run']
SF_metric = SceneFlowMetric()

# dataset = SFDataset4D(**exp_config)
dataset = NSF_dataset(root_dir='/mnt/personal/vacekpa2/sceneflow/', dataset_type=exp_config['dataset_type'])


print("Processing: ", exp_config['dataset_type'].capitalize())
print('Model: ', model._get_name())

for frame_id, data in enumerate(tqdm(dataset)):

    data['pc1'] = data['pc1'].to(device)
    data['pc2'] = data['pc2'].to(device)
    data['gt_flow'] = data['gt_flow'].to(device)
    # data['relative_pose']
    # data['pose1']
    # data['pose2']
    model.initialize()

    start_time = time()
    output = model(data)
    end_time = time()
    data['eval_time'] = end_time - start_time

    data['pred_flow'] = output['pred_flow']


    SF_metric.update(data)

    # if frame_id == 1: break

# metric_df[model_name].append(SF_metric.get_metric())
print(SF_metric.get_metric().mean())

# version = exp_config['version']
# if model_name in ['NeuralPrior']:
    # version = 'baseline'

df = SF_metric.get_metric()
df['dataset_type'] = exp_config['dataset_type']
df['model_name'] = model_name
df['run'] = run
df['early_stop'] = exp_config['early_stop']

folder = model_name #+ f"-{version}"
save_path = f'/mnt/personal/vacekpa2/experiments/4D-RNSFP/{folder}/'
os.makedirs(save_path + f'/{exp_config["dataset_type"]}', exist_ok=True)
df.to_csv(save_path + f'/{exp_config["dataset_type"]}/metric-{exp_nbr:03d}.csv')

import shutil

ignore_patterns = ['datasets_api', 'pointnet2']
shutil.copytree('.', save_path + 'code', ignore=shutil.ignore_patterns(*ignore_patterns), dirs_exist_ok=True)

# should be last just to avoid all processes writing into same file
cfg_df.to_csv(save_path + f'config.csv')


