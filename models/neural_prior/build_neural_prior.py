import numpy as np
import torch
import argparse
import os.path

from models.neural_prior.optimization import solver

from models.neural_prior.model import Neural_Prior

from models.neural_prior import config

from vis.deprecated_vis import visualize_flow3d, visualize_points3D
from ops.transform import transform_pc

parser = argparse.ArgumentParser(description="Neural Scene Flow Prior.")
config.add_config(parser)
options = parser.parse_args()


device = torch.device("cuda:2")
net = Neural_Prior(filter_size=options.hidden_units, act_fn=options.act_fn, layer_size=options.layer_size).to(device)

data_root = f"{os.path.expanduser('~')}/data/petr_waymo/0001/00"

pc1 = np.fromfile(f"{data_root}/velodyne/00.bin", dtype=np.float32).reshape(-1, 4)
pc2 = np.fromfile(f"{data_root}/velodyne/01.bin", dtype=np.float32).reshape(-1, 4)
pc1[:, 3] = 1
pc2[:, 3] = 1

pose1 = np.load(f"{data_root}/pose/00.npy")
pose2 = np.load(f"{data_root}/pose/01.npy")

pc1 = transform_pc(pc1, pose1)[:, :3]
pc2 = transform_pc(pc2, pose2)[:, :3]

# to same center
pc1 -= pose1[:3, 3]
pc2 -= pose1[:3, 3]

supervox1 = np.fromfile(f"{data_root}/supervoxel/00.bin", dtype=np.int32)
supervox2 = np.fromfile(f"{data_root}/supervoxel/01.bin", dtype=np.int32)

ground_mask1 = np.load(f"{data_root}/ground_mask/00.npy")
ground_mask2 = np.load(f"{data_root}/ground_mask/01.npy")
# remove ground points
pc1 = pc1[ground_mask1==False]
pc2 = pc2[ground_mask2==False]

supervox1 = supervox1[ground_mask1==False]
supervox2 = supervox2[ground_mask2==False]

supervox_inst = 29






visualize_points3D(pc1, supervox1)

# visualize_points3D(pc1, ground_mask1)
# subsample pc1 and pc2
# nbr_pts = min(pc1.shape[0], pc2.shape[0])
nbr_pts = 2048
sup_pc = pc1[supervox1==supervox_inst]

center = sup_pc.mean(0)

pc1 = pc1[np.linalg.norm(pc1-center, axis=1) < 5]
pc2 = pc2[np.linalg.norm(pc2-center, axis=1) < 5]

visualize_points3D(pc2)
# breakpoint()

pc1 = torch.from_numpy(pc1).unsqueeze(0).float()
pc2 = torch.from_numpy(pc2).unsqueeze(0).float()

flow = torch.zeros(1, pc1.shape[0], 3)

pc1 = pc1.to(device)
pc2 = pc2.to(device)
flow = flow.to(device)



info_dict = solver(pc1, pc2, flow, options, net, 1)

pred_flow = info_dict['pred_flow']


print(info_dict)


visualize_flow3d(pc1[0].cpu().numpy(), pc2[0].cpu().numpy(), pred_flow[0].detach().cpu().numpy())
