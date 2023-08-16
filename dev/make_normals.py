from pytorch3d.ops.points_normals import estimate_pointcloud_normals
import torch
import numpy as np

our = np.load('/home/vacekpa2/data/sceneflow/kittisf/all_data_format/000162.npz', allow_pickle=True)
normals_K3 = estimate_pointcloud_normals(torch.from_numpy(our['pc1'][None, :]), 3)
normals_K4 = estimate_pointcloud_normals(torch.from_numpy(our['pc1'][None, :]), 3)
np.savez('normals.npz', normals_K3=normals_K3.detach().numpy()[0], normals_K4=normals_K4.detach().numpy()[0], pc=our['pc1'])
# print(normals.shape)
