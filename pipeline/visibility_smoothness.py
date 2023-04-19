import os.path

import numpy as np
from tqdm import tqdm
import torch
from pytorch3d.ops.knn import knn_points

from data.kitti_sf.general_dataloader import KittiSF_Dataset
from ops.visibility2D import KNN_visibility_solver, substitute_NN_by_mask
from ops.metric import KNN_precision

from vis.deprecated_vis import *

device = torch.device("cuda:0")


# preprocessing function - specific to models
def preprocess_frame(frame, radius=35, remove_ground=True):

    pc1 = frame['pc1']
    pc2 = frame['pc2']
    gt_flow = frame['flow']
    inst_pc1 = frame['inst_pc1']
    inst_pc2 = frame['inst_pc2']
    depth1 = frame['depth1']
    depth2 = frame['depth2']

    valid_mask = frame['valid_mask']
    image_indices = np.stack(valid_mask.nonzero()).T    # here the point clouds are bijective for KITTI_SF

    if remove_ground:
        mask1 = (np.linalg.norm(pc1, axis=1) < radius) & (pc1[:, 1] > -1.4)   # 30cm above ground as in benchmark
        mask2 = (np.linalg.norm(pc2, axis=1) < radius) & (pc2[:, 1] > -1.4)
    else:
        mask1 = np.linalg.norm(pc1, axis=1) < radius
        mask2 = np.linalg.norm(pc2, axis=1) < radius

    # breakpoint()
    return pc1[mask1], pc2[mask2], gt_flow[mask1], inst_pc1[mask1], inst_pc2[mask2], depth1, depth2, image_indices[mask2]

# PARAMS
K = 32

dataset = KittiSF_Dataset()

# frame = dataset[0]
#
# raw_pc1 = frame['pc1']
# raw_pc2 = frame['pc2']
# raw_gt_flow = frame['flow']
# raw_inst_pc1 = frame['inst_pc1']
# raw_inst_pc2 = frame['inst_pc2']
# depth2 = frame['depth2']
#
# pc1, pc2, gt_flow, inst_pc1, inst_pc2, image_indices, depth1, depth2 = preprocess_frame(frame)


res_dict = {'knn_precision' : [],
            'knn_vis_precision' : [],}


for frame_idx in tqdm(range(len(dataset))):

    frame = dataset[frame_idx]
    pc1, pc2, gt_flow, inst_pc1, inst_pc2, depth1, depth2, image_indices = preprocess_frame(frame)


    torch_pc2 = torch.from_numpy(pc2).to(device).unsqueeze(0)
    depth2 = torch.from_numpy(depth2).to(device)
    # inst_pc2 = torch.from_numpy(inst_pc2).to(device).unsqueeze(0)
    image_indices = torch.from_numpy(image_indices).to(device)

    out_nn = knn_points(torch_pc2, torch_pc2, K=K, return_nn=True)
    nn_dist, nn_ind = out_nn[0][0], out_nn[1][0]



    KNN_image_indices = image_indices[nn_ind]

    valid_KNN = KNN_visibility_solver(KNN_image_indices, depth2, margin=0.05)

    # metric
    with torch.no_grad():
        knn_P, incorrect = KNN_precision(nn_ind.detach().cpu().numpy(), instance_mask=inst_pc2)

        new_nn_ind = substitute_NN_by_mask(nn_ind, valid_KNN)
        new_knn_P, new_incorrect = KNN_precision(new_nn_ind.detach().cpu().numpy(), instance_mask=inst_pc2)

    np.save(f"{os.path.expanduser('~')}/data/HPLFlowNet/KITTI_processed_occ_final/{frame_idx:06d}/vis_knn.npy", new_nn_ind.detach().cpu().numpy())

    res_dict['knn_precision'].append(knn_P)
    res_dict['knn_vis_precision'].append(new_knn_P)

print('Original KNN precision: ', f'{np.mean(res_dict["knn_precision"]):.4f}', 'Visibility-aware KNN precision: ', f'{np.mean(res_dict["knn_vis_precision"]):.4f}')
