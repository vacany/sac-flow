import os

import torch
from pytorch3d.ops.knn import knn_points

def gather_KNN_correspondences_from_flow(pc1, pc2, flow):
    '''

    :param pc1: batched point cloud
    :param pc2: batch point cloud
    :param flow: flow for pc1
    :return: trajectory tensor with matched points for first time in pc1
    trajectory [BS, N, 3] where points in the row along the BS are forming the trajectory
    '''
    M = len(pc1)

    for i in range(M):
        dist, NN, _ = knn_points(pc1[i:i + 1] + gt_flow[i:i + 1], pc2[i:i + 1], K=1)

        # distant matches
        dist > max_distance
        # todo calculate with max_distance, maybe eliminate points? It should continue with trajectory trend? But that also make mistakes
        matched_indices = NN[0, ..., 0]
        NN_indices[i] = matched_indices

    trajectories = torch.zeros(pc1[...].shape, device=device)

    for i in range(M):
        p1 = pc1[i]
        p2 = pc2[i]
        nn1 = NN_indices[i]

        correspondences = p2[nn1]

        trajectories[i] = correspondences

    return trajectories

if __name__ == '__main__':
    from data.dataloader import SFDataset4D
    # os.chdir('/home/vacekpa2/4D-RNSFP')

    device = torch.device('cuda:0')
    M = 8
    dataset = SFDataset4D(dataset_type='argoverse', n_frames=M, only_first=True)

    frame_idx = 1
    data = dataset.__getitem__(frame_idx)

    pc1 = data['pc1'].to(device)
    pc2 = data['pc2'].to(device)
    gt_flow = data['gt_flow'].to(device)
    NN_indices = torch.zeros_like(pc1[..., 0], dtype=torch.long)
    max_distance = 2


    trajectories = gather_KNN_correspondences_from_flow(pc1, pc2, gt_flow)

    single_trajectory = trajectories[:, 0, :]
