import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# from vis.deprecated_vis import *


from pytorch3d.ops.knn import knn_points





# substitune NN
def substitute_NN_by_mask(KNN_matrix, valid_KNN_mask):

    indentity_matrix = torch.tile(torch.arange(KNN_matrix.shape[0], device=KNN_matrix.device), (KNN_matrix.shape[1], 1)).T
    new_nn_ind = KNN_matrix.clone()

    new_nn_ind[valid_KNN_mask.detach().cpu().numpy() == False] = indentity_matrix[valid_KNN_mask.detach().cpu().numpy() == False]

    return new_nn_ind

def KNN_visibility_solver(KNN_image_indices, depth, margin=0.05):
    # todo batch multiplication
    '''

    Args:
        KNN_image_indices: image indices of 3D point KNN [N, K, 2]
        depth: Depth image for visibility check [H, W]
        margin: Close distant from linear connection to be considered valid

    Returns: Bool mask of valid KNN [N, K], which can mask original KNN matrix

    '''
    # how to set function for solver?
    # breakpoint()
    K = KNN_image_indices.shape[1]
    # print(K)
    valid_KNN = torch.ones((KNN_image_indices.shape[0], KNN_image_indices.shape[1]), dtype=torch.bool, device=KNN_image_indices.device)
    # print(valid_KNN.shape)
    direction_vector = KNN_image_indices - KNN_image_indices[:,0:1]
    # print(direction_vector.shape)


    max_distance = direction_vector.abs().max()
    # print(max_distance)


    # main visibility solver
    intermediate = torch.linspace(0, 1, max_distance, device=KNN_image_indices.device)

    for increment in intermediate:


        parts = KNN_image_indices - direction_vector * increment

        parts = parts.to(torch.long)


        parts_idx = parts.view(-1, 2)
        curr_connection_depth = depth[parts_idx[:,0], parts_idx[:,1]]
        origins_depth = depth[parts[:,0,0], parts[:,0,1]]


        end_points_depth = depth[parts[:,:,0], parts[:,:,1]]

        # podminka s koncovym bodem, jako linearni interpolace
        curr_linear_depth = origins_depth + (end_points_depth.permute(1,0) - origins_depth) * increment
        curr_linear_depth = curr_linear_depth.permute(1,0)

        # valid connection is the one where linear connection is before the point in image, i.e.
        # depth of connection is smaller than depth of point in image as it comes before
        valid_connection = curr_connection_depth.reshape(-1, K) <= curr_linear_depth + margin

        valid_KNN *= valid_connection


    return valid_KNN
    # visuals


if __name__ == "__main__":
    device = torch.device("cuda:1")

    data_path = os.path.expanduser("~") + '/pcflow/toy_samples/000000.npz'
    data = np.load(data_path)

    depth2 = data['depth2']
    pc2 = data['pc2']
    valid_mask = data['valid_mask']
    pc2_depth = depth2[valid_mask]

    # image Neighboors without invalid
    K = 32
    NN_by_pts = np.zeros((pc2.shape[0], K), dtype=np.int32)

    image_indices = np.stack(valid_mask.nonzero()).T
    vis = depth2.copy()





    torch_pc2 = torch.from_numpy(pc2).to(device).unsqueeze(0)

    out_nn = knn_points(torch_pc2, torch_pc2, K=K, return_nn=True)
    nn_dist, nn_ind = out_nn[0][0], out_nn[1][0]

    nn_ind = nn_ind.cpu().numpy()
    nn_dist = nn_dist.cpu().numpy()

    KNN_image_indices = image_indices[nn_ind]

    chosen_NN = 2132

    chos_ind = KNN_image_indices[chosen_NN]
    vis[chos_ind[:, 0], chos_ind[:, 1]] = 100

    from ops.rays import create_connecting_pts

    chos_mask = np.zeros_like(nn_ind[:, 0])
    chos_mask[nn_ind[chosen_NN]] = 1

    # # vectorized version

    depth2 = torch.from_numpy(depth2).to(device)
    KNN_image_indices = torch.from_numpy(KNN_image_indices).to(device)

    margin = 0.05

    KNN_visibility_solver(KNN_image_indices, depth2, margin=0.05)

    # visualize_one_KNN_in_depth(KNN_image_indices, depth2, chosen_NN, K, output_path='./toy_samples/tmp_vis/out.png')
    # visualize_KNN_connections(pc2, nn_ind, fill_pts=2)
    from ops.metric import KNN_precision
    instance_mask = data['inst_pc2']

    KNN_precision(nn_ind, instance_mask=instance_mask)
    knn_P, incorrect = KNN_precision(nn_ind, instance_mask=instance_mask)
