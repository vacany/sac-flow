import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



from pytorch3d.ops.knn import knn_points

def calculate_polar_coords(pc):

    calc_depth = np.linalg.norm(pc[:,:3], axis=1)
    yaw = -np.arctan2(pc[:, 1], pc[:, 0])
    pitch = np.arcsin(pc[:, 2] / (calc_depth + 1e-8))

    return yaw, pitch, calc_depth

def get_range_img_coords(pc, VOF, HOF, ):

    yaw, pitch, calc_depth = calculate_polar_coords(pc)

    # to rads
    fov = VOF * np.pi / 180.0

    fov_down = fov / 2

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    # proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= HOF                              # in [0.0, W]
    proj_y *= VOF                              # in [0.0, H]

    u, v = np.floor(proj_x).astype(int), np.floor(proj_y).astype(int)

    # print(np.unique(proj_y))
    depth_image = - np.ones((HOF, VOF))
    depth_image[u, v] = calc_depth

    return u, v, depth_image


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
        # breakpoint()
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

def KNN_coords(pc, nn_ind, VOF, HOF):
    '''
    :param pc2: point cloud to strip as Nx3
    :param nn_ind: KNN indices as NxK
    :return: Coordinates of KNN in depth image as NxKx2
    '''
    H_coords, V_coords, depth_image = get_range_img_coords(pc.detach().cpu().numpy(), VOF, HOF)

    # plt.imshow(self.depth)
    # plt.savefig('toy_samples/depth.png')

    image_indices = np.stack((V_coords, H_coords)).T
    image_indices = torch.from_numpy(image_indices).to(pc.device)

    KNN_image_indices = image_indices[nn_ind]

    return KNN_image_indices

def strip_KNN_with_vis(pc, nn_ind, VOF, HOF, margin=3):
    '''
    :param pc2: point cloud to strip as Nx3
    :param nn_ind: KNN indices as NxK
    :return: visibility aware KNN indices as NxK (stripped KNNs)
    '''

    _,_, depth = get_range_img_coords(pc.detach().cpu().numpy(), VOF, HOF)

    KNN_image_indices = KNN_coords(pc, nn_ind, VOF, HOF)

    depth = torch.from_numpy(depth).to(pc.device)

    valid_KNN = KNN_visibility_solver(KNN_image_indices, depth.T, margin=margin)

    visibility_aware_KNN = substitute_NN_by_mask(nn_ind, valid_KNN)

    return visibility_aware_KNN

