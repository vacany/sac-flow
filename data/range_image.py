import importlib
import torch

def pixel2xyz(depth, P_rect, px=None, py=None):
    assert P_rect[0, 1] == 0
    assert P_rect[1, 0] == 0
    assert P_rect[2, 0] == 0
    assert P_rect[2, 1] == 0
    assert P_rect[0, 0] == P_rect[1, 1]
    focal_length_pixel = P_rect[0, 0]

    height, width = depth.shape[:2]
    if px is None:
        px = torch.tile(torch.arange(width, dtype=torch.float)[None, :], (height, 1))
    if py is None:
        py = torch.tile(torch.arange(height, dtype=torch.float)[:, None], (1, width))

    const_x = P_rect[0, 2] * depth + P_rect[0, 3]
    const_y = P_rect[1, 2] * depth + P_rect[1, 3]

    x = ((px * (depth + P_rect[2, 3]) - const_x) / focal_length_pixel)[:, :, None]
    y = ((py * (depth + P_rect[2, 3]) - const_y) / focal_length_pixel)[:, :, None]
    pc = torch.cat((x, y, depth[:, :, None]), dim=-1)

    pc[..., :2] *= -1.

    pc = pc[depth > 0] # JESUS
    pc = pc.reshape(-1, 3)

    return pc

def xyz2pixel(pc, P_rect, height, width):
    # todo check signatures etc.
    '''

    Args:
        pc: In camera depth coordinates (x,y,z) = (z, x, y)
        P_rect:
        height:
        width:

    Returns:

    '''

    x = - pc[:,0]
    y = - pc[:,1]
    z = pc[:,2]

    depth = z   # in this case
    # depth = pc.norm(dim=-1)   # in this case
    focal_length_pixel = P_rect[0, 0]
    # height, width = depth_image.shape[:2]

    const_x = P_rect[0, 2] * depth + P_rect[0, 3]
    const_y = P_rect[1, 2] * depth + P_rect[1, 3]

    px = (x * focal_length_pixel + const_x) / (depth + P_rect[2, 3])
    py = (y * focal_length_pixel + const_y) / (depth + P_rect[2, 3])

    # drop outside image points
    mask = (px > 0) & (py > 0) & (px < width) & (py < height)

    new_depth_image = - torch.ones((height, width), dtype=torch.float, device=pc.device)
    new_depth_image[py[mask].to(int), px[mask].to(int)] = depth[mask]

    return new_depth_image, px, py, mask

def range_image_coords(project_pc, fov_up, fov_down, proj_H, proj_W):

    # laser parameters
    fov_up_rad = fov_up / 180.0 * torch.pi  # field of view up in radians
    fov_down_rad = fov_down / 180.0 * torch.pi  # field of view down in radians
    fov = abs(fov_down_rad) + abs(fov_up_rad)  # get field of view total in radians

    # get depth of all points
    depth = torch.linalg.norm(project_pc[:, :3], dim=1)

    # get angles of all points
    yaw = - torch.arctan2(project_pc[:,1], project_pc[:,0])
    pitch = torch.arcsin(project_pc[:,2] / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / torch.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down_rad)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # indices for range_image
    idx_w = torch.floor(proj_x).long()
    idx_h = torch.floor(proj_y).long()

    inside_range_img = (idx_w >= 0) & (idx_w < proj_W) & (idx_h >= 0) & (idx_h < proj_H)

    return depth, idx_w, idx_h, inside_range_img

def create_depth_img(depth, idx_w, idx_h, proj_H, proj_W, inside_range_img, deterministic=True):

    if deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=False)  # this ...
    ''' just separated so it is not calculated twice '''
    range_image = torch.zeros((proj_H, proj_W), dtype=torch.float32, device=depth.device)
    # minimal depth has a priority

    # Simultaneously reading and writing to the same tensor can lead to random results.
    # https://github.com/pytorch/pytorch/issues/45964
    # torch.use_deterministic_algorithms(mode=True) will set it up in expense of speed?

    order = torch.argsort(depth).flip(0)

    ordered_depth = depth[order]
    ordered_idx_w = idx_w[order]
    ordered_idx_h = idx_h[order]
    ordered_inside_img = inside_range_img[order]

    valid_depth = ordered_depth[ordered_inside_img]
    valid_idx_w = ordered_idx_w[ordered_inside_img]
    valid_idx_h = ordered_idx_h[ordered_inside_img]

    range_image[valid_idx_h, valid_idx_w] = valid_depth

    if deterministic:
        torch.use_deterministic_algorithms(mode=False, warn_only=False)  # this ...

    return range_image

# this can assign minimal depth based on neighboors
def reassign_depth_by_NN(pc, nn_ind, mode='minimal'):

    connected_pc = pc[nn_ind]

    if mode == 'minimal':
        min_depth = torch.norm(connected_pc, dim=-1).min(dim=-1)[0]

    else:
        raise NotImplementedError

    return min_depth

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# from pytorch3d.ops.knn import knn_points

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
    # TODO not visualized!
    valid_KNN = torch.ones((KNN_image_indices.shape[0], KNN_image_indices.shape[1]), dtype=torch.bool, device=KNN_image_indices.device)
    # print(valid_KNN.shape)
    # print(KNN_image_indices.max(), KNN_image_indices.min())

    direction_vector = KNN_image_indices - KNN_image_indices[:,0:1]

    max_distance = direction_vector.abs().max()
    # print(max_distance)

    # main visibility solver

    intermediate = torch.linspace(0, 1, max_distance.long(), device=KNN_image_indices.device)

    for increment in intermediate:

        parts = KNN_image_indices - direction_vector * increment

        parts = parts.to(torch.long)

        parts_idx = parts.view(-1, 2)
        # todo ended here, forgot about inside indices
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

class VisibilityScene():

    def __init__(self, dataset, pc_scene):
        self.dataset = dataset
        self.pc_scene = pc_scene
        # self.H = H
        # self.W = W
        # print('Setting up visibility scene with deterministic algorithms')
        # torch.use_deterministic_algorithms(True)
        if dataset in ['kitti_t', 'kitti_o']:
            self.H = 375
            self.W = 1242


        elif dataset in ['argoverse', 'waymo', 'nuscenes', 'valeo_emre']:

            datamodule = importlib.import_module('data.params.' + dataset)
            self.dataconfig = datamodule.data_config
            # for k, v in self.dataconfig.items():
            #     setattr(self, k, v)

            self.fov_up = self.dataconfig['fov_up']
            self.fov_down = self.dataconfig['fov_down']
            self.H = self.dataconfig['H']
            self.W = self.dataconfig['W']

        else:
            raise NotImplementedError

        self.depth_image = self.calculate_depth_image()
    def visibility_aware_smoothness_KNN(self, nn_ind):


        px, py = self.calculate_image_coors(self.pc_scene)

        # depth, w, h, inside_mask = self.generate_range_coors(pc)

        image_indices = torch.stack((py, px)).T  # is it the same order?

        KNN_image_indices = image_indices[nn_ind[0]].long()


        valid_KNN = KNN_visibility_solver(KNN_image_indices, self.depth_image, margin=0.5)
        VA_KNN = substitute_NN_by_mask(nn_ind[0], valid_KNN)

        return VA_KNN

    # def generate_range_coors(self, pc):
    #
    #     if self.dataset in ['kitti_t', 'kitti_o']:
    #         px, py = self.calculate_image_coors(pc[0])
    #
    #     elif self.dataset in ['argoverse', 'waymo', 'nuscenes']:
    #         print('not yet implemented')
    #         px, py = range_image_coords(pc[0], self.fov_up, self.fov_down, self.H, self.W)
    #
    #     else:
    #         raise NotImplementedError
    #
    #     use_min_depth = True
    #     if use_min_depth:
    #         pass
    #
    #     return px, py


    @staticmethod
    def calculate_polar_coords(pc):

        calc_depth = pc[:,:3].norm(dim=1)
        yaw = - torch.arctan2(pc[:, 1], pc[:, 0])
        pitch = torch.arcsin(pc[:, 2] / (calc_depth + 1e-8))

        return yaw, pitch, calc_depth

    def calculate_image_coors(self, pc):

        yaw, pitch, depth = self.calculate_polar_coords(pc)

        # depth_image = - torch.ones((H, W), dtype=torch.float, device=pc.device)
        px = (yaw - yaw.min()) / self.vert_fov * (self.W - 1)
        py = (pitch - pitch.min()) / self.hor_fov * (self.H - 1)

        return px, py

    def calculate_depth_image(self):

        yaw, pitch, depth = self.calculate_polar_coords(self.pc_scene)

        reverse_mask = depth.argsort().flip(0)
        # only for depth image creation
        yaw_rev = yaw[reverse_mask]
        pitch_rev = pitch[reverse_mask]
        depth_rev = depth[reverse_mask]

        # pocitej s xyz
        self.vert_fov = (yaw_rev.max() - yaw_rev.min()).abs()
        self.hor_fov = (pitch_rev.max() - pitch_rev.min()).abs()

        # depth_image = - torch.ones((H, W), dtype=torch.float, device=pc.device)
        px = (yaw_rev - yaw_rev.min()) / self.vert_fov * (self.W - 1)
        py = (pitch_rev - pitch_rev.min()) / self.hor_fov * (self.H - 1)

        self.depth_image = - torch.ones((self.H, self.W), dtype=torch.float, device=self.pc_scene.device)
        self.depth_image[py.long(), px.long()] = depth_rev

        return self.depth_image

    def assign_depth_to_flow(self, pc_flow):
        # more fx,fy
        fx, fy = self.calculate_image_coors(pc_flow)

        inside_mask = (fx > 0) & (fy > 0) & (fx < self.W) & (fy < self.H)

        image_depth = - torch.ones(pc_flow.shape[0], dtype=torch.float, device=pc_flow.device)
        image_depth[inside_mask] = self.depth_image[fy[inside_mask].long(), fx[inside_mask].long()]

        return image_depth



