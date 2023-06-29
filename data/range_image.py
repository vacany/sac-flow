import torch

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

def create_depth_img(depth, idx_w, idx_h, proj_H, proj_W, inside_range_img):

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

    return range_image

