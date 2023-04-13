import numpy as np
from sklearn.neighbors import NearestNeighbors

from ops.rays import create_connecting_pts, raycast_pts, eliminate_rays_by_margin_distance


def transfer_voxel_visibility(accum_freespace : np.ndarray, global_pts, cell_size):
    '''

    :param accum_freespace: Accumulated freespace feature
    :param global_pts: Point cloud of interest
    :param cell_size: Voxel size
    :return: Mask of points that are visible from the lidar
    '''
    # calculate max and min coordinates of global pcl list
    # acc_x_min, acc_x_max = np.min(accum_freespace[:, 0]), np.max(accum_freespace[:, 0])
    # acc_y_min, acc_y_max = np.min(accum_freespace[:, 1]), np.max(accum_freespace[:, 1])
    # acc_z_min, acc_z_max = np.min(accum_freespace[:, 2]), np.max(accum_freespace[:, 2])

    x_min, x_max = np.min(global_pts[:, 0]), np.max(global_pts[:, 0])
    y_min, y_max = np.min(global_pts[:, 1]), np.max(global_pts[:, 1])
    z_min, z_max = np.min(global_pts[:, 2]), np.max(global_pts[:, 2])

    # x_min, x_max = np.min([x_min, acc_x_min]), np.max([x_max, acc_x_max])
    # y_min, y_max = np.min([y_min, acc_y_min]), np.max([y_max, acc_y_max])
    # z_min, z_max = np.min([z_min, acc_z_min]), np.max([z_max, acc_z_max])


    filtered_accum = accum_freespace[(accum_freespace[:,0] > x_min) & (accum_freespace[:,0] < x_max) & \
                                        (accum_freespace[:,1] > y_min) & (accum_freespace[:,1] < y_max) & \
                                        (accum_freespace[:,2] > z_min) & (accum_freespace[:,2] < z_max)]


    voxel_grid = np.zeros((int((x_max - x_min) / cell_size[0] + 2), int((y_max - y_min) / cell_size[1] + 2),
                           int((z_max - z_min) / cell_size[2] + 2)))

    # not indices, but coordinates...
    global_pts_idx = np.round((global_pts[:, :3] - np.array([x_min, y_min, z_min])) / cell_size).astype(int)
    filtered_accum_idx = np.round((filtered_accum[:, :3] - np.array([x_min, y_min, z_min])) / cell_size).astype(int)

    voxel_grid[filtered_accum_idx[:,0].astype(int), filtered_accum_idx[:,1].astype(int), filtered_accum_idx[:,2].astype(int)] += 1

    static_mask = voxel_grid[global_pts_idx[:,0], global_pts_idx[:,1], global_pts_idx[:,2]]

    return static_mask



def visibility_freespace(curr_pts, pose, cfg):
    '''
    Local point cloud and lidar position with respect to the point local frame. Then it is consistent.
    :param curr_pts: point cloud for raycasting
    :param pose: pose of lidar from where the beams are raycasted
    :param cfg: config file with cell sizes etc.
    :return: point cloud of safely visible areas based on lidar rays
    '''
    assert len(curr_pts.shape) == 2

    cell_size = cfg['cell_size']
    size_of_block = cfg['size_of_block']
    # Sort the point from closest to farthest
    distance = np.sqrt(curr_pts[..., 0] ** 2 + curr_pts[..., 1] ** 2 + curr_pts[..., 2] ** 2)
    index_by_distance = distance.argsort()
    curr_pts = curr_pts[index_by_distance]

    # Get the boundaries of the raycasted point cloud
    x_min, x_max, y_min, y_max, z_min, z_max = cfg['x_min'], cfg['x_max'], cfg['y_min'], cfg['y_max'], cfg['z_min'], cfg['z_max']

    x_min -= 2
    y_min -= 2
    z_min -= 1

    x_max += 2
    y_max += 2
    z_max += 1

    # Create voxel grid
    xyz_shape = np.array(
            (np.round((x_max - x_min) / cell_size[0]) + 3,
             np.round((y_max - y_min) / cell_size[1]) + 3,
             np.round((z_max - z_min) / cell_size[2]) + 3),
            dtype=int)

    # 0 is no stat, -1 is block, 1 is free, 2 is point
    cur_xyz_voxel = np.zeros(xyz_shape)
    accum_xyz_voxel = np.zeros(xyz_shape)

    curr_xyz_points = np.array(np.round(((curr_pts[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
    cur_xyz_voxel[curr_xyz_points[:, 0], curr_xyz_points[:, 1], curr_xyz_points[:, 2]] = 2


    # Iterate one-by-one and update the voxel grid with visibility and blockage for next rays
    for p in curr_pts:
        # Calculate number of intermediate points based on the cell size of voxel grid
        nbr_inter = int(cfg['x_max'] / cell_size[0])
        # Raycast the beam from pose to the point
        ray = np.array((np.linspace(pose[0], p[0], nbr_inter),
                       np.linspace(pose[1], p[1], nbr_inter),
                          np.linspace(pose[2], p[2], nbr_inter))).T

        # Transform the ray to voxel grid coordinates
        xyz_points = np.array(np.round(((ray[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
        # xyz_points = xyz_points[((xyz_points != xyz_points[-1]).all(1)) & ((xyz_points != xyz_points[0]).all(1))] # leave last and first cell
        # xyz_points = xyz_points[(xyz_points[:,2] != xyz_points[-1,2] - 1) &
        #                         (xyz_points[:,2] != xyz_points[-1,2]) &
        #                         (xyz_points[:,2] != xyz_points[-1,2] + 1)]

        # find the intersection of ray and current status of voxels
        ray_stats = cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]]

        if len(xyz_points) == 0: continue  # if the ray is eliminated by security checks

        # Take last point of the ray and create blockage around it for other rays (to create occlusion)
        last_ray_pts = xyz_points[-1]
        cur_xyz_voxel[last_ray_pts[0] - (size_of_block + 1): last_ray_pts[0] + size_of_block,
        last_ray_pts[1] - (size_of_block + 1): last_ray_pts[1] + size_of_block,
        last_ray_pts[2] - (size_of_block + 1): last_ray_pts[2] + size_of_block] = - 1

        # Take only the part of ray before the blockage
        if (ray_stats == -1).any():
            # find the first intersection index
            first_intersection = (np.where(ray_stats == -1)[0][0])
            xyz_points = xyz_points[:first_intersection]

        # Update voxel grid with the visibility of the ray
        cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] = 1
        accum_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] += 1

    # point_coords = np.argwhere(cur_xyz_voxel == 2)
    ray_coords = np.argwhere(cur_xyz_voxel == 1)
    # blocks_coords = np.argwhere(cur_xyz_voxel == -1)

    accum_freespace_feature = accum_xyz_voxel[ray_coords[:,0], ray_coords[:,1], ray_coords[:,2]]

    # restore the original coordinates in meters and add x,y,z, freespace feature
    accum_freespace_meters = ray_coords[:, :3] * cell_size + np.array((x_min, y_min, z_min))
    accum_freespace_meters = np.insert(accum_freespace_meters, 3, accum_freespace_feature, axis=1)

    return accum_freespace_meters

def KNN_outside_freespace(KNN_rays, freespace, margin=1, min_ray_dist=0.1, device='cpu'):

    KNN_ray = torch.rand(K, 100, 3, device=device) * 1

    # get boundaries of the ray point cloud
    x_min, y_min, z_min = torch.min(KNN_ray, dim=1)[0].min(dim=0)[0] - margin
    x_max, y_max, z_max = torch.max(KNN_ray, dim=1)[0].max(dim=0)[0] + margin

    # get freespace indices that are inside the ray point cloud
    freespace_idx = (freespace[:, :, 0] >= x_min) & (freespace[:, :, 0] <= x_max) & \
                    (freespace[:, :, 1] >= y_min) & (freespace[:, :, 1] <= y_max) & \
                    (freespace[:, :, 2] >= z_min) & (freespace[:, :, 2] <= z_max)


    curr_freespace = freespace[freespace_idx]

    distance_matrix = torch.cdist(curr_freespace, KNN_ray)

    closest_K_dist = torch.min(distance_matrix.flatten(1, 2), dim=1)[0]

    keep_NN = closest_K_dist > min_ray_dist

    return keep_NN


def raycast_NN(pts, KNN, fill_pts=10):
    '''

    :param pts: Points Nx3
    :param KNN: Nearest neighboor indices NxK
    :param fill_pts: number of intermediate points to fill the ray
    :return: all rays as array of points, corresponding indices [NK x 2] to the rays for furthr masking
    '''
    all_NN_pts = pts[KNN]
    K = KNN.shape[1]

    all_rays = []

    rolled_NN_pts = np.concatenate(all_NN_pts, axis=0)

    rolled_pts = pts.repeat(K, axis=0)  # I need to repeat here!

    indices_N = np.arange(len(pts)).repeat(K)
    indices_K = np.tile(np.arange(K), len(pts))

    indices = np.stack((indices_N, indices_K)).T

    for i in range(fill_pts):
        ray = rolled_pts + (rolled_NN_pts - rolled_pts) * (i / fill_pts)
        all_rays.append(ray)

    all_indices = np.tile(indices, (fill_pts, 1))
    all_rays_array = np.concatenate(all_rays, axis=0)

    return all_rays_array, all_indices





def smooth_loss(est_flow, NN_idx, loss_norm=1, mask=None):
    # todo add mask
    bs, n, c = est_flow.shape
    K = NN_idx.shape[1]

    est_flow_neigh = est_flow.view(bs * n, c)
    est_flow_neigh = est_flow_neigh[NN_idx.view(bs * n, K)]
    # est_flow_neigh = est_flow_neigh[:, 1:K+1, :]
    flow_diff = est_flow.view(bs * n, c) - est_flow_neigh.permute(1,0,2)

    flow_diff = (flow_diff).norm(p=loss_norm, dim=2)
    smooth_flow_loss = flow_diff.mean()
    smooth_flow_per_point = flow_diff.mean(dim=0).view(bs, n)

    return smooth_flow_loss, smooth_flow_per_point

def margin_visibility_freespace(curr_pts, pose, cfg, margin=0.1):
    '''
    Local point cloud and lidar position with respect to the point local frame. Then it is consistent.
    :param curr_pts: point cloud for raycasting
    :param pose: pose of lidar from where the beams are raycasted
    :param cfg: config file with cell sizes etc.
    :return: point cloud of safely visible areas based on lidar rays
    '''
    assert len(curr_pts.shape) == 2

    cell_size = cfg['cell_size']
    size_of_block = cfg['size_of_block']
    # Sort the point from closest to farthest
    distance = np.sqrt(curr_pts[..., 0] ** 2 + curr_pts[..., 1] ** 2 + curr_pts[..., 2] ** 2)
    index_by_distance = distance.argsort()
    curr_pts = curr_pts[index_by_distance]

    # Get the boundaries of the raycasted point cloud
    x_min, x_max, y_min, y_max, z_min, z_max = cfg['x_min'], cfg['x_max'], cfg['y_min'], cfg['y_max'], cfg['z_min'], cfg['z_max']

    x_min -= 2
    y_min -= 2
    z_min -= 1

    x_max += 2
    y_max += 2
    z_max += 1

    # Create voxel grid
    xyz_shape = np.array(
            (np.round((x_max - x_min) / cell_size[0]) + 3,
             np.round((y_max - y_min) / cell_size[1]) + 3,
             np.round((z_max - z_min) / cell_size[2]) + 3),
            dtype=int)

    # 0 is no stat, -1 is block, 1 is free, 2 is point
    cur_xyz_voxel = np.zeros(xyz_shape)
    accum_xyz_voxel = np.zeros(xyz_shape)

    curr_xyz_points = np.array(np.round(((curr_pts[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
    cur_xyz_voxel[curr_xyz_points[:, 0], curr_xyz_points[:, 1], curr_xyz_points[:, 2]] = 2

    # KNN_to_pc = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(curr_pts)
    # Iterate one-by-one and update the voxel grid with visibility and blockage for next rays
    for p in curr_pts:
        # Calculate number of intermediate points based on the cell size of voxel grid
        nbr_inter = int(cfg['x_max'] / cell_size[0])
        # Raycast the beam from pose to the point
        # if pose[2] > p[2] + margin:
        #     ray = np.array((np.linspace(pose[0], p[0], nbr_inter),
        #                    np.linspace(pose[1], p[1], nbr_inter),
        #                       np.linspace(pose[2], p[2], nbr_inter))).T
        #
        # elif pose[2] < p[2] - margin:   # still not finished, also not for backward radius
        #     ray = np.array((np.linspace(pose[0], p[0] + margin, nbr_inter),
        #                 np.linspace(pose[1], p[1] + margin, nbr_inter),
        #                 np.linspace(pose[2], p[2] + margin, nbr_inter))).T

        # else:
        ray = np.array((np.linspace(pose[0], p[0], nbr_inter),
                        np.linspace(pose[1], p[1], nbr_inter),
                        np.linspace(pose[2], p[2], nbr_inter))).T

        # ray = ray[(np.abs(ray - p) > margin).any(1)]  # remove points that are too close to the point axis-wise (per coordinate)
        ray = ray[(np.abs(ray - p) > margin).all(1)]  # remove points that are too close to the point

        # if ray.shape[0] < 1:
        #     continue
        #
        # dist, nn = KNN_to_pc.kneighbors(ray, return_distance=True)
        #
        # if np.max(dist) < margin:
        #     continue

        # Transform the ray to voxel grid coordinates

        xyz_points = np.array(np.round(((ray[:, :3] - np.array((x_min, y_min, z_min))) / cell_size)), dtype=int)
        # xyz_points = xyz_points[((xyz_points != xyz_points[-1]).all(1)) & ((xyz_points != xyz_points[0]).all(1))] # leave last and first cell
        # xyz_points = xyz_points[(xyz_points[:,2] != xyz_points[-1,2] - 1) &
        #                         (xyz_points[:,2] != xyz_points[-1,2]) &
        #                         (xyz_points[:,2] != xyz_points[-1,2] + 1)]

        # find the intersection of ray and current status of voxels
        ray_stats = cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]]


        # if len(xyz_points) == 0: continue  # if the ray is eliminated by security checks
        #
        # # Take last point of the ray and create blockage around it for other rays (to create occlusion)
        # last_ray_pts = xyz_points[-1]
        # cur_xyz_voxel[last_ray_pts[0] - (size_of_block + 1): last_ray_pts[0] + size_of_block,
        # last_ray_pts[1] - (size_of_block + 1): last_ray_pts[1] + size_of_block,
        # last_ray_pts[2] - (size_of_block + 1): last_ray_pts[2] + size_of_block] = - 1
        #
        # # Take only the part of ray before the blockage
        # if (ray_stats == -1).any():
        #     # find the first intersection index
        #     first_intersection = (np.where(ray_stats == -1)[0][0])
        #     xyz_points = xyz_points[:first_intersection]

        # Update voxel grid with the visibility of the ray
        cur_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] = 1
        accum_xyz_voxel[xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]] += 1

    # point_coords = np.argwhere(cur_xyz_voxel == 2)
    ray_coords = np.argwhere(cur_xyz_voxel == 1)
    # blocks_coords = np.argwhere(cur_xyz_voxel == -1)

    accum_freespace_feature = accum_xyz_voxel[ray_coords[:,0], ray_coords[:,1], ray_coords[:,2]]

    # restore the original coordinates in meters and add x,y,z, freespace feature
    accum_freespace_meters = ray_coords[:, :3] * cell_size + np.array((x_min, y_min, z_min))
    accum_freespace_meters = np.insert(accum_freespace_meters, 3, accum_freespace_feature, axis=1)

    return accum_freespace_meters

def mask_KNN_by_visibility(pts, K, cfg, margin=0.15):

    KNN_pts = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(pts).kneighbors(pts[:, :3], return_distance=True)


    # free_pts = margin_visibility_freespace(pts, pose=(0, 0, 0), cfg=cfg, margin=margin)
    free_pts = visibility_freespace(pts, pose=(0, 0, 0), cfg=cfg)

    close_freespace = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts).kneighbors(free_pts[:, :3],
                                                                                               return_distance=True)
    dist_to_freespace = close_freespace[0][:, 0]
    filtered_free_pts = free_pts[dist_to_freespace >= margin]


    ray_array, all_indices = raycast_NN(pts, KNN_pts[1], fill_pts=10)

    disconnect_mask = transfer_voxel_visibility(filtered_free_pts, ray_array, cfg['cell_size'])

    # filter NN by all indices masking from disconnected mask
    invalid_indices = all_indices[disconnect_mask==1]

    masked_KNN = KNN_pts[1].copy()
    masked_KNN[invalid_indices[:,0], invalid_indices[:,1]] = invalid_indices[:,0]

    return masked_KNN

if __name__ == "__main__":
    from vis.deprecated_vis import visualize_multiple_pcls, visualize_points3D
    import torch


    # TODO key is to do visibility right! Probably sim?
    # todo do visibility by NN of "image" pixels and create area by making every neighbor visible to each other. Margin based on ground accuracy or angle?

    # Todo Use visibility to detect ground and assign rigid transform
    # todo check why the point are shifted


    data_path = '/home/patrik/rci/data/kitti_sf/new/000000.npz'
    data = np.load(data_path)

    # visualize_points3D(data['pc2'], data['px2'])

    # column of pts
    pc2 = data['pc2'][:, [0, 2, 1]] # swap
    valid_mask = np.linalg.norm(pc2, axis=1) < 35

    chosen_px = 776.
    chosen_py = 500.


    mask = (data['px2'] > chosen_px - 5) & (data['px2'] < chosen_px + 5) & (valid_mask) #&\
           # (data['py2'] > chosen_py) & (data['py2'] < chosen_py + 1)

    pc2_col = pc2[mask].copy()

    # visualize_multiple_pcls(*[pc2_col, pc2])
    # fill intermediate point in pc2_col linearly
    # pc2_col = np.concatenate([pc2_col, np.array([[0, 0, 0]])], axis=0)

    # sort by distance
    # pc2_col[:, 2] = pc2_col[pc2_col[:,2].argsort()[::1]]
    # visualize_points3D(pc2_col, pc2_col[:, 0].argsort())
    # pc2_col[:, 2].argsort()
    visualize_points3D(pc2_col, np.arange(pc2_col.shape[0]))

    # pc2_col = np.sort(pc2_col, axis=0)  # sort by dist?


    sensor_pose = np.array([0, 0, 0])
    margin = 0.05


    # raycast point cloud

    # eliminate by NN

    K = 16
    col_NN_dist, col_NN_ind = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(pc2_col).kneighbors(pc2_col, return_distance=True)

    rays_NN, indices_NN = raycast_NN(pc2_col, col_NN_ind, fill_pts=100)

    # eliminate NN rays from rays


    # apply multiprocessing per-pixel
    # use rays from whole pc?

    # margin chosen by data so nearby point do not get eliminated

    point_connection = create_connecting_pts(pc2_col, inbetween_dist=margin)
    all_rays = raycast_pts(pc2_col, sensor_pose=(0, 0, 0), margin=margin)
    kept_rays = eliminate_rays_by_margin_distance(all_rays, point_connection, margin=margin * 2)

    close_dist_NN, close_indices_NN = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(kept_rays).kneighbors(rays_NN, return_distance=True)

    kept_rays_mask = close_dist_NN[:, 0] > margin


    invalid_indices = indices_NN[kept_rays_mask == 0]

    masked_KNN = col_NN_ind.copy()
    masked_KNN[invalid_indices[:, 0], invalid_indices[:, 1]] = invalid_indices[:, 0]

    masked_rays_NN, _ = raycast_NN(pc2_col, masked_KNN, fill_pts=100)

    # distance margin

    # visualize_multiple_pcls(*[kept_rays, point_connection, rays_NN, pc2_col], bg_color=(1, 1, 1, 1))
    visualize_multiple_pcls(*[kept_rays, point_connection, masked_rays_NN, pc2_col, pc2], bg_color=(1, 1, 1, 1))
