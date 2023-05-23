import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def create_connecting_pts(pts, inbetween_dist=0.1):
    '''
    Still slow, but works
    Args:
        pts: ORDERED set of points
        fill: nbr of pts in between

    Returns: point cloud of connected points

    '''

    connecting_line = []

    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]

        # breakpoint()
        fill = int(torch.min(p1.float() - p2.float()).abs() / inbetween_dist)

        x = torch.linspace(p1[0], p2[0], fill)
        y = torch.linspace(p1[1], p2[1], fill)

        if len(p1) == 3:
            z = torch.linspace(p1[2], p2[2], fill)
            connecting_line.append(torch.stack([x, y, z], dim=1))
        else:
            connecting_line.append(torch.stack([x, y], dim=1))

    return torch.cat(connecting_line)


def raycast_pts(pts, sensor_pose=(0,0,0), inbeetween_dist=0.1):

    out = np.split(pts, 1, axis=0)[0]

    ray_correspondences = [np.array((sensor_pose, point)) for point in out]

    all_rays = np.concatenate([create_connecting_pts(ray_correspondence, inbetween_dist=inbeetween_dist) for ray_correspondence in ray_correspondences], axis=0)


    return all_rays


def eliminate_rays_by_margin_distance(rays, point_connection, margin=0.05):

    dist, indices = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(point_connection).kneighbors(rays, return_distance=True)

    kept_rays = rays[dist[:, 0] > margin, :]

    return kept_rays


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

