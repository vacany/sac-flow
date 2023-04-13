import numpy as np
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

        fill = int(np.linalg.norm(p1 - p2) / inbetween_dist)

        x = np.linspace(p1[0], p2[0], fill)
        y = np.linspace(p1[1], p2[1], fill)
        z = np.linspace(p1[2], p2[2], fill)

        connecting_line.append(np.stack([x, y, z], axis=1))

    return np.concatenate(connecting_line)


def raycast_pts(pts, sensor_pose=(0,0,0), margin=0.05):

    out = np.split(pts, 1, axis=0)[0]

    ray_correspondences = [np.array((sensor_pose, point)) for point in out]

    all_rays = np.concatenate([create_connecting_pts(ray_correspondence, inbetween_dist=margin) for ray_correspondence in ray_correspondences], axis=0)


    return all_rays


def eliminate_rays_by_margin_distance(rays, point_connection, margin=0.05):

    dist, indices = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(point_connection).kneighbors(rays, return_distance=True)

    kept_rays = rays[dist[:, 0] > margin, :]

    return kept_rays
