import numpy as np
import torch

# global_pc1[:, :3] = (np.linalg.inv(pose) @ global_pc1.T).T[:, :3]

def synchronize_poses(pose1, pose2):

    sync_pose = np.linalg.inv(pose1) @ pose2.T
    # back_pts2 = (np.linalg.inv(pose1) @ g_pc2.T).T

    return sync_pose

def transform_pc(pts, pose):

    '''

    :param pts: point cloud
    :param pose: 4x4 transformation matrix
    :return:
    '''
    transformed_pts = np.insert(pts.copy(), 3, 1, axis=1)
    transformed_pts[:, 3] = 1
    transformed_pts[:, :3] = (transformed_pts[:, :4] @ pose.T)[:, :3]

    # transformed_pts[:, 3:] = pts[:, 3:]

    return transformed_pts

def find_weighted_rigid_alignment(A, B, weights, use_epsilon_on_weights=False):
    """
    Calculates the weighted rigid transformation that aligns two sets of points.
    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the first set of points.
        B (torch.Tensor): A tensor of shape (batch_size, num_points, 3) containing the second set of points.
        weights (torch.Tensor): A tensor of shaep (batch_size, num_points) containing weights.
        use_epsilon_on_weights (bool): A condition if to use eps for weights.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 4, 4) containing the rigid transformation matrix that aligns A to B.
    """
    assert (weights >= 0.0).all(), "Negative weights found"
    if use_epsilon_on_weights:
        weights += torch.finfo(weights.dtype).eps
        count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
        not_enough_points = count_nonzero_weighted_points < 3
    else:
        # Add eps if not enough points with weight over zero
        count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
        not_enough_points = count_nonzero_weighted_points < 3
        eps = not_enough_points.float() * torch.finfo(weights.dtype).eps
        weights += eps.unsqueeze(-1)
    assert not not_enough_points, f"pcl0 shape {A.shape}, pcl1 shape {B.shape}, points {count_nonzero_weighted_points}"

    weights = weights.unsqueeze(-1)
    sum_weights = torch.sum(weights, dim=1)

    A_weighted = A * weights
    B_weighted = B * weights

    a_mean = A_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)
    b_mean = B_weighted.sum(axis=1) / sum_weights.unsqueeze(-1)

    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = ((A_c * weights).transpose(1, 2) @ B_c) / sum_weights
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(1, 2)
    # Translation vector
    t = b_mean.transpose(1, 2) - (R @ a_mean.transpose(1, 2))

    T = torch.cat((R, t), dim=2)
    T = torch.cat((T, torch.tensor([[[0,0,0,1]]], device=A.device)), dim=1)
    return T

def try_weighted_kabsch():

    pc1 = torch.rand(1, 100, 3)
    rigid_flow = torch.rand(1, 100, 3, requires_grad=True)
    pc2 = pc1 + 3
    weights = torch.ones(1, pc1.shape[1])

    dist_nn, nearest_indices = NN_local(pc1, pc2, K=5)

    pose = find_weighted_rigid_alignment(pc1, pc1 + rigid_flow, weights)

    deformed_pc1 = pc1 @ pose[:, :3, :3] + pose[:, :3, -1]
    next_rigid_flow = deformed_pc1 - pc1


def NN_local(pc1, pc2, K=1):
    # todo refactor elsewhere
    # Calculate the Euclidean distances between the query point and all the data points
    distances = torch.cdist(pc1, pc2)
    # Get the indices of the K nearest neighbors
    distances_nearest, nearest_indices = torch.topk(distances, k=K, largest=False)

    return distances_nearest, nearest_indices

if __name__ == "__main__":

    try_weighted_kabsch()
