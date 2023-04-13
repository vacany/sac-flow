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
