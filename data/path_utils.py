import torch
from ops.transform import matrix_to_xyz_axis_angle, xyz_rpy_to_matrix
from scipy.spatial.transform import Rotation


__all__ = [
    'get_traj',
    'get_inst_trajes',
    'rel_poses2traj',
    'trajlen',
]

def rel_poses2traj(rel_poses):
    pose = torch.eye(4, dtype=rel_poses.dtype, device=rel_poses.device).unsqueeze(0)
    poses = pose.clone()
    for i in range(len(rel_poses)):
        pose = pose @ torch.linalg.inv(rel_poses[i])
        poses = torch.cat([poses, pose], dim=0)
    return poses


def get_traj(data_sample, axis_angle=True, noise=None):
    """"
    Define trajectory as tensor of shape (n_frames, 6)
    where n_frames is the number of poses
    and each pose is a 6D vector (x, y, z, ax, ay, az)
    """
    poses12 = data_sample['relative_pose']

    # construct path from relative poses
    poses = rel_poses2traj(poses12)
    n_frames = len(poses)
    assert poses.shape == (n_frames, 4, 4)

    # add noise
    if noise is not None:
        poses[:, :3, 3] += torch.randn_like(poses[:, :3, 3]) * noise

    if axis_angle:
        poses = matrix_to_xyz_axis_angle(poses)
        assert poses.shape == (n_frames, 6)

    return poses


def get_inst_trajes(data_sample, axis_angle=False, min_traj_len=None, noise=None, verbose=False):
    trajlen = lambda traj: torch.sqrt((torch.diff(traj[:, :3, 3] - traj[0, :3, 3], dim=0) ** 2).sum(dim=1)).sum()\
        if len(traj) > 1 else 0.
    boxes_list = data_sample['box1']

    # construct path from relative poses
    ego_poses = get_traj(data_sample, axis_angle=False, noise=None)

    # get instances traj observed at time moment 0 from ego pose
    inst_poses = {}
    for t in range(len(boxes_list)):
        for inst_i in range(len(boxes_list[0])):
            if inst_i >= len(boxes_list[t]):
                continue
            uuid = boxes_list[t][inst_i]['uuid']
            pose = torch.eye(4)
            pose[:3, :3] = torch.as_tensor(
                Rotation.from_euler('xyz', boxes_list[t][inst_i]['rotation']).as_matrix())
            pose[:3, 3] = torch.as_tensor(boxes_list[t][inst_i]['translation'])
            pose = ego_poses[t] @ pose
            if uuid not in inst_poses:
                inst_poses[uuid] = pose[None]
            else:
                inst_poses[uuid] = torch.cat([inst_poses[uuid], pose[None]], dim=0)

    if verbose:
        for traj_i, (uuid, traj) in enumerate(inst_poses.items()):
            traj_length = trajlen(traj)
            print(f'Trajectory N {traj_i} of instance {uuid} has {len(traj)} poses and length: {traj_length:.3f} [m]')

    # filter short trajectories
    if min_traj_len is not None:
        inst_poses = {uuid: traj for uuid, traj in inst_poses.items() if trajlen(traj) > min_traj_len}

    # convert to a list of poses
    inst_poses = [traj for uuid, traj in inst_poses.items()]

    # add noise
    if noise is not None:
        for i in range(len(inst_poses)):
            inst_poses[i][:, :3, 3] += torch.randn_like(inst_poses[i][:, :3, 3]) * noise
        ego_poses[:, :3, 3] += torch.randn_like(ego_poses[:, :3, 3]) * noise

    if axis_angle:
        ego_poses = matrix_to_xyz_axis_angle(ego_poses)
        inst_poses = [matrix_to_xyz_axis_angle(traj) for traj in inst_poses]

    return ego_poses, inst_poses


def trajlen(poses):
    assert isinstance(poses, torch.Tensor)
    assert (poses.ndim == 3 and poses.shape[1:] == (4, 4)) or (poses.ndim == 2 and poses.shape[1] == 6)
    if poses.ndim == 2:
        poses = xyz_rpy_to_matrix(poses)
    if len(poses) > 1:
        return torch.sqrt((torch.diff(poses[:, :3, 3] - poses[0, :3, 3], dim=0) ** 2).sum(dim=1)).sum().item()
    else:
        return 0.
