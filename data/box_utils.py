import numpy as np
import torch
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, Delaunay
from ops.transform import xyz_rpy_to_matrix
# from vis.mayavi_interactive import draw_coord_frame
# from mayavi import mlab


box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def get_ego_points(poses, box, cell_size=(0.1, 0.1)):

    coors = []

    for pose in poses:

        y = np.linspace(-box[4] / 2, box[4] / 2, int(box[4] / cell_size[1] * 2))

        for j in y:
            x = np.linspace(- box[3] / 2, box[3] / 2, int(box[3] / cell_size[0] * 2))
            ego_points = np.insert(x[:, np.newaxis], obj=1, values=j, axis=1)
            ego_points = np.insert(ego_points, obj=2, values=1, axis=1)
            # Rotation
            ego_points = ego_points @ pose[:3, :3].T
            # Translation
            ego_points[:, :2] = ego_points[:, :2] + pose[:2, -1]

            coors.append(ego_points)

    coors = np.concatenate(coors)

    return coors

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def connect_3d_corners(bboxes, fill_points=10, add_label=None):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    corners = boxes_to_corners_3d(bboxes)

    point_list = []
    line = np.linspace(0, 1, fill_points)

    for id, box in enumerate(corners):
        for i in range(len(box)):
            if i != 3 and i != 7:
                points = box[i] + (box[i + 1] - box[i]) * line[:, None]
            if i < 4:
                points = box[i] + (box[i + 4] - box[i]) * line[:, None]
                # point_list.append(points)
            if i in [3, 7]:
                points = box[i] + (box[i - 3] - box[i]) * line[:, None]
                # point_list.append(points)
            points = np.insert(points, 3, id, axis=1)
            point_list.append(points)

    points = np.concatenate(point_list)
    if add_label is not None:
        points = np.insert(points, 3, add_label, axis=1)

    return points

def concatenate_box_pcl(boxes, pcl, label, box_label=1):
    box_points = connect_3d_corners(boxes)
    box_points = np.insert(box_points, 3, box_label, axis=1)
    pcl = np.concatenate((pcl[:,:3], box_points[:,:3]))
    label = np.concatenate((label, box_points[:,3]))

    return pcl, label

def get_bbox_points(bboxes, feature_values=None, fill_points = 30):
    '''

    :param bbox: (N ; x,y,z,l,w,h,yaw)
           feature_values: Features assigned to the box, input per-box array/list
    :return: point cloud of box: x,y,z,l
    '''
    bbox_vis = connect_3d_corners(bboxes, fill_points=fill_points)
    bbox_vis = np.insert(bbox_vis, 3, 1, axis=1)  # class label

    if feature_values is not None:
        nbr_pts_per_box = fill_points * 12
        pts_feature = np.concatenate([np.ones(nbr_pts_per_box) * feat for feat in feature_values])

        bbox_vis = np.insert(bbox_vis, 3, pts_feature, axis=1)  # class label

    return bbox_vis

def get_point_mask(pcl : np.array, bbox : dict, x_add=(0., 0.), y_add=(0., 0.), z_add=(0., 0.)):
    '''
    :param pcl: x,y,z ...
    :param bbox: x,y,z,l,w,h,yaw
    :param x_add:
    :param y_add:
    :param z_add:
    :return: Segmentation mask
    '''
    # this needs to be rewritten then.
    box_translation = bbox['translation']
    box_rotation = bbox['rotation']
    # angle = bbox[6]
    # Rot_z = np.array(([np.cos(angle), - np.sin(angle), 0],
    #                   [np.sin(angle), np.cos(angle), 0],
    #                   [0, 0, 1]))
    box_size = bbox['size']
    s = pcl.copy()
    # s[:, :3] -= bbox[:3]
    s[:, :3] -= box_translation
    s[:, :3] = (s[:, :3] @ box_rotation)[:, :3]
    size = np.array((- box_size[0]/2, box_size[0]/2, - box_size[1]/2, box_size[1]/2, - box_size[2]/2, box_size[2]/2))
    point_mask = (size[0] - x_add[0] <= s[:, 0]) & (s[:, 0] <= size[1] + x_add[1]) & (size[2] - y_add[0] <= s[:, 1])\
                 & (s[:, 1] <= size[3] + y_add[1]) & (size[4] - z_add[0] <= s[:,2]) & (s[:,2] <= size[5] + z_add[1])

    return point_mask


def extend_height_box(full_pcl, pcl_cluster, box):
    '''

    :param pcl: Full pcl
    :return:
    '''
    z_max = pcl_cluster[:,2].max()
    h_best = 0
    z_best = 0
    max_points = 0

    for h in range(1,30, 1):
        h = h/10
        z = z_max - h / 2
        box[2] = z
        box[5] = h
        mask = get_point_mask(full_pcl, box)
        contained_points = np.sum(mask)

        if max_points < contained_points:
            max_points = contained_points

            z_best = z
            h_best = h

    # if box[7] == self.config['Vehicle']['label']:
        # lenght
        # if box[3] < 3.5:
        #     box[3] = 3.5    # TODO Taken from ego dimensions, do it as param and from inten !!!

    box[2] = z_best + 0.05
    box[5] = h_best
    return box

def reorder_corners(corners):
    # calculate distnace
    dist = ((corners - corners[0]) ** 2).sum(1)
    indices = np.argsort(dist)

    # init, closest, farthrest, and last one
    order = [indices[0], indices[1], indices[3], indices[2], indices[0]]

    corners = corners[order]
    return corners

def centroid_poly(poly):
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = 0

    for m in range(n):
        sp = poly[T[m, :], :]
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    return C / np.sum(W)


def show_box_info(pcl, bounding_box, threshold):
    corners = np.array(list((bounding_box.corner_points)))

    corners = reorder_corners(corners)
    print('----------')
    print(f"Nbr of points {len(pcl)}, Approx. distance {np.sqrt(pcl[:, :3] ** 2).mean():.2f}")
    print(f"Paralelel {bounding_box.length_parallel:.2f}, orthogonal {bounding_box.length_orthogonal:.2f}")




def contain_all_points(pcl, corners):
    hull = ConvexHull(pcl[:, :2])

    hullpoints = pcl[hull.vertices, :]

    contain_points = []
    for point in hullpoints:
        a = Point(point[0], point[1])
        b = Polygon([corners[0], corners[1], corners[2], corners[3], corners[0]])
        contain_points.append(b.contains(a))

    all_points = np.all(contain_points)

    return all_points


def calculate_distance_to_box(pcl, corners):
    hull = ConvexHull(pcl[:,:2])

    hullpoints = pcl[hull.vertices, :]

    criterion_list = []

    for point in hullpoints:
        a = Point(point[0], point[1])
        b = Polygon([corners[0], corners[1], corners[2], corners[3], corners[0]])
        dist = a.distance(b.exterior)
        if point[3] > 0.95:
            dist = dist * 1.2
        criterion_list.append(dist)

    criterion = np.max(criterion_list)
    return criterion

def min_area_to_detection_box(box):
    '''

    :param full_pcl: All points for extension
    :param pcl_cluster: Cluster pcl
    :param box: Min Area bounding box with orthogonal and so on
    :param clz: Add class label
    :return:
    '''

    x, y = box.rectangle_center
    h, z = 1.8, 1.0
    l, w = box.length_parallel, box.length_orthogonal
    yaw = box.unit_vector_angle

    bbox = np.array((x, y, z, l, w, h, yaw))

    return bbox




def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou

def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    boxes3d = torch.tensor(boxes3d)
    rot_angle = limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def get_boxes_from_ego_poses(poses, ego_box):

    box_list = []

    # breakpoint()
    for pose in poses:

        translation = np.insert(ego_box['translation'], obj=3, values=1, axis=0)

        new_translation = translation @ pose.T
        new_rotation = ego_box['rotation'] @ pose[:3,:3]

        box = {'translation' : new_translation[:3],
               'rotation' : new_rotation,
               'size' : ego_box['size']}

        box_list.append(box)

    return np.stack(box_list)


def get_inst_bbox_sizes(data_sample):
    boxes_list = data_sample['box1']

    # get instances boxes observed at time moment 0 from ego pose
    inst_bbox_sizes = {}
    for t in range(len(boxes_list)):
        for inst_i in range(len(boxes_list[0])):
            if inst_i >= len(boxes_list[t]):
                continue
            uuid = boxes_list[t][inst_i]['uuid']
            lwh = torch.as_tensor(boxes_list[t][inst_i]['size'])
            if uuid not in inst_bbox_sizes:
                inst_bbox_sizes[uuid] = lwh[None]
            else:
                inst_bbox_sizes[uuid] = torch.cat([inst_bbox_sizes[uuid], lwh[None]], dim=0)

    # convert to a list of poses
    inst_bbox_sizes = [traj for uuid, traj in inst_bbox_sizes.items()]

    return inst_bbox_sizes


# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""SE3 class for point cloud rotation and translation."""
class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.
        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)
        """
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)
        self.rotation = rotation
        self.translation = translation

        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = self.rotation
        self.transform_matrix[:3, 3] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(3) transformation to this point cloud.
        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`
        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return point_cloud @ self.rotation.T + self.translation

    def inverse_transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Undo the translation and then the rotation (Inverse SE(3) transformation)."""
        return (point_cloud.copy() - self.translation) @ self.rotation

    def inverse(self) -> "SE3":
        """Return the inverse of the current SE3 transformation.
        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.
        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        return SE3(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def compose(self, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.
        Algebraic representation: chained_se3 = T * right_se3
        Args:
            right_se3: another instance of SE3 class
        Returns:
            chained_se3: new instance of SE3 class
        """
        chained_transform_matrix = self.transform_matrix @ right_se3.transform_matrix
        chained_se3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_se3

    def right_multiply_with_se3(self, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.
        Algebraic representation: chained_se3 = T * right_se3
        Args:
            right_se3: another instance of SE3 class
        Returns:
            chained_se3: new instance of SE3 class
        """
        return self.compose(right_se3)


class Box:
    def __init__(self, xyz, rpy, size):
        self.xyz = xyz
        self.rpy = rpy
        self.size = size
        self.device = xyz.device

    def get_pose(self):
        return xyz_rpy_to_matrix(torch.tensor([*self.xyz, *self.rpy], device=self.device))

    def get_verts(self):
        l, w, h = self.size
        vertices = torch.cat([torch.stack([-l / 2, -w / 2, -h / 2]),
                              torch.stack([l / 2, -w / 2, -h / 2]),
                              torch.stack([l / 2, w / 2, -h / 2]),
                              torch.stack([-l / 2, w / 2, -h / 2]),
                              torch.stack([-l / 2, -w / 2, h / 2]),
                              torch.stack([l / 2, -w / 2, h / 2]),
                              torch.stack([l / 2, w / 2, h / 2]),
                              torch.stack([-l / 2, w / 2, h / 2])], dim=0).to(self.device)
        vertices = vertices.reshape(-1, 3)
        pose = self.get_pose()
        vertices = pose[:3, :3] @ vertices.T + pose[:3, 3:4]
        vertices = vertices.T
        return vertices

    def get_volume(self):
        l, w, h = self.size
        return l * w * h

    def show(self):
        with torch.no_grad():
            verts = self.get_verts().cpu()
            lines = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0],
                                  [4, 5], [5, 6], [6, 7], [7, 4],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])

            mlab.points3d(verts[:, 0].cpu(), verts[:, 1].cpu(), verts[:, 2].cpu(), color=(0, 0, 0), scale_factor=0.1)
            draw_coord_frame(torch.eye(4), scale=1.)  # world frame
            draw_coord_frame(self.get_pose().cpu(), scale=0.5)
            for line in lines:
                mlab.plot3d(verts[line, 0], verts[line, 1], verts[line, 2], color=(0, 0, 0), tube_radius=0.01)


def get_boxes(points, ids, flow=None, ignore_index=-1):
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert ids.ndim == 1
    assert len(points) == len(ids)
    assert flow is None or flow.shape == points.shape

    # find points with the same id
    box_poses = []
    box_sizes = []
    for id in torch.unique(ids):
        if id == ignore_index:
            continue

        inst_points = points[ids == id]
        xyz = (inst_points.max(dim=0)[0] + inst_points.min(dim=0)[0]) / 2
        # create box parameters
        pose = torch.eye(4)
        pose[:3, 3] = xyz
        if flow is not None:
            inst_flows = flow[ids == id]
            flow_vector = inst_flows.mean(dim=0)
            # normalize flow vector
            flow_vector /= torch.linalg.norm(flow_vector)
            # find yaw angle
            yaw = torch.atan2(flow_vector[1], flow_vector[0])
            R = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                              [torch.sin(yaw), torch.cos(yaw), 0],
                              [0., 0., 1.]])
            pose[:3, :3] = R
        # estimate box size
        inst_points_centered = inst_points - xyz
        lhw = inst_points_centered.max(dim=0)[0] - inst_points_centered.min(dim=0)[0]
        # add to lists
        box_poses.append(pose)
        box_sizes.append(lhw)

    box_poses = torch.stack(box_poses).to(points.device)
    box_sizes = torch.stack(box_sizes).to(points.device)

    return box_poses, box_sizes
