import numpy as np
import glob

from kiss_icp.pipeline import OdometryPipeline
import os

class Dataset():
    def __init__(self, data_dir, scan_list):
        self.data_dir = data_dir
        self.scans = scan_list

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.read_point_cloud(idx), idx

    def read_point_cloud(self, idx):
        pts = self.scans[idx]

        return pts   # it takes only x,y,z ... no another feature

def transform_pc(pts, pose):

    '''

    :param pts: point cloud
    :param pose: 4x4 transformation matrix
    :return:
    '''
    transformed_pts = np.insert(pts.copy(), 3, 1, axis=1)
    transformed_pts[:, 3] = 1
    transformed_pts[:, :3] = (transformed_pts[:, :4] @ pose.T)[:, :3]

    return transformed_pts

def apply_kiss_icp(scan_list):

    if len(scan_list) < 10:
        print('Too few scans for KISS-ICP, it requires more based on issues.')
    pose_list = []
    # Should be float
    for idx, scan in enumerate(scan_list):
        scan_list[idx] = scan.astype('float')

    dataset = Dataset('../../dev', scan_list=scan_list)
    # breakpoint()
    config_file = os.path.expanduser("~") + '/pcflow/models/kiss_icp/test_icp.yaml'

    kiss_model = OdometryPipeline(dataset=dataset, config=config_file)

    kiss_model.run()

    pose_list.append(np.stack(kiss_model.poses))

    pose = pose_list[0][1]

    # for idx, pose in enumerate(pose_list[0]):
    #     print(f'frame: {idx} \n', pose)

    global_pc_list = []

    for idx, scan in enumerate(scan_list):
        gl_pc2 = transform_pc(scan, pose)
        global_pc_list.append(np.insert(gl_pc2, 3, idx, axis=1))

    return global_pc_list, pose_list

