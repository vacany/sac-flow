
import numpy as np
import glob

from kiss_icp.pipeline import OdometryPipeline

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


if __name__ == "__main__":
    pose_list = []

    data = np.load('./000162_res.npz')
    pc1 = data['pc1'][:, [0, 2, 1]]
    pc2 = data['pc2'][:, [0, 2, 1]]

    # Should be float
    scan_list = [pc1.astype('float'), pc2.astype('float')]

    dataset = Dataset('../../dev', scan_list=scan_list)
    kiss_model = OdometryPipeline(dataset=dataset, config='./test_icp.yaml')
    kiss_model.run()

    pose_list.append(np.stack(kiss_model.poses))

    pose = pose_list[0][1]

    for idx, pose in enumerate(pose_list[0]):
        print(f'frame: {idx} \n', pose)



    gl_pc2 = transform_pc(pc2, pose)

    print('visualize in your interface to see result')

