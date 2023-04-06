
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

if __name__ == "__main__":
    pose_list = []
    dataset = Dataset('.', scan_list=[pc1.astype('float'), pc2.astype('float')])
    kiss_model = OdometryPipeline(dataset=dataset, config='/home/patrik/SCOOP/scripts/test_icp.yaml')
    kiss_model.run()

    pose_list.append(np.stack(kiss_model.poses))

    pose = pose_list[0][1]
