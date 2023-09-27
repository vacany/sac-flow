# argoverse
import numpy as np
import glob
import os

paths = sorted(glob.glob(f'{os.path.expanduser("~")}/data/sceneflow/argoverse/val/f9*/*.npz'))
pc_list = []
for i in range(len(paths)):
    data = np.load(paths[i], allow_pickle=True)
    pc_list.append(data['pc1'].astype('float'))
    print(i)
    if len(pc_list) == 3: break

# pc = np.concatenate(pc_list)

# visualize_points3D(pc, pc[:,3], lookat=[0,0,0])

from models.kiss_icp.kissicp import apply_kiss_icp

apply_kiss_icp(pc_list)
