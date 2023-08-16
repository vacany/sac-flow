from .kittisf import *
from data.PATHS import DATA_PATH

data_path = f'{DATA_PATH}/sceneflow/kittisf/'

lidar_pose = (0,0,0)

all_files = sorted(glob.glob(data_path + 'all_data_format/*.npz'))

len_dataset = len(all_files)

train_idx = [0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33,
                     34, 35, 36, 39, 40, 42, 43, 44, 45, 47, 49, 51, 53, 55, 56, 60, 62, 64, 65, 66, 67, 68, 69, 70,
                     73, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 89, 93, 94, 95, 96, 97, 98, 101, 105, 108, 109, 110,
                     111, 112, 113, 114, 115, 117, 118, 122, 123, 124, 126, 128, 130, 131, 133, 135, 137, 138, 140]

# test_idx = [i for i in range(len_dataset) if i not in train_idx]
# THIS IS SCOOP SPLIT Kitti_T
test_idx = [3, 7, 11, 19, 25, 26, 34, 37, 42, 43, 46, 51, 53, 55, 57, 59, 62, 63, 64, 66,
            68, 76, 77, 79, 80, 85, 94, 95, 97, 98, 105, 112, 113, 115, 116, 117, 119, 120,
            129, 132, 141, 142, 146, 148, 150, 158, 160, 162, 168, 199]
# test_idx = [162]
# kittit


