import os
import socket

server_name = socket.gethostname()

DATA_PATH = f"{os.path.expanduser('~')}/data/"

if server_name.startswith("Pat"):
    KITTI_SF_PATH = f"{os.path.expanduser('~')}/rci/data/kitti_sf/"

# elif server_name.startswith('g') or server_name.startswith("login"):

elif server_name.startswith("boruvka"):
    KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/sceneflow/kitti_sf/"
    TMP_VIS_PATH = f"{os.path.expanduser('~')}/pcflow/toy_samples/tmp_vis/"
    EXP_PATH = f"{os.path.expanduser('~')}/experiments/"

    # todo, create bash structure
elif server_name.startswith("login") or server_name.startswith('g'):
    # KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/"
    TMP_VIS_PATH = f"{os.path.expanduser('~')}/pcflow/toy_samples/tmp_vis/"
    EXP_PATH = f"{os.path.expanduser('~')}/experiments/"


else:
    TMP_VIS_PATH = f"{os.path.expanduser('~')}/pcflow/toy_samples/tmp_vis/"
    EXP_PATH = f"{os.path.expanduser('~')}/experiments/"
    # raise NotImplementedError(f"Server {server_name} not recognized")

