import os
import socket

server_name = socket.gethostname()

DATA_PATH = f"{os.path.expanduser('~')}/data/"

if server_name.startswith("Pat"):
    KITTI_SF_PATH = f"{os.path.expanduser('~')}/rci/data/kitti_sf/"


elif server_name.startswith('g') or server_name.startswith("login"):
    KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/kitti_sf/"

elif server_name.startswith("boruvka"):
    KITTI_SF_PATH = f"{os.path.expanduser('~')}/data/kitti_sf/"





else:
    raise NotImplementedError(f"Server {server_name} not recognized")

