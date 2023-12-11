import os
import yaml
import socket

server_name = socket.gethostname()

# Set up paths where you want to store data, visualize data, and store experiments
if "BASE_PATH" in os.environ:
    pass
else:
    BASE_PATH = f"/mnt/personal/vacekpa2/" # <----- change this to your f

DATA_PATH = f"{BASE_PATH}/data"
VIS_PATH = f"{BASE_PATH}/visuals/"
EXP_PATH = f"{BASE_PATH}/experiments/"


# DATA_PATH = os.path.normpath(DATA_PATH)
# VIS_PATH = os.path.normpath(VIS_PATH)
# EXP_PATH = os.path.normpath(EXP_PATH)

for path in [DATA_PATH, VIS_PATH, EXP_PATH]:
    os.makedirs(path, exist_ok=True)
