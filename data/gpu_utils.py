import socket

import torch
import os
import time


import subprocess

# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/11
def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]

# for RCI only


def get_device_idx_for_port():  # TODO not for multiple gpus
    gpu_txt = open('/home/vacekpa2/gpu.txt', 'r').readlines()
    os.system('nvidia-smi -L > /home/vacekpa2/gpu_all.txt')

    time.sleep(0.1)
    gpu_all_txt = open('/home/vacekpa2/gpu_all.txt', 'r').readlines()

    gpu_all_txt = [text[7:] for text in gpu_all_txt]
    device_idx = 0
    for idx, gpu_id in enumerate(gpu_all_txt):
        if gpu_txt[0][7:] == gpu_id:
            device_idx = idx

    return device_idx

def get_device(device=None):

    if torch.cuda.is_available():

        # RCI
        if socket.gethostbyname(socket.gethostname()).startswith('g'):
            device_idx = get_device_idx_for_port()
            device = torch.device(device_idx)
        # CMP
        else:

            if device is None:

                num_of_gpus = torch.cuda.device_count()
                print("This should not be used! All devices will be filled with memory")
                free_mb_list = []
                for idx, gpu in enumerate(range(num_of_gpus)):
                    device = torch.device(f'cuda:{gpu}')
                    free_mb = torch.cuda.mem_get_info(device=device)[0]
                    free_mb_list.append(free_mb)

                free_idx = torch.tensor(free_mb_list).argmax().item()
                device = free_idx

            else:
                device = torch.device(device)

    else:
        device = torch.device('cpu')

    return device

def print_gpu_memory(device=None):
    if device is None:
        device = get_device()

    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info(device=device)[0] / 1024 / 1024
        max_memory = torch.cuda.mem_get_info(device=device)[1] / 1024 / 1024
        memory_consumed = max_memory - free_memory
        print(f"Memory consumption: {memory_consumed:.0f} MB")

def wait_for_gpu(time_to_wait=20):
    print(get_free_gpu_indices())
    while len(get_free_gpu_indices()) == 0:
        print('waiting for gpu')
        time.sleep(time_to_wait)

    return get_free_gpu_indices()[0]
