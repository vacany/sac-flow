import socket

import torch
import os
import time

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
