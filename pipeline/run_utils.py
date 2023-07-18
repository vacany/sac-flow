import os
import subprocess
from data.gpu_utils import get_free_gpu_indices


def join_args(cfg):

    str_args = []
    for arg, value in cfg.items():
        if value is not None:
            # added_arg = f' --{arg} {value}'
            # str_args.append(added_arg)
            str_args.append(f'--{arg}')
            str_args.append(f'{value}')

    return str_args

def run_experiment(cfg, DETACH=False):
    #

    # maybe basic smooth, + forward so flow is same and then normals?
    str_args = join_args(cfg)

    # Paralelize
    # if gpu not available, wait

    # Udelat convinient, s tim si clovek i ujasni, co chce. Delat ten process nazacatku v klidu dohromady
    available_gpus = get_free_gpu_indices()

    str_args.append(f'--gpu')
    str_args.append(f'{available_gpus[0]}')
    # str_args.append(f' &')

    os.chdir(os.path.expanduser('~') + '/pcflow')

    # works!
    if DETACH:
        subprocess.Popen(["nohup", "python", "pipeline/sceneflow.py"] + str_args,
                         stdout=open('/dev/null', 'w'),
                         stderr=open('logfile.log', 'a'),
                         preexec_fn=os.setpgrp)
    else:
        os.system(f'python pipeline/sceneflow.py {" ".join(str_args)}')
