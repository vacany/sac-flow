import torch

data_flow = {'pc1' : torch.rand(1, 100, 3), 'pc2' : torch.rand(1, 100, 3), 'gt_flow' : torch.rand(1, 100, 3)}


def decor(func):
    input_args = func.__code__.co_varnames[:func.__code__.co_argcount]
    def wrapper(batch):
        new_batch = {batch[key] for key in input_args}
        output = func(**new_batch)
        return output
    return wrapper


def add_flow(gt_flow, add=1.):
    new_gt_flow = gt_flow + add

    return new_gt_flow


function_blocks = []

add_flow = decor(add_flow)
out = add_flow(data_flow)
