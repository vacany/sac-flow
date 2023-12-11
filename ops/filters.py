from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch

__all__ = [
    'filter_range',
    'within_bounds',
    'filter_box',
    'filter_grid',
]

default_rng = np.random.default_rng(135)


def filter_grid(cloud, grid_res, only_mask=False, keep='random', preserve_order=False, log=False, rng=default_rng):
    """Keep single point within each cell. Order is not preserved."""
    assert isinstance(cloud, (np.ndarray, torch.Tensor))
    assert isinstance(grid_res, float) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    # Convert to numpy array with positions.
    if isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            x = structured_to_unstructured(cloud[['x', 'y', 'z']])
        else:
            x = cloud
    elif isinstance(cloud, torch.Tensor):
        x = cloud.detach().cpu().numpy()

    # Create voxel indices.
    keys = np.floor(x / grid_res).astype(int).tolist()

    # Last key will be kept, shuffle if needed.
    # Create index array for tracking the input points.
    ind = list(range(len(keys)))
    if keep == 'first':
        # Make the first item last.
        keys = keys[::-1]
        ind = ind[::-1]
    elif keep == 'random':
        # Make the last item random.
        rng.shuffle(ind)
        # keys = keys[ind]
        keys = [keys[i] for i in ind]
    elif keep == 'last':
        # Keep the last item last.
        pass

    # Convert to immutable keys (tuples).
    keys = [tuple(i) for i in keys]

    # Dict keeps the last value for each key (already reshuffled).
    key_to_ind = dict(zip(keys, ind))
    if preserve_order:
        ind = sorted(key_to_ind.values())
    else:
        ind = list(key_to_ind.values())

    if log:
        # print('%.3f = %i / %i points kept (grid res. %.3f m).'
        #       % (mask.double().mean(), mask.sum(), mask.numel(), grid_res))
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    # TODO: Convert to boolean mask?
    if only_mask:
        # return mask
        return ind

    filtered = cloud[ind]
    return filtered


def within_bounds(x, min=None, max=None, bounds=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if bounds:
        assert min is None and max is None
        min, max = bounds

    if min is not None and min > -float('inf'):
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None and max < float('inf'):
        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        keep = keep & (x.flatten() <= max)

    if log_variable is not None:
        print('%.3f = %i / %i points kept (%.3g <= %s <= %.3g).'
              % (keep.double().mean(), keep.sum(), keep.numel(),
                 min if min is not None else float('nan'),
                 log_variable,
                 max if max is not None else float('nan')))

    return keep


def filter_range(cloud, min=None, max=None, only_mask=False, log=False):
    """Keep points with depth in bounds."""
    assert isinstance(cloud, torch.Tensor)

    x = torch.as_tensor(cloud)
    vp = torch.zeros((1, 3), dtype=x.dtype, device=x.device)
    depth = torch.linalg.norm(x - vp, dim=1, keepdim=True)

    keep = within_bounds(depth, min=min, max=max, log_variable='depth' if log else None)
    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def filter_box(cloud, box_size, box_T=None, only_mask=False):
    """Keep points with rectangular bounds."""
    assert isinstance(cloud, np.ndarray) or isinstance(cloud, torch.Tensor)

    if isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            pts = structured_to_unstructured(cloud[['x', 'y', 'z']])
        else:
            pts = cloud
        assert pts.ndim == 2, "Input points tensor dimensions is %i (only 2 is supported)" % pts.ndim
        pts = torch.from_numpy(pts)
    else:
        pts = cloud.clone()

    if box_T is None:
        box_T = torch.eye(4)
    assert isinstance(box_T, torch.Tensor)
    assert box_T.shape == (4, 4)
    box_center = box_T[:3, 3]
    box_orient = box_T[:3, :3]

    pts = (pts - box_center) @ box_orient

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    keep_x = within_bounds(x, min=-box_size[0] / 2, max=+box_size[0] / 2)
    keep_y = within_bounds(y, min=-box_size[1] / 2, max=+box_size[1] / 2)
    keep_z = within_bounds(z, min=-box_size[2] / 2, max=+box_size[2] / 2)

    keep = torch.logical_and(keep_x, keep_y)
    keep = torch.logical_and(keep, keep_z)

    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered
