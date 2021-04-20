""" enhance torch's implementation by providing stable argsort
"""
import torch
from .helper import get_kernel

def argsort(value, increasing=True, stable=False):
    """ sort tensor and returning sorted index

    Args:
        value (tensor): the input tensor (1-D)
        increasing (bool): whether to sort in increasing order
        stable (bool): whether performing stable sort
    Returns:
        index (tensor): the index to sort (int64)
    """
    if stable:
        assert value.dtype in [torch.float32, torch.float64, torch.int32, torch.int64], "unsupported data type for `value`!"

        index_out = torch.arange(len(value), device=value.device)
        kernel = get_kernel("stable_argsort_kernel", value.dtype, torch.int64, torch.int64)
        kernel(value.clone(), index_out, increasing) # change in place
        return index_out

    return torch.argsort(value, dim=0, descending=(not increasing))
