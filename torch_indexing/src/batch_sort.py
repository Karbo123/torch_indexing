""" sort 1-D tensor in batch
"""
import torch
from .helper import get_kernel

def batch_sort(value, batch, increasing=True, dtype_out=torch.int64):
    """ sort value in batch, and return the index
    
    Args:
        value (tensor): the value to sort
        batch (tensor): the batch index in increasing order
        increasing (bool): whether to sort in increasing order
        dtype_out (dtype): the desired data type of output index
    Returns:
        index (tensor): the index to sort
    """
    assert value.dtype in [torch.float32, torch.float64, torch.int32, torch.int64], "unsupported data type for `value`!"
    assert batch.dtype in [torch.int32, torch.int64], "unsupported data type for `batch`!"
    assert dtype_out   in [torch.int32, torch.int64], "unsupported data type for output index!"
    assert value.device == batch.device, "`value` and `batch` must be on the same device!"
    
    index_out = torch.arange(len(batch), device=batch.device, dtype=dtype_out)
    kernel = get_kernel("batch_sort_kernel", value.dtype, batch.dtype, dtype_out)
    kernel(value.clone(), batch.clone(), index_out, increasing) # change in place
    return index_out

