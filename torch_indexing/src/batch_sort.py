""" sort 1-D tensor in batch
"""
import torch
from .libtorch_indexing import batch_sort_kernel


def batch_sort(value, batch, increasing=True):
    """ sort value in batch, and return the index
    
    Args:
        value (tensor): the value to sort
        batch (tensor): the batch index in increasing order
        increasing (bool): whether to sort in increasing order
    Returns:
        index (tensor): the index to sort
    """
    assert value.device == batch.device, "`value` and `batch` must be on the same device!"
    index_out = torch.arange(len(batch), device=batch.device)
    batch_sort_kernel(value.clone(), batch.clone(), index_out, increasing) # change in place
    return index_out

