""" sort 1-D tensor in batch
"""
import torch
from .helper import get_kernel
from torch_scatter import scatter

def batch_sort(value, batch, increasing=True, baseline="auto"):
    """ sort value in batch, and return the index
    
    Args:
        value (tensor): the value to sort
        batch (tensor): the batch index in increasing order
        increasing (bool): whether to sort in increasing order
        baseline (bool or "auto"): baseline is to use pure pytorch implementation
    Returns:
        index (tensor): the index to sort (int64)
    """
    assert value.dtype in [torch.float32, torch.float64, torch.int32, torch.int64], "unsupported data type for `value`!"
    assert batch.dtype in [torch.int64], "unsupported data type for `batch`!"
    assert value.device == batch.device, "`value` and `batch` must be on the same device!"
    if baseline == "auto": baseline = False # always use our faster implementation
    if baseline: return batch_sort_baseline(value, batch, increasing)

    index_out = torch.arange(len(batch), device=batch.device)
    kernel = get_kernel("batch_sort_kernel", value.dtype, batch.dtype, torch.int64)
    kernel(value.clone(), batch.clone(), index_out, increasing) # change in place
    return index_out


def batch_sort_baseline(value, batch, increasing=True):
    device = batch.device
    with torch.no_grad():
        num          = scatter(torch.ones(len(value), device=device, dtype=torch.int64), batch, reduce="sum")
        val_min      = torch.repeat_interleave(scatter(value, batch, reduce="min"),  num)
        val_max      = torch.repeat_interleave(scatter(value, batch, reduce="max"),  num)
        val_offset   = torch.repeat_interleave(torch.arange(len(num), device=device),  num)
        val_offset   = val_offset if increasing else ((len(num) - 1) - val_offset)
        offset_value = (value - val_min) / (val_max - val_min) + (val_offset * 2) # *2 for enlarging the margins
        index_out    = torch.argsort(offset_value, dim=0, descending=(not increasing))

    return index_out
