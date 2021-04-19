import torch
from .batch_sort import batch_sort

def batch_sample(batch, num, baseline=False):
    """ sample indices in batch (not repeat)

    Args:
        batch (tensor): the batch index in increasing order
        num (tensor): the num of samples for each batch
        baseline (bool): baseline is to use pure pytorch implementation
    Returns:
        index (tensor): the sampled indices (int64)
    """
    assert batch[-1] + 1 == len(num), "num of batch does not match!"
    assert batch.dtype in [torch.int64], "unsupported data type for `batch`!"
    assert batch.device == num.device, "`batch` and `num` must be on the same device!"

    device = batch.device
    value = torch.rand(batch.shape, device=device)
    start_ind = torch.repeat_interleave(torch.searchsorted(batch, torch.arange(len(num), device=device)), num)
    ind = torch.arange(len(start_ind), device=device) - \
            torch.repeat_interleave(torch.cat([torch.tensor([0], device=device), 
                                               torch.cumsum(num, dim=0)[:-1]], dim=0), num)
    index_out = batch_sort(value, batch, baseline=baseline)[start_ind + ind]
    return index_out

