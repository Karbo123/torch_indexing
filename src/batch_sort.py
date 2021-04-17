""" sort 1-D tensor in batch
"""
import torch
from libtorch_indexing import batch_sort_kernel


def batch_sort(value, batch, increasing=True):
    """ sort value in batch, and return the index
    
    Args:
        value (tensor): the value to sort
        batch (tensor): the batch index in increasing order
        increasing (bool): whether to sort in increasing order
    Returns:
        index (tensor): the index to sort
    """
    index_out = torch.arange(len(batch), device=batch.device)
    batch_sort_kernel(value.clone(), batch.clone(), index_out, increasing) # change in place
    return index_out



if __name__ == "__main__": # CUDA_VISIBLE_DEVICES=6 python batch_sort.py
    value = torch.rand(100).cuda()
    batch = torch.repeat_interleave(torch.arange(3), torch.tensor([30, 65, 5])).cuda()
    print(f"value = {value}")
    print(f"batch = {batch}")
    index = batch_sort(value, batch, increasing=True)
    print(f"index = {index}")
    print(f"sorted value = {value[index]}")

    import ipdb; ipdb.set_trace()