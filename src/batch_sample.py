import torch

@torch.jit.script
def batch_sample(batch, num):
    """ sample indices in batch

    Args:
        batch (tensor): the batch index in ascending order, must be 1-D tensor
        num (tensor): the num of samples for each batch, must be 1-D tensor
    Returns:
        ind (tensor): the sampled indices (1-D tensor)
    """
    device = batch.device
    rand_num = torch.rand_like(batch)
    start_end_ind = torch.cat([torch.tensor([0], device=device), 
                               torch.searchsorted(batch, torch.arange(batch[-1] + 1, device=device), right=True)], dim=0)
    




if __name__ == "__main__":

    batch = torch.tensor([0, 0, 0, 
                          1, 1, 1, 1, 1, 
                          2, 2, 3, 2,
                          3, 3, 3])
    num = torch.tensor([3, 2, 2, 1])
    ind = batch_sample(batch, num)
    
    print(f"batch = {batch}")
    print(f"num   = {num}")
    print(f"ind   = {ind}")

