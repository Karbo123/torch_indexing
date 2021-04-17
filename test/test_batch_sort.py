import torch
from torch_indexing import batch_sort

def test():
    for device in ["cuda", "cpu"]:
        for increasing in [True, False]:
            value = torch.rand(1000).to(device=device)
            batch = torch.repeat_interleave(torch.arange(3), torch.tensor([300, 650, 50])).to(device=device)
            
            # our prediction
            index = batch_sort(value, batch, increasing)
            
            # making target
            target_index = torch.cat([torch.argsort(value[batch==i], descending=not increasing) + [0, 300, 950][i] for i in range(3)], dim=0)
            
            # asserting
            assert (index - target_index).abs().max() == 0

