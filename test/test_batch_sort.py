import torch
from torch_indexing import batch_sort

def test():
    for device in ["cuda", "cpu"]:
        for increasing in [True, False]:
            for value_dtype in [torch.float32, torch.float64]:
                for batch_dtype in [torch.int64]:
                    for out_dtype in [torch.int64]:
                        value = (torch.rand(1000).to(device=device) * 100).to(dtype=value_dtype)
                        batch = torch.repeat_interleave(torch.arange(3), torch.tensor([300, 650, 50])).to(device=device).to(dtype=batch_dtype)
                        
                        # our prediction
                        index = batch_sort(value, batch, increasing, dtype_out=out_dtype)
                        
                        # making target
                        target_index = torch.cat([torch.argsort(value[batch==i], descending=not increasing) + [0, 300, 950][i] for i in range(3)], dim=0)
                        target_index = target_index.to(dtype=out_dtype)

                        # asserting
                        assert (index - target_index).abs().max() == 0
