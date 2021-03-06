import torch
from torch_indexing import batch_sort

def test():
    for device in ["cuda", "cpu"]:
        for increasing in [True, False]:
            for value_dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                value = (torch.rand(1000).to(device=device) * 100).to(dtype=value_dtype)
                batch = torch.repeat_interleave(torch.arange(3), torch.tensor([300, 650, 50])).to(device=device)
                
                # our prediction
                index = batch_sort(value, batch, increasing)
                sorted_value = value[index]

                # making target
                target_index = torch.cat([torch.argsort(value[batch==i], descending=(not increasing)) + [0, 300, 950][i] for i in range(3)], dim=0)
                target_sorted_value = value[target_index]

                # asserting
                assert (sorted_value - target_sorted_value).abs().max() == 0


if __name__ == "__main__":
    test()
