import torch
from torch_indexing import argsort

def test():
    for device in ["cuda", "cpu"]:
        for increasing in [True, False]:
            for value_dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                value = (torch.rand(1000).to(device=device) * 100).to(dtype=value_dtype)
                
                # our prediction
                index = argsort(value, increasing, stable=True)
                sorted_value = value[index]

                # making target
                target_index = argsort(value, increasing, stable=False)
                target_sorted_value = value[target_index]

                # asserting
                assert (sorted_value - target_sorted_value).abs().max() == 0


if __name__ == "__main__":
    test()
