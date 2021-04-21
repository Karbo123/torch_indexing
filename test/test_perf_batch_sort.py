""" test performance of `batch_sort` function
"""

import time
import torch
from torch_indexing import batch_sort

def test():
    for device in ["cuda", "cpu"]:
        num = torch.randint(low=5, high=10000, size=(100, ))
        batch = torch.repeat_interleave(torch.arange(100), num).to(device=device)
        value = torch.rand(num.sum()).to(device=device)

        N_REPEAT = 100
        time_list_ours, time_list_torch = list(), list()
        for i in range(N_REPEAT):
            t0 = time.time()
            batch_sort(value, batch, baseline=True)
            torch.cuda.synchronize()
            t1 = time.time()
            batch_sort(value, batch, baseline=False)
            torch.cuda.synchronize()
            t2 = time.time()
            time_list_ours.append(t2 - t1)
            time_list_torch.append(t1 - t0)
            time_ours = sum(time_list_ours) / len(time_list_ours)
            time_torch = sum(time_list_torch) / len(time_list_torch)
            print(f"{device}|{i} ==> Ours={time_ours:.4f} | Torch={time_torch:.4f}")
        
        print(f"device={device} | repeat={N_REPEAT} | Ours={time_ours:.4f} | Torch={time_torch:.4f}")


if __name__ == "__main__":
    test()

