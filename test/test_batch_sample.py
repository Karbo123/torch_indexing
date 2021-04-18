import torch
from torch_indexing import batch_sample

def test():
    batch = torch.repeat_interleave(torch.arange(5), torch.tensor([100, 200, 5, 30, 1000]))
    num = torch.tensor([100, 150, 5, 1, 700])
    index = batch_sample(batch, num)

    assert index[0  :100].min() >= 0   and index[0  :100].max() < 100
    assert index[100:250].min() >= 100 and index[100:250].max() < 300
    assert index[250:255].min() >= 300 and index[250:255].max() < 305
    assert index[255:256].min() >= 305 and index[255:256].max() < 335
    assert index[256:956].min() >= 335 and index[256:956].max() < 1335

