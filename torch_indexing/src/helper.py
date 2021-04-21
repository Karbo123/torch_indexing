import torch
from .libtorch_indexing import *

def get_kernel(base_name, *data_types):
    """ get the kernel function to call """
    convert_dict = {
        torch.float32 : "f4",
        torch.float64 : "f8",
        torch.int32   : "i4",
        torch.int64   : "i8",
    }
    name = base_name
    for i, dt in enumerate(data_types):
        if i == 0: name += "_["
        name += convert_dict[dt] + ","
        if i == len(data_types) - 1: name = name[:-1] + "]"
    
    fn = globals()[name]
    return fn

