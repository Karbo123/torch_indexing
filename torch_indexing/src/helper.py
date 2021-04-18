import torch
from .libtorch_indexing import *

def get_kernel(base_name, *data_types):
    """ get the kernel function to call """
    convert_dict = {
        torch.float32 : "float",
        torch.float64 : "double",
        torch.int32   : "int",
        torch.int64   : "long_int",
    }
    name = base_name
    for i, dt in enumerate(data_types):
        if i == 0: name += "_["
        name += convert_dict[dt] + ","
        if i == len(data_types) - 1: name = name[:-1] + "]"
    
    fn = globals()[name]
    return fn

