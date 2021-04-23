# Auxiliary library for advanced pytorch tensor indexing

All its functions are implemented on both CPU and CUDA with parallel computation.


# Functions

Currently implemented functions:
- **batch sort**: sort in batch with stablity (1-D tensor)
- **batch sample**: randomly sample in batch without repeat (1-D tensor)
- **stable argsort**: argsort with stablity (1-D tensor)

# Benchmarking

1. batch sort
   - CUDA: Ours (**0.0029** sec.) < Baseline (0.0891 sec.)
   - CPU: Ours (**0.0374** sec.) < Baseline (0.0915 sec.)


# Build

My development environment:
- pytorch: 1.8.0+cu101 (installed by pip)
- gcc: 7.3.0
- nvcc: 10.2.89
- thrust: 1.12.0
- cub: 1.12.0
- tbb: 2020.3
- cmake: 3.18.4


Compiling example:
```bash
# create folder
mkdir build && cd build

# configure
cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DThrust_DIR="`pwd`/../../thrust-1.12.0/thrust/cmake" \
-DCUB_DIR="`pwd`/../../cub-1.12.0/cub/cmake" \
-DTBB_LIBRARY="$CONDA_PREFIX/lib/libtbb.so" \
-DTBB_INCLUDE_DIR="$CONDA_PREFIX/include" \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
-DCMAKE_CUDA_ARCHITECTURES=75 \
-DCUDA_TOOLKIT_ROOT_DIR=$CU102_CUDA_TOOLKIT_DIR


# build
make -j8

# install as conda package (NOTE: after installation, please do not delete the folder)
make install
```

# Examples

1. batch argsort
```python
import torch
from torch_indexing import batch_sort
value = torch.tensor([0.5, 0.7, 0.6, 0.9, 0.8], device="cuda")
batch = torch.tensor([0, 0, 0, 1, 1], device="cuda")
print(batch_sort(value, batch, increasing=True)) 
# printing:
# tensor([0, 2, 1, 4, 3], device='cuda:0')
#         ^  ^  ^         batch==0
#                  ^  ^   batch==1
```

2. batch sample
```python
import torch
from torch_indexing import batch_sample
batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2], device="cuda")
# randomly sample 3 items from batch==0
# randomly sample 2 items from batch==1
# randomly sample 2 items from batch==2
num   = torch.tensor([3, 2, 2], device="cuda")
print(batch_sample(batch, num))
# printing:
# tensor([1, 0, 3, 6, 5, 9, 8], device='cuda:0')
#         ^  ^  ^               batch==0 (3 items)
#                  ^  ^         batch==1 (2 items)
#                        ^  ^   batch==2 (2 items)
```

3. stable argsort
```python
import torch
from torch_indexing import argsort
value = torch.tensor([0.5, 0.4, 0.4, 0.4, 0.7, 0.1], device="cuda")
print(argsort(value, increasing=True, stable=True))
# printing:
# tensor([5, 1, 2, 3, 0, 4], device='cuda:0')
#            ^  ^  ^         preserving order
```


# TODO list

- [ ] More functions to be implemented (e.g. CUDA-KDTree)

