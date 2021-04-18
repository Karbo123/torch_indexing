# Auxiliary library for advanced pytorch tensor indexing

All its functions are implemented on both CPU and CUDA with parallel computation.


# Functions

Currently implemented functions:
- **batch sort**: sort in batch (1-D tensor)
- **batch sample**: randomly sample in batch without repeat (1-D tensor)

# Build

My development environment:
- pytorch: 1.8.0+cu111 (installed by pip)
- cuda: 11.2
- nvcc: 11.2
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
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`

# build
make -j8

# install as conda package (NOTE: after installation, please do not delete the folder)
make install
```

# Examples

Please check our test scripts for details.

