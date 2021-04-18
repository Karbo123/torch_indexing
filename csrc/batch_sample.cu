/*
    sample in batch
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////


torch::Tensor batch_sample(torch::Tensor batch, torch::Tensor num, torch::Tensor rand_num, torch::Tensor start_end_ind)
{

    // sort
    auto first = thrust::make_zip_iterator(thrust::make_tuple(segments.begin(), key_vec.begin()));
    thrust::stable_sort_by_key(value_ptr, value_ptr + one_batch_len * batch_num, first, thrust::greater<FloatType>());

    // NOTE: thrust of old version may fail to execute this command (and compile with warnings)
    thrust::stable_sort_by_key(segments.begin(), segments.begin() + one_batch_len * batch_num, key_vec.begin(), thrust::less<int>());
}



