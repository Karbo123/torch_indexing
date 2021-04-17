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

#define TOTAL_THREADS_DENSE 512

inline int opt_n_threads(int work_size)
{
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS_DENSE), 1);
}

#define CUDA_CHECK_ERRORS()                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (cudaSuccess != err)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",                         \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, __FILE__);             \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                                                        \
    do                                                                                             \
    {                                                                                              \
        TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor");                         \
    } while (0)

#define CHECK_IS_FLOAT(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor");       \
    } while (0)

#define CHECK_CUDA(x)                                                                              \
    do                                                                                             \
    {                                                                                              \
        TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                                     \
    } while (0)

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif


#define MAX2(x, y) ((x)>(y) ? (x): (y))
#define MAX3(x, y, z) (MAX2(MAX2(x,y), z))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////





torch::Tensor batch_sample(torch::Tensor batch, torch::Tensor num, torch::Tensor rand_num, torch::Tensor start_end_ind)
{

    // sort
    auto first = thrust::make_zip_iterator(thrust::make_tuple(segments.begin(), key_vec.begin()));
    thrust::stable_sort_by_key(value_ptr, value_ptr + one_batch_len * batch_num, first, thrust::greater<FloatType>());

    // NOTE: thrust of old version may fail to execute this command (and compile with warnings)
    thrust::stable_sort_by_key(segments.begin(), segments.begin() + one_batch_len * batch_num, key_vec.begin(), thrust::less<int>());
}