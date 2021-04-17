
#include <torch/extension.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////

template <typename FloatType, typename IndexType, typename SizeType, bool Increasing, bool CUDA>
void batchSort_kernel(FloatType *value, IndexType *batch, IndexType *index_out, SizeType length)
{   
    if constexpr (CUDA)
    {
        thrust::device_ptr<FloatType> value_ptr = thrust::device_pointer_cast(value);
        thrust::device_ptr<IndexType> batch_ptr = thrust::device_pointer_cast(batch);
        thrust::device_ptr<IndexType> index_ptr = thrust::device_pointer_cast(index_out);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(batch_ptr, index_ptr));
        if constexpr (Increasing) thrust::stable_sort_by_key(thrust::device, value_ptr, value_ptr + length, first, thrust::less<FloatType>());
        else thrust::stable_sort_by_key(thrust::device,value_ptr, value_ptr + length, first, thrust::greater<FloatType>());
        thrust::stable_sort_by_key(thrust::device, batch_ptr, batch_ptr + length, index_ptr, thrust::less<IndexType>());
    } else
    {
        FloatType *value_ptr = value;
        IndexType *batch_ptr = batch;
        IndexType *index_ptr = index_out;

        auto first = thrust::make_zip_iterator(thrust::make_tuple(batch_ptr, index_ptr));
        if constexpr (Increasing) thrust::stable_sort_by_key(thrust::host, value_ptr, value_ptr + length, first, thrust::less<FloatType>());
        else thrust::stable_sort_by_key(thrust::host, value_ptr, value_ptr + length, first, thrust::greater<FloatType>());
        thrust::stable_sort_by_key(thrust::host, batch_ptr, batch_ptr + length, index_ptr, thrust::less<IndexType>());
    }
}


template <typename FloatType, typename IndexType, typename SizeType>
void batchSort(torch::Tensor value, torch::Tensor batch, torch::Tensor index_out, bool increasing)
{
    FloatType* value_ptr     = value.data_ptr<FloatType>();
    IndexType* batch_ptr     = batch.data_ptr<IndexType>();
    IndexType* index_out_ptr = index_out.data_ptr<IndexType>();
    SizeType   length        = value.size(0);

    if (value.is_cuda())
    {
        if (increasing) batchSort_kernel<FloatType, IndexType, SizeType, true, true>(value_ptr, batch_ptr, index_out_ptr, length);
        else batchSort_kernel<FloatType, IndexType, SizeType, false, true>(value_ptr, batch_ptr, index_out_ptr, length);
    } else
    {
        if (increasing) batchSort_kernel<FloatType, IndexType, SizeType, true, false>(value_ptr, batch_ptr, index_out_ptr, length);
        else batchSort_kernel<FloatType, IndexType, SizeType, false, false>(value_ptr, batch_ptr, index_out_ptr, length);
    }
}


/////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batch_sort_kernel", &batchSort<float, int64_t, int64_t>, "sort in batch (CUDA)",
                                py::arg("value"),
                                py::arg("batch"),
                                py::arg("index_out"),
                                py::arg("increasing")
        );
}


