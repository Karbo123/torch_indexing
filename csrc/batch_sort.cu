
#include <torch/extension.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

#include <helper.h>
#include <boost/mp11.hpp>
#include <pybind11/pybind11.h>

namespace mp11 = boost::mp11;
namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////

template <typename ValueType, typename IndexType, typename SizeType, bool Increasing, bool CUDA>
void batchSort_kernel(ValueType *value, IndexType *batch, IndexType *index_out, SizeType length)
{   
    if constexpr (CUDA)
    {
        thrust::device_ptr<ValueType> value_ptr = thrust::device_pointer_cast(value);
        thrust::device_ptr<IndexType> batch_ptr = thrust::device_pointer_cast(batch);
        thrust::device_ptr<IndexType> index_ptr = thrust::device_pointer_cast(index_out);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(batch_ptr, index_ptr));
        if constexpr (Increasing) thrust::stable_sort_by_key(thrust::device, value_ptr, value_ptr + length, first, thrust::less<ValueType>());
        else thrust::stable_sort_by_key(thrust::device,value_ptr, value_ptr + length, first, thrust::greater<ValueType>());
        thrust::stable_sort_by_key(thrust::device, batch_ptr, batch_ptr + length, index_ptr, thrust::less<IndexType>());
    } else
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(batch, index_out));
        if constexpr (Increasing) thrust::stable_sort_by_key(thrust::host, value, value + length, first, thrust::less<ValueType>());
        else thrust::stable_sort_by_key(thrust::host, value, value + length, first, thrust::greater<ValueType>());
        thrust::stable_sort_by_key(thrust::host, batch, batch + length, index_out, thrust::less<IndexType>());
    }
}


template <typename ValueType, typename IndexType, typename SizeType>
void batchSort(torch::Tensor value, torch::Tensor batch, torch::Tensor index_out, bool increasing)
{
    ValueType* value_ptr     = value.data_ptr<ValueType>();
    IndexType* batch_ptr     = batch.data_ptr<IndexType>();
    IndexType* index_out_ptr = index_out.data_ptr<IndexType>();
    SizeType   length        = value.size(0);

    if (value.is_cuda())
    {
        if (increasing) batchSort_kernel<ValueType, IndexType, SizeType, true, true>(value_ptr, batch_ptr, index_out_ptr, length);
        else batchSort_kernel<ValueType, IndexType, SizeType, false, true>(value_ptr, batch_ptr, index_out_ptr, length);
    } else
    {
        if (increasing) batchSort_kernel<ValueType, IndexType, SizeType, true, false>(value_ptr, batch_ptr, index_out_ptr, length);
        else batchSort_kernel<ValueType, IndexType, SizeType, false, false>(value_ptr, batch_ptr, index_out_ptr, length);
    }
}


/////////////////////////////////////////////////////////////////////////////
template <typename...> struct type_list {};
using value_type_list = type_list<float, double, int32_t, int64_t>;
using index_type_list = type_list<int32_t, int64_t>;
using size_type_list  = type_list<int32_t, int64_t>;
using my_type_pairs   = mp11::mp_product<type_list, value_type_list, index_type_list, size_type_list>;
template <typename Tx, typename Ty, typename Tz> void inject_fn(py::module_& m, const type_list<Tx, Ty, Tz>&) {
    static constexpr std::string_view base_name = "batch_sort_kernel_";
    static constexpr auto function_name = get_type_names<base_name, Tx, Ty, Tz>::value.data();
    m.def(function_name, &batchSort<Tx, Ty, Tz>,
                          py::arg("value"), 
                          py::arg("batch"),
                          py::arg("index_out"),
                          py::arg("increasing"));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    mp11::mp_for_each<my_type_pairs>([&](const auto& x) { inject_fn(m, x); });
}


