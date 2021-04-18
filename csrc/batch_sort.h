namespace batch_sort 
{
    template <typename ValueType, typename IndexType, typename SizeType>
    void batchSort_kernel(ValueType *value, IndexType *batch, IndexType *index_out, SizeType length, bool increasing, bool is_cuda)
    {   
        auto first = thrust::make_zip_iterator(thrust::make_tuple(batch, index_out));
        if (is_cuda)
        {
            auto policy = thrust::device;
            if (increasing) thrust::stable_sort_by_key(policy, value, value + length, first, thrust::less<ValueType>());
            else thrust::stable_sort_by_key(policy, value, value + length, first, thrust::greater<ValueType>());
            thrust::stable_sort_by_key(policy, batch, batch + length, index_out, thrust::less<IndexType>());
        } else
        {
            auto policy = thrust::host;
            if (increasing) thrust::stable_sort_by_key(policy, value, value + length, first, thrust::less<ValueType>());
            else thrust::stable_sort_by_key(policy, value, value + length, first, thrust::greater<ValueType>());
            thrust::stable_sort_by_key(policy, batch, batch + length, index_out, thrust::less<IndexType>());
        }
    }

    template <typename ValueType, typename IndexType, typename SizeType>
    void batchSort(torch::Tensor value, torch::Tensor batch, torch::Tensor index_out, bool increasing)
    {
        ValueType* value_ptr     = value.data_ptr<ValueType>();
        IndexType* batch_ptr     = batch.data_ptr<IndexType>();
        IndexType* index_out_ptr = index_out.data_ptr<IndexType>();
        SizeType   length        = value.size(0);

        bool is_cuda = value.is_cuda();
        batchSort_kernel<ValueType, IndexType, SizeType>(value_ptr, batch_ptr, index_out_ptr, length, increasing, is_cuda);
    }


    /////////////////////////////////////////////////////////////////////////////
    // binding function
    template <typename Tx, typename Ty, typename Tz> 
    void inject_fn(py::module_& m, const type_list<Tx, Ty, Tz>&) 
    {
        static constexpr std::string_view base_name = "batch_sort_kernel_";
        static constexpr auto function_name = get_type_names<base_name, Tx, Ty, Tz>::value.data();
        m.def(function_name, &batchSort<Tx, Ty, Tz>,
                              py::arg("value"), 
                              py::arg("batch"),
                              py::arg("index_out"),
                              py::arg("increasing"));
    }
}


