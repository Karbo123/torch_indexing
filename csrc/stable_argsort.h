namespace stable_argsort 
{
    template <typename ValueType, typename IndexType, typename SizeType>
    void stableArgsort_kernel(ValueType *value, IndexType *index_out, SizeType length, bool increasing, bool is_cuda)
    {
        if (is_cuda)
        {
            auto policy = thrust::device;
            if (increasing) thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::less<ValueType>());
            else thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::greater<ValueType>());
        } else
        {
            auto policy = thrust::host;
            if (increasing) thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::less<ValueType>());
            else thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::greater<ValueType>());
        }
    }

    template <typename ValueType, typename IndexType, typename SizeType>
    void stableArgsort(torch::Tensor value, torch::Tensor index_out, bool increasing)
    {
        ValueType* value_ptr     = value.data_ptr<ValueType>();
        IndexType* index_out_ptr = index_out.data_ptr<IndexType>();
        SizeType   length        = value.size(0);

        bool is_cuda = value.is_cuda();
        stableArgsort_kernel<ValueType, IndexType, SizeType>(value_ptr, index_out_ptr, length, increasing, is_cuda);
    }
}
