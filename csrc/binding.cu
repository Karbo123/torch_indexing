
#include <common.h>
#include <batch_sort.h>
#include <stable_argsort.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // batch_sort
    m.def("batch_sort_kernel_[f4,i8,i8]", batch_sort::batchSort<f4,i8,i8>, py::arg("value"), py::arg("batch"), py::arg("index_out"), py::arg("increasing"));
    m.def("batch_sort_kernel_[f8,i8,i8]", batch_sort::batchSort<f8,i8,i8>, py::arg("value"), py::arg("batch"), py::arg("index_out"), py::arg("increasing"));
    m.def("batch_sort_kernel_[i4,i8,i8]", batch_sort::batchSort<i4,i8,i8>, py::arg("value"), py::arg("batch"), py::arg("index_out"), py::arg("increasing"));
    m.def("batch_sort_kernel_[i8,i8,i8]", batch_sort::batchSort<i8,i8,i8>, py::arg("value"), py::arg("batch"), py::arg("index_out"), py::arg("increasing"));

    // stable_argsort
    m.def("stable_argsort_kernel_[f4,i8,i8]", stable_argsort::stableArgsort<f4,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[f8,i8,i8]", stable_argsort::stableArgsort<f8,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[i4,i8,i8]", stable_argsort::stableArgsort<i4,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[i8,i8,i8]", stable_argsort::stableArgsort<i8,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
}

