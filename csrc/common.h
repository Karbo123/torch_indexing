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


template <typename...> struct type_list {};
using f4f8_list = type_list<float, double>;
using i8_list   = type_list<int64_t>;

using f4f8_i8_i8_pairs = mp11::mp_product<type_list, f4f8_list, i8_list, i8_list>;


