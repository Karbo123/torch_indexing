
#include <common.h>
#include <batch_sort.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // batch_sort
    mp11::mp_for_each<f4f8_i8_i8_pairs>([&](const auto& x) { batch_sort::inject_fn(m, x); });
}

