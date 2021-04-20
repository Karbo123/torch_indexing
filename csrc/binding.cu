
#include <common.h>
#include <batch_sort.h>
#include <stable_argsort.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // batch_sort
    mp11::mp_for_each<f4f8i4i8_i8_i8_pairs>([&](const auto& x) { batch_sort::inject_fn(m, x); });

    // stable_sort
    mp11::mp_for_each<f4f8i4i8_i8_i8_pairs>([&](const auto& x) { stable_argsort::inject_fn(m, x); });
}

