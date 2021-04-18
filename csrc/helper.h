////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// copy from: https://stackoverflow.com/questions/35941045/can-i-obtain-c-type-names-in-a-constexpr-way

#include <iostream>
#include <string_view>
// If you can't use C++17's standard library, you'll need to use the GSL 
// string_view or implement your own struct (which would not be very difficult,
// since we only need a few methods here)
template <typename T> constexpr std::string_view type_name();

template <>
constexpr std::string_view type_name<void>()
{ return "void"; }

namespace detail {

using type_name_prober = void;

template <typename T>
constexpr std::string_view wrapped_type_name() 
{
#ifdef __clang__
    return __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
}

constexpr std::size_t wrapped_type_name_prefix_length() { 
    return wrapped_type_name<type_name_prober>().find(type_name<type_name_prober>()); 
}

constexpr std::size_t wrapped_type_name_suffix_length() { 
    return wrapped_type_name<type_name_prober>().length() 
        - wrapped_type_name_prefix_length() 
        - type_name<type_name_prober>().length();
}

} // namespace detail

template <typename T>
constexpr std::string_view type_name() {
    constexpr auto wrapped_name = detail::wrapped_type_name<T>();
    constexpr auto prefix_length = detail::wrapped_type_name_prefix_length();
    constexpr auto suffix_length = detail::wrapped_type_name_suffix_length();
    constexpr auto type_name_length = wrapped_name.length() - prefix_length - suffix_length;
    return wrapped_name.substr(prefix_length, type_name_length);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// copy from: https://stackoverflow.com/questions/38955940/how-to-concatenate-static-strings-at-compile-time

#include <array>

template <std::string_view const&... Strs>
struct join
{
    // Join all strings into a single std::array of chars
    static constexpr auto impl() noexcept
    {
        constexpr std::size_t len = (Strs.size() + ... + 0);
        std::array<char, len + 1> arr{};
        auto append = [i = 0, &arr](auto const& s) mutable {
            for (auto c : s) arr[i++] = c;
        };
        (append(Strs), ...);
        arr[len] = 0;
        return arr;
    }
    // Give the joined string static storage
    static constexpr auto arr = impl();
    // View as a std::string_view
    static constexpr std::string_view value {arr.data(), arr.size() - 1};
};
// Helper to get the value out
template <std::string_view const&... Strs>
static constexpr auto join_v = join<Strs...>::value;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// combining the above two

template <std::string_view const& Base, typename ... T>
struct get_type_names
{
    static constexpr auto impl() noexcept
    {
        std::array<char, 1000> arr{};
        int i = 0;
        auto append = [&i, &arr](auto const& s) mutable {
            for (auto c : s) arr[i++] = (c==' ' ? '_' : c);
            arr[i++] = ',';
        };
        for (auto c : Base) arr[i++] = c;
        arr[i++] = '[';
        (append(type_name<T>()), ...);
        arr[i-1] = ']';
        return arr;
    }
    static constexpr auto arr = impl();
    static constexpr std::string_view value {arr.data(), arr.size() - 1};
};

