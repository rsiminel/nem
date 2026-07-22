#include <nanobind/nanobind.h>

namespace nb = nanobind;

#if defined(__aarch64__) && defined(__linux__)
#  include <sys/auxv.h>
#  ifndef HWCAP_SVE
#    define HWCAP_SVE (1 << 22)
#  endif
#endif

NB_MODULE(_cpu, m) {
    m.def("has_avx2", [] {
#if defined(__x86_64__) || defined(_M_X64)
        return (bool)__builtin_cpu_supports("avx2");
#else
        return false;
#endif
    });
    m.def("has_sse2", [] {
#if defined(__x86_64__) || defined(_M_X64)
        return (bool)__builtin_cpu_supports("sse2");
#else
        return false;
#endif
    });
    m.def("has_sve", [] {
#if defined(__aarch64__) && defined(__linux__)
        return (bool)(getauxval(AT_HWCAP) & HWCAP_SVE);
#else
        return false;
#endif
    });
}
