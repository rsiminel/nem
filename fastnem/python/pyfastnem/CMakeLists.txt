cmake_minimum_required(VERSION 3.21)
project(pyfastnem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_library(fastnem_headers INTERFACE)
target_include_directories(fastnem_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include)
target_compile_features(fastnem_headers INTERFACE cxx_std_20)
target_compile_options(fastnem_headers INTERFACE -O3)

include(FetchContent)
FetchContent_Declare(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG        v0.8.2
)
FetchContent_MakeAvailable(simde)
target_include_directories(fastnem_headers INTERFACE ${simde_SOURCE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(fastnem_headers INTERFACE Threads::Threads)

nanobind_add_module(_cpu STABLE_ABI src/cpu_detect.cpp)
install(TARGETS _cpu LIBRARY DESTINATION pyfastnem)

function(add_core_variant SUFFIX)
    nanobind_add_module(_core_${SUFFIX} STABLE_ABI src/bindings.cpp)
    target_compile_definitions(_core_${SUFFIX} PRIVATE
        "FASTNEM_MODULE_NAME=_core_${SUFFIX}")
    target_compile_options(_core_${SUFFIX} PRIVATE ${ARGN})
    target_link_libraries(_core_${SUFFIX} PRIVATE fastnem_headers)
    install(TARGETS _core_${SUFFIX} LIBRARY DESTINATION pyfastnem)
endfunction()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_core_variant(scalar  -march=x86-64)
    add_core_variant(sse2    -msse2)
    add_core_variant(avx2    -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    if(APPLE)
        add_core_variant(darwin_arm -mcpu=apple-m1)
    else()
        add_core_variant(arm8     -march=armv8-a)
        add_core_variant(arm8_sve -march=armv8-a+sve)
    endif()
else()
    message(WARNING "Unrecognised processor ${CMAKE_SYSTEM_PROCESSOR}, building scalar only")
    add_core_variant(scalar)
endif()
