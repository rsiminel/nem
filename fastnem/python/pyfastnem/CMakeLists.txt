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
target_compile_options(fastnem_headers INTERFACE -mavx2 -mfma -O3)

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

nanobind_add_module(_core STABLE_ABI src/bindings.cpp)
target_link_libraries(_core PRIVATE fastnem_headers)

install(TARGETS _core LIBRARY DESTINATION pyfastnem)
