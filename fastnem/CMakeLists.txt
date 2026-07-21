cmake_minimum_required(VERSION 3.21)
project(fastnem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FASTNEM_TESTS "Build nem_engine unit tests" ON)

include(FetchContent)
FetchContent_Declare(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG        v0.8.2
)
FetchContent_MakeAvailable(simde)

add_library(fastnem INTERFACE)
target_include_directories(fastnem INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include>
    ${simde_SOURCE_DIR}
)
target_compile_features(fastnem INTERFACE cxx_std_20)
target_compile_options(fastnem INTERFACE -mavx2 -mfma)

find_package(Threads REQUIRED)
target_link_libraries(fastnem INTERFACE Threads::Threads)

add_subdirectory(app)

include(FetchContent)

if(FASTNEM_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

  enable_testing()
  add_subdirectory(tests)
endif()
