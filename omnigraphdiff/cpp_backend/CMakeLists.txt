cmake_minimum_required(VERSION 3.14)
project(omnigraph_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include PyTorch (from Python)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/sparse_graph.cpp
    src/graph_ops.cpp
    src/message_passing.cpp
    src/sampling.cpp
)

# Main library (static for internal use)
add_library(omnigraph_core STATIC ${SOURCES})
target_link_libraries(omnigraph_core
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${TORCH_LIBRARIES}
)

# Python bindings (shared library)
pybind11_add_module(omnigraph_cpp src/bindings.cpp ${SOURCES})
target_link_libraries(omnigraph_cpp PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${TORCH_LIBRARIES}
)

# Set output directory
set_target_properties(omnigraph_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../omnigraphdiff
)

# Tests (optional)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "Google Test not found, skipping tests")
    endif()
endif()

# Installation
install(TARGETS omnigraph_cpp
    LIBRARY DESTINATION ${Python3_SITELIB}/omnigraphdiff
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
