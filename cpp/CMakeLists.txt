cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_libtorch)
set(CMAKE_PREFIX_PATH "~/lib/libtorch/")
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(inference inference.cpp utils.cpp)
target_link_libraries(inference ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 11)