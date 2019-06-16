cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
set(CMAKE_PREFIX_PATH "../../libtorch/")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(example example.cpp)
target_link_libraries(example ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 11)