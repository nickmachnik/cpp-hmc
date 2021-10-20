cmake_minimum_required(VERSION 3.16)
project(hmc)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)
include_directories(lib/eigen)

add_subdirectory(src)
add_subdirectory(lib/eigen)