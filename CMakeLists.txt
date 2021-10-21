cmake_minimum_required(VERSION 3.16)
project(hmc)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SRC LIST_DIRECTORIES true src/*.hpp src/*.cpp)
set(SRC ${SRC})

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE lib/eigen)
target_include_directories(${PROJECT_NAME} INTERFACE src)
target_include_directories(${PROJECT_NAME} INTERFACE lib/eigen)