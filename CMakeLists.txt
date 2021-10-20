cmake_minimum_required(VERSION 3.16)
project(hmc)

set(CMAKE_CXX_STANDARD 17)

# ------ stuff in src, for main binary
file(GLOB_RECURSE SRC LIST_DIRECTORIES true src/*.hpp src/*.cpp)
set(SRC ${SRC})

add_executable(
    ${PROJECT_NAME}
    ${SRC}
)

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PUBLIC lib/eigen)

add_subdirectory(lib/eigen)