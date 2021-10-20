set(BINARY cpp-hmc)

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})

add_executable(cpp-hmc_run ${SOURCES})

add_library(cpp-hmc_lib STATIC ${SOURCES})