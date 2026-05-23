cmake_minimum_required(VERSION 3.20)
project(jumpstar_chinese_checkers_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cczero_core src/cczero.cpp src/model.cpp src/mcts.cpp src/cli_utils.cpp)
target_include_directories(cczero_core PUBLIC include)
target_compile_definitions(cczero_core PUBLIC $<$<CONFIG:Release>:CCZERO_RELEASE_BUILD>)

add_executable(cczero src/main.cpp src/cli_main.cpp)
target_link_libraries(cczero PRIVATE cczero_core)
option(CCZERO_USE_ACCELERATE "Use Apple Accelerate for neural inference when available" ON)
if(APPLE AND CCZERO_USE_ACCELERATE)
  target_compile_definitions(cczero_core PUBLIC CCZERO_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(cczero_core PUBLIC "-framework Accelerate")
endif()

enable_testing()
add_executable(cczero_tests tests/core_tests.cpp)
target_link_libraries(cczero_tests PRIVATE cczero_core)
add_test(NAME cczero_tests COMMAND cczero_tests)
