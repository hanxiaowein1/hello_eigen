cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_TOOLCHAIN_FILE "D:\\Library\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")

project(hello_eigen)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})
find_package(GTest CONFIG REQUIRED)

add_executable(${PROJECT_NAME}_test linear_equations_solver_test.cpp unit_test.cpp)
target_link_libraries(${PROJECT_NAME}_test GTest::gtest GTest::gmock)
