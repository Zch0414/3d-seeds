cmake_minimum_required(VERSION 3.12)
project(seeds3d_python)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

set(OpenCV_DIR /Users/Zach/Zch/Research/seeds-3d/install_opencv/lib/cmake/opencv4)
set(pybind11_DIR /opt/homebrew/Cellar/pybind11/2.13.6_1/include/pybind11)
find_package( OpenCV REQUIRED )
find_package(pybind11 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ../seeds-3d
)

pybind11_add_module(seeds3d MODULE
    seeds3d_bindings.cpp      # Binding code
    ../seeds-3d/seeds3d.cpp   # Implementation code from the seeds-3d directory
)

target_link_libraries(seeds3d PRIVATE ${OpenCV_LIBS})