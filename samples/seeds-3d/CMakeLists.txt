cmake_minimum_required(VERSION 2.8)
project(seeds3d_sample)
set(OpenCV_DIR /Users/Zach/Zch/Research/seeds-3d/install_opencv/lib/cmake/opencv4)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCV REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ../../seeds-3d
)
set(
    SOURCES 
    seeds3d_sample.cpp 
    ../../seeds-3d/seeds3d.cpp
)
add_executable(seeds3d_sample ${SOURCES})
target_link_libraries(seeds3d_sample ${OpenCV_LIBS})
