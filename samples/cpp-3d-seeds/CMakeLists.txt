cmake_minimum_required(VERSION 2.8)
project(demo)
set(OpenCV_DIR /Users/Zach/Zch/Research/3d-seeds/install_opencv/lib/cmake/opencv4)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCV REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ../../src
)
set(
    SOURCES 
    seeds.cpp 
    ../../src/seeds.cpp
)
add_executable(demo ${SOURCES})
target_link_libraries(demo ${OpenCV_LIBS})
