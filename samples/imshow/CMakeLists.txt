cmake_minimum_required(VERSION 3.10)
project(imshow)
set(OpenCV_DIR /Users/Zach/Zch/Research/3d-seeds/install_opencv/lib/cmake/opencv4)
set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(imshow main.cpp)
target_link_libraries(imshow ${OpenCV_LIBS})