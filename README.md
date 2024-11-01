# seeds-3d


## Installation

Create the conda environment with the following command:
```
> conda env create -f environment.yaml
> conda activate seeds3d
```

Build OpenCV with the following command (https://thecodinginterface.com/blog/opencv-cpp-vscode/):
```
> git clone https://github.com/opencv/opencv.git
> git clone https://github.com/opencv/opencv_contrib.git
> mkdir build_opencv install
> cd build_opencv
> cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=../install \
      -D INSTALL_C_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ../opencv
> export CPUS=$(sysctl -n hw.physicalcpu)
> make -j $CPUS
> make install
```

Test OpenCV with the following command:
```
> modify opencv_test_code/CMakeLists.txt::#3::/your/path/to/install/lib/cmake/opencv4
> cd opencv_test_code/build
> cmake ..
> make
> ./test_demo
```