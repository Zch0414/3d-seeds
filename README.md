# seeds-3d

## Build Work Space

Build the work space with the following command:
```
> mkdir seeds-3d && cd seeds-3d
> git clone git@github.com:Zch0414/seeds-3d.git
```

## OpenCV C++ Installation

Build OpenCV with the following command (https://thecodinginterface.com/blog/opencv-cpp-vscode/):
```
> git clone https://github.com/opencv/opencv.git
> git clone https://github.com/opencv/opencv_contrib.git
> mkdir build_opencv install_opencv
> cd build_opencv
> cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=../install_opencv \
      -D INSTALL_C_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ../opencv
> export CPUS=$(sysctl -n hw.physicalcpu)
> make -j $CPUS
> make install
```

Test OpenCV C++ with the following command:
```
> cd seeds-3d/samples/imshow
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
> [manually modify] main.cpp:#6:/your/path/to/seeds-3d/00.jpg
> mkdir build && cd build
> cmake ..
> make
> ./imshow
```

Try SEEDS demo (C++) with the following command:
```
> cd seeds-3d/samples/seeds-cpp
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
> mkdir build && cd build
> cmake ..
> make
> ./seeds_demo /your/path/to/seeds-3d/00.jpg
```

## SEEDS-3D C++ Implementation

Prepare input with the following command:
```
> python nifti_to_array.py
```

Run SEEDS3D with the following command:
```
> cd seeds-3d
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
> mkdir build && cd build
> cmake ..
> make
> ./seeds3d_demo
```

Prepare output for visualization with 3D Slicer
```
> cd ..
> python bin_to_nifti.py
```

## OpenCV Python Installation

Create the conda environment with the following command:
```
> conda env create -f environment.yaml
> conda activate seeds3d
```

Try SEEDS demo (Python) with the following command:
```
> cd seeds-3d/samples/seeds-python3
> python seeds.py
```