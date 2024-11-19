# SEEDS3D

## Build Work Space

Build the work space with the following command:
```
> mkdir seeds3d && cd seeds3d
```

## OpenCV C++ Installation (in current directory)

Build OpenCV with the following command (https://thecodinginterface.com/blog/opencv-cpp-vscode/):
```
> git clone https://github.com/opencv/opencv.git (4.10.84)
> git clone https://github.com/opencv/opencv_contrib.git (4.10.84)
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

## Build SEEDS3D
```
> git clone git@github.com:Zch0414/seeds3d.git
```

Your workspace should like:
- seeds3d/
  - build_opencv/
  - install_opencv/
  - opencv/
  - opencv_contrib/
  - **seeds3d/ (this repository)**
    
The following command should be executed within the **seeds3d/ (this repository)**. So first:
```
> cd seeds3d
```

### CMakeLists.txt

Here are CMakeLists templates for Mac and Linux:

1. Mac /samples/mat/CMakeLists.txt:

```
> cmake_minimum_required(VERSION 2.8)
> project(test_demo)
> set(OpenCV_DIR /your/path/to/seeds3d/install_opencv/lib/cmake/opencv4)
> set(CMAKE_CXX_STANDARD 14)
> set(CMAKE_BUILD_TYPE Debug)
> find_package( OpenCV REQUIRED )
> include_directories(${OpenCV_INCLUDE_DIRS})
> add_executable(test_demo ${SOURCES})
> target_link_libraries(test_demo ${OpenCV_LIBS})
```

2. Linux (gcc/10.3.0) /samples/mat/CMakeLists.txt:

```
> cmake_minimum_required(VERSION 3.10)
> project(test_demo)
> set(CMAKE_PREFIX_PATH "/your/path/to/seeds3d/install_opencv" ${CMAKE_PREFIX_PATH})
> set(OpenCV_DIR /your/path/to/seeds3d/install_opencv/share/opencv4)
> set(CMAKE_CXX_STANDARD 14)
> set(CMAKE_CXX_STANDARD_REQUIRED ON)
> set(CMAKE_CXX_EXTENSIONS OFF)
> find_package( OpenCV REQUIRED )
> include_directories( ${OpenCV_INCLUDE_DIRS} )
> add_executable(test_demo main.cpp)
> target_link_libraries(test_demo ${OpenCV_LIBS})
```

### Test OpenCV C++ with the following command:
```
cd /samples/mat
> [manually modify] CMakeLists.txt
> mkdir build && cd build
> cmake ..
> make
> ./test_demo
```

### Try SEEDS demo (OpenCV C++) with the following command:
```
cd /samples/seeds-opencv-cpp
> [manually modify] CMakeLists.txt
> mkdir build && cd build
> cmake ..
> make
> ./seeds_demo /your/path/to/seeds3d/seeds3d/00.jpg
```
Note that a similar implementation can be found in **/seeds3d/samples/seeds-cpp**. It does not depend on ximgproc and can be used for debugging purposes.

### Try SEEDS3D demo (C++) (only support Mac)

Prepare input with the following command:
```
> python nifti_to_array.py
```

Run SEEDS3D (C++) with the following command:
```
cd /samples/seeds3d-cpp
> [manually modify] CMakeLists.txt
> [manually modify] seeds3d_sample.cpp:#35:/your/path/to/seeds3d/seeds3d/array_3d.bin
> [manually modify] seeds3d_sample.cpp:#95:/your/path/to/seeds3d/seeds3d/result.bin
> mkdir build && cd build
> cmake ..
> make
> ./seeds3d_sample
```
This will provide an interactive window where you can adjust various hyperparameters in the SEEDS algorithm. 
Additionally, it will save result.bin in the /seeds3d directory, which you can process using the following command and visualize with 3D Slicer:
```
> python bin_to_nifti.py
```
This will generate a /results.nii.gz file, which can be visualized using 3D Slicer.

## Create Python Environment

Create the conda environment with the following command:
```
> conda env create -f environment.yaml
> conda activate seeds3d
```

### Try SEEDS demo (OpenCV Python) with the following command (only support Mac):
```
> cd /samples/seeds-opencv-python
> python seeds.py
```

## SEEDS3D Python Installation

After activating the "seeds3d" environment, you can install the seeds3d package using the following commands:
```
[manually modify] setup.py:#8:/your/path/to/seeds3d/install_opencv/include/opencv4
[manually modify]
  - (For Mac) setup.py:#9:/your/path/to/seeds3d/install_opencv/lib
  - (For Linux) setup.py:#9:/your/path/to/seeds3d/install_opencv/lib64
> python setup.py clean --all
> python setup.py build
> pip install .
```

### Try SEEDS3D demo (Python) with the following command:
```
> cd /samples/seeds3d-python
> python seeds3d_sample.py
```
This will generate a /samples/seeds3d-python/results.nii.gz file, which can be visualized using 3D Slicer.
