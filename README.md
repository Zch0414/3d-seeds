# 3D SEEDS

## Build Work Space

**Build the work space with the following command:**
```
> mkdir 3d-seeds && cd 3d-seeds
```

## OpenCV C++ Installation (in current directory)

**Build OpenCV with the following command (https://thecodinginterface.com/blog/opencv-cpp-vscode/):**
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

## 3D SEEDS (C++)

**First, clone this repository with the following command:**
```
> git clone git@github.com:Zch0414/3d-seeds.git
```

**Your workspace should like:**
- 3d-seeds/
  - build_opencv/
  - install_opencv/
  - opencv/
  - opencv_contrib/
  - **3d-seeds/ (this repository)**
    
**The following command should be executed within the **3d-seeds/ (this repository)**. So first:**
```
> cd 3d-seeds
> rm -rf .git
```

### CMakeLists.txt

Here are CMakeLists templates for Mac and Linux:

1. **Mac samples/mat/CMakeLists.txt:**
```
> cmake_minimum_required(VERSION 2.8)
> project(mat)
> set(OpenCV_DIR /your/path/to/3d-seeds/install_opencv/lib/cmake/opencv4)
> set(CMAKE_CXX_STANDARD 14)
> set(CMAKE_BUILD_TYPE Debug)
> find_package( OpenCV REQUIRED )
> include_directories(${OpenCV_INCLUDE_DIRS})
> add_executable(mat ${SOURCES})
> target_link_libraries(mat ${OpenCV_LIBS})
```

2. **Linux (gcc/10.3.0) samples/mat/CMakeLists.txt:**
```
> cmake_minimum_required(VERSION 3.10)
> project(mat)
> set(CMAKE_PREFIX_PATH "/your/path/to/3d-seeds/install_opencv" ${CMAKE_PREFIX_PATH})
> set(OpenCV_DIR /your/path/to/3d-seeds/install_opencv/share/opencv4)
> set(CMAKE_CXX_STANDARD 14)
> set(CMAKE_CXX_STANDARD_REQUIRED ON)
> set(CMAKE_CXX_EXTENSIONS OFF)
> find_package( OpenCV REQUIRED )
> include_directories( ${OpenCV_INCLUDE_DIRS} )
> add_executable(mat main.cpp)
> target_link_libraries(mat ${OpenCV_LIBS})
```

### Test OpenCV

**Test Opencv (C++) with the following command:**
```
cd samples/mat
> [manually modify] CMakeLists.txt
> mkdir build && cd build
> cmake ..
> make
> ./mat
```

### SEEDS Demo (OpenCV && Only Tested on Mac)

**Try SEEDS demo with the following command:**
```
cd samples/opencv-cpp-seeds
> [manually modify] CMakeLists.txt
> mkdir build && cd build
> cmake ..
> make
> ./demo /your/path/to/3d-seeds/3d-seeds/data/00.jpg
```

### 3D SEEDS Demo (Only Tested on Mac)

**Prepare input with the following command:**
```
> cd data
> python nifti_to_array.py
> cd ..
```

This will give you input.bin, and input.nii.gz for further visualization purposes. 
Both files will be saved in /your/path/to/3d-seeds/3d-seeds/data directory.

**Run 3D SEEDS (C++) with the following command:**
```
cd samples/cpp_3d-seeds
> [manually modify] CMakeLists.txt
> [manually modify] seeds.cpp:#35:/your/path/to/3d-seeds/3d-seeds/data/input.bin
> [manually modify] seeds.cpp:#97:/your/path/to/3d-seeds/3d-seeds/data/result.bin
> mkdir build && cd build
> cmake ..
> make
> ./demo
```

This will provide an interactive window where you can adjust various hyperparameters in the SEEDS algorithm. 
Additionally, it will save result.bin in the /your/path/to/3d-seeds/3d-seeds/data directory.

**You can get a NIfTI file using the following command:**
```
> cd data
> python bin_to_nifti.py
> cd ..
```

This will generate a result.nii.gz file, bound with input.nii.gz, which can be visualized with 3D Slicer.

## 3D SEEDS (Python)

### Create Python Environment

**Create the conda environment with the following command:**
```
> conda env create -f environment.yaml
> conda activate 3d-seeds
```

### SEEDS Demo (OpenCV && Only Tested on Mac)

**Try SEEDS demo with the following command:**
```
> cd samples/opencv-python-seeds
> python seeds.py
```

### 3D SEEDS Installation

**After activating the "3d-seeds" environment, you can install the python_3d-seeds package using the following commands:**
```
[manually modify] setup.py:#6: /your/path/to/3d-seeds/install_opencv/include/opencv4
[manually modify] setup.py:#7
  - (For Mac): /your/path/to/3d-seeds/install_opencv/lib
  - (For Linux): /your/path/to/3d-seeds/install_opencv/lib64
> python setup.py clean --all
> python setup.py build
> pip install .
```

### 3D SEEDS Demo:

**Try 3D SEEDS demo with the following command:**
```
> cd samples/python-3d-seeds
> python seeds.py
```

This will generate a result.nii.gz file in the /your/path/to/3d-seeds/3d-seeds/samples/python-3d-seeds.
Bound with /your/path/to/3d-seeds/3d-seeds/data/input.nii.gz, you can visualize them with 3D Slicer.
