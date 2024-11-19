# SEEDS3D

## Build Work Space

Build the work space with the following command:
```
> mkdir seeds3d && cd seeds3d
> git clone git@github.com:Zch0414/seeds3d.git
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

### Test OpenCV C++ with the following command:
```
cd /samples/imshow
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
> [manually modify] main.cpp:#6:/your/path/to/seeds3d/seeds3d/00.jpg
> mkdir build && cd build
> cmake ..
> make
> ./imshow
```

### Try SEEDS demo (OpenCV C++) with the following command:
```
cd /samples/seeds-opencv-cpp
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
> mkdir build && cd build
> cmake ..
> make
> ./seeds_demo /your/path/to/seeds3d/seeds3d/00.jpg
```
Note that a similar implementation can be found in **/seeds3d/samples/seeds-cpp**. It does not depend on ximgproc and can be used for debugging purposes.

## SEEDS-3D C++ Implementation

Prepare input with the following command:
```
> python nifti_to_array.py
```

Run SEEDS3D (C++) with the following command:
```
cd /samples/seeds3d-cpp
> [manually modify] seeds3d_sample.cpp:#35:/your/path/to/seeds3d/seeds3d/array_3d.bin
> [manually modify] seeds3d_sample.cpp:#95:/your/path/to/seeds3d/seeds3d/result.bin
> [manually modify] CMakeLists.txt:#3:/your/path/to/install_opencv/lib/cmake/opencv4
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

### Try SEEDS demo (OpenCV Python) with the following command:
```
> cd /samples/seeds-opencv-python
> python seeds.py
```

## SEEDS3D Python Installation

After activating the "seeds3d" environment, you can install the seeds3d package using the following commands:
```
[manually modify] setup.py:#8:/your/path/to/seeds3d/install_opencv/include/opencv4
[manually modify] setup.py:#9:/your/path/to/seeds3d/install_opencv/lib
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
