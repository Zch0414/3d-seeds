import pybind11
from setuptools import setup, Extension


# Path to your OpenCV installation in the project
opencv_include_dir = '/Users/Zach/Zch/Research/3d-seeds/install_opencv/include/opencv4'
opencv_library_dir = '/Users/Zach/Zch/Research/3d-seeds/install_opencv/lib'

ext_modules = [
    Extension(
        'python_3d_seeds',
        ['src/bindings.cpp', 'src/seeds.cpp'],
        include_dirs=[
            pybind11.get_include(),
            opencv_include_dir,
        ],
        library_dirs=[
            opencv_library_dir,
        ],
        libraries=['opencv_core', 'opencv_imgproc', 'opencv_ximgproc'],
        language='c++',
        extra_compile_args=['-std=c++11', '-fvisibility=hidden'],
        extra_link_args=[
            f'-Wl,-rpath,{opencv_library_dir}',
        ],
    ),
]

setup(
    name='python-3d-seeds',
    version='0.1.0',
    ext_modules=ext_modules,
    zip_safe=False,
)