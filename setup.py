from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os


# Path to your OpenCV installation in the project
opencv_include_dir = '/Users/Zach/Zch/Research/seeds3d/install_opencv/include/opencv4'
opencv_library_dir = '/Users/Zach/Zch/Research/seeds3d/install_opencv/lib'

ext_modules = [
    Extension(
        'seeds3d',
        ['src/bindings.cpp', 'src/seeds3d.cpp'],
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
    name='seeds3d',
    version='0.1.0',
    ext_modules=ext_modules,
    zip_safe=False,
)