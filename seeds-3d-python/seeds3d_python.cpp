#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../seeds-3d/seeds3d.hpp"
#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>

namespace py = pybind11;
using namespace cv;
using namespace cv::ximgproc;

PYBIND11_MODULE(seeds3d, m) {
    py::class_<SuperpixelSEEDS3D, cv::Ptr<SuperpixelSEEDS3D>>(m, "SuperpixelSEEDS3D")
        .def(py::init([](int width, int height, int depth, int channels,
                         int num_superpixels, int num_levels, int prior,
                         int histogram_bins, bool double_step) {
            return createSuperpixelSEEDS3D(width, height, depth, channels,
                                           num_superpixels, num_levels, prior,
                                           histogram_bins, double_step);
        }))
        .def("iterate", &SuperpixelSEEDS3D::iterate)
        .def("getLabels", [](SuperpixelSEEDS3D& self) {
            cv::Mat labels;
            self.getLabels(labels);
            // Convert cv::Mat to NumPy array
            py::array_t<int> array(labels.size[0] * labels.size[1] * labels.size[2],
                                   labels.data);
            array.resize({labels.size[0], labels.size[1], labels.size[2]});
            return array;
        })
        // .def("getLabelContourMask", [](SuperpixelSEEDS3D& self, bool thick_line, int idx) {
        //     cv::Mat mask;
        //     self.getLabelContourMask(mask, thick_line, idx);
        //     // Convert cv::Mat to NumPy array
        //     py::array_t<uchar> array(mask.rows * mask.cols, mask.data);
        //     array.resize({mask.rows, mask.cols});
        //     return array;
        // })
        // .def("getNumberOfSuperpixels", &SuperpixelSEEDS3D::getNumberOfSuperpixels);
}