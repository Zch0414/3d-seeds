#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "seeds3d.hpp"

namespace py = pybind11;
using namespace cv;
using namespace cv::ximgproc;

PYBIND11_MODULE(seeds3d, m) {
    py::class_<SuperpixelSEEDS3D, std::shared_ptr<SuperpixelSEEDS3D>>(m, "SuperpixelSEEDS3D")
        .def("getNumberOfSuperpixels", &SuperpixelSEEDS3D::getNumberOfSuperpixels)
        .def("iterate", &SuperpixelSEEDS3D::iterate, 
             py::arg("data"), py::arg("num_iterations"))
        .def("getLabels", [](SuperpixelSEEDS3D& self) {
            cv::Mat labels;
            self.getLabels(labels);
            return labels;
        })
        .def("getLabelContourMask", [](SuperpixelSEEDS3D& self, bool thick_line, int idx) {
            cv::Mat mask;
            self.getLabelContourMask(mask, thick_line, idx);
            return mask;
        });

    m.def("createSuperpixelSEEDS3D", [](int width, int height, int depth, int channels, int num_superpixels,
                                       int num_levels, int prior, int num_histogram_bins, bool double_step) -> std::shared_ptr<SuperpixelSEEDS3D> {
        cv::Ptr<SuperpixelSEEDS3D> ptr = cv::ximgproc::createSuperpixelSEEDS3D(width, height, depth, channels, num_superpixels,
                                                                              num_levels, prior, num_histogram_bins, double_step);
        // Convert cv::Ptr to std::shared_ptr with a custom deleter
        return std::shared_ptr<SuperpixelSEEDS3D>(ptr.get(), [ptr](SuperpixelSEEDS3D*) mutable { ptr.release(); });
    });
}