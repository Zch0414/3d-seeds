#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "seeds3d.hpp"

namespace py = pybind11;
using namespace cv;
using namespace cv::ximgproc;

PYBIND11_MODULE(seeds3d, m) {
    py::class_<SuperpixelSEEDS3D, std::shared_ptr<SuperpixelSEEDS3D>>(m, "SuperpixelSEEDS3D")
        .def("getNumberOfSuperpixels", &SuperpixelSEEDS3D::getNumberOfSuperpixels)
        .def("iterate", [](SuperpixelSEEDS3D& self, py::array_t<float> data, int num_iterations) {
            // Validate input dimensions and type
            // if (data.ndim() != 3) {
            //     throw std::runtime_error("Input data must be a 3D NumPy array");
            // }
            // if (data.dtype().kind() != 'f') {  // 'f' stands for floating point
            //     throw std::runtime_error("Input data must be of type float32");
            // }
            // // Ensure data is contiguous
            // if (!data.flags() & py::array::c_style) {
            //     throw std::runtime_error("Input data must be C-contiguous");
            // }
            // Convert NumPy array to cv::Mat without copying data
            py::buffer_info buf = data.request();
            int depth = buf.shape[0];
            int height = buf.shape[1];
            int width = buf.shape[2];
            cv::Mat mat(3, (int[]){depth, height, width}, CV_32FC1, (void*)buf.ptr);
            self.iterate(mat, num_iterations);
        },
        py::arg("data"), 
        py::arg("num_iterations")
        )
        .def("getLabels", [](SuperpixelSEEDS3D& self) -> py::array_t<int> {
            cv::Mat labels;
            self.getLabels(labels);
            //  if (labels.dims != 3) {
            //      throw std::runtime_error("Labels must be a 3D Mat");
            //  }
            //  if (labels.type() != CV_32S) {
            //      throw std::runtime_error("Labels must be of type CV_32S (int)");
            //  }
            // Create a NumPy array that shares memory with the 3D cv::Mat
            return py::array_t<int>(
                { labels.size[0], labels.size[1], labels.size[2] }, // Shape: depth, height, width
                { sizeof(int) * labels.size[1] * labels.size[2], sizeof(int) * labels.size[2], sizeof(int) }, // Strides
                labels.ptr<int>(),                                      // Data pointer
                py::cast(labels)                                        // Reference to keep data alive
            );
        })
        .def("getLabelContourMask", [](SuperpixelSEEDS3D& self, bool thick_line, int idx) -> py::array_t<unsigned char> {
            cv::Mat mask;
            self.getLabelContourMask(mask, thick_line, idx);
            // if (mask.type() != CV_8U) {
            //     throw std::runtime_error("Contour mask must be of type CV_8U (unsigned char)");
            // }
            // Create a NumPy array that shares memory with the cv::Mat
            return py::array_t<unsigned char>(
                { mask.rows, mask.cols },                   // Shape
                { sizeof(unsigned char) * mask.cols, sizeof(unsigned char) }, // Strides
                mask.ptr<unsigned char>(),                   // Data pointer
                py::cast(mask)                               // Reference to keep data alive
            );
        });

    m.def("createSuperpixelSEEDS3D", [](int width, int height, int depth, int channels, int num_superpixels,
                                       int num_levels, int prior, int num_histogram_bins, bool double_step) -> std::shared_ptr<SuperpixelSEEDS3D> {
        cv::Ptr<SuperpixelSEEDS3D> ptr = cv::ximgproc::createSuperpixelSEEDS3D(width, height, depth, channels, num_superpixels,
                                                                              num_levels, prior, num_histogram_bins, double_step);
        // Convert cv::Ptr to std::shared_ptr with a custom deleter
        return std::shared_ptr<SuperpixelSEEDS3D>(ptr.get(), [ptr](SuperpixelSEEDS3D*) mutable { ptr.release(); });
    });
}