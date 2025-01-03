#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "seeds.hpp"

namespace py = pybind11;
using namespace cv;
using namespace cv::ximgproc;

PYBIND11_MODULE(python_3d_seeds, m) {
    py::class_<SupervoxelSEEDS, std::shared_ptr<SupervoxelSEEDS>>(m, "SupervoxelSEEDS")
        .def("getNumberOfSuperpixels", &SupervoxelSEEDS::getNumberOfSuperpixels)
        .def("iterate", 
            [](SupervoxelSEEDS& self, py::array_t<float> data, int num_iterations) {
                // // Validate input dimensions and type
                // if (data.ndim() != 3) {
                //     throw std::runtime_error("Input data must be a 3D NumPy array");
                // }
                // if (data.dtype().kind() != 'f') {  // 'f' stands for floating point
                //     throw std::runtime_error("Input data must be of type float32");
                // }
                // // Ensure data is contiguous
                // if (!(data.flags() & py::array::c_style)) {
                //     throw std::runtime_error("Input data must be C-contiguous");
                // }

                // Convert NumPy array to 3D cv::Mat without copying data
                py::buffer_info buf = data.request();
                int depth = buf.shape[0];
                int height = buf.shape[1];
                int width = buf.shape[2];
                 
                // // Validate that all dimensions are positive
                // if (depth <= 0 || height <= 0 || width <= 0) {
                //     throw std::runtime_error("All dimensions must be positive");
                // }
                 
                int sizes[3] = { depth, height, width };
                cv::Mat mat(3, sizes, CV_32FC1, (void*)buf.ptr);
                self.iterate(mat, num_iterations);
            }, 
            py::arg("data"), py::arg("num_iterations")
        )
        .def("getLabels", 
            [](SupervoxelSEEDS& self) -> py::array_t<int> {
                cv::Mat labels;
                self.getLabels(labels);
                 
                // // Validate that labels is a 3D Mat
                // if (labels.dims != 3) {
                //     throw std::runtime_error("Labels must be a 3D Mat");
                // }
                 
                // // Validate data type
                // if (labels.type() != CV_32S) {
                //     throw std::runtime_error("Labels must be of type CV_32S (int)");
                // }
                 
                // // Ensure all dimensions are positive
                // for(int i = 0; i < labels.dims; ++i){
                //     if(labels.size[i] <= 0 ){
                //         throw std::runtime_error("Label dimensions must be positive");
                //     }
                // }
                 
                // Convert OpenCV's size to a std::vector<size_t> for NumPy
                std::vector<size_t> shape(labels.dims);
                std::vector<size_t> strides(labels.dims);
                size_t stride_bytes = sizeof(int);
                for(int i = labels.dims -1; i >=0; --i){
                    shape[i] = static_cast<size_t>(labels.size[i]);
                    strides[i] = stride_bytes;
                    stride_bytes *= labels.size[i];
                }
                 
                // Create a shared_ptr to manage the cv::Mat
                std::shared_ptr<cv::Mat> labels_ptr = std::make_shared<cv::Mat>(labels);
                
                // Create a capsule that holds the shared_ptr
                py::capsule labels_capsule(new std::shared_ptr<cv::Mat>(labels_ptr), [](void* c) {
                    delete static_cast<std::shared_ptr<cv::Mat>*>(c);
                });
                
                // Create the NumPy array that shares memory with the 3D cv::Mat
                return py::array_t<int>(
                    shape,                // Shape: depth, height, width
                    strides,              // Strides
                    labels_ptr->ptr<int>(),// Data pointer
                    labels_capsule         // Capsule to keep data alive
                );
            }
        );

    m.def("createSupervoxelSEEDS", 
        [](int width, int height, int depth, int channels, 
        int num_superpixels, int num_levels, int prior, int histogram_bins, 
        bool double_step) -> std::shared_ptr<SupervoxelSEEDS> {
            cv::Ptr<SupervoxelSEEDS> ptr = cv::ximgproc::createSupervoxelSEEDS(
                width, height, depth, channels, num_superpixels,
                num_levels, prior, histogram_bins, double_step
            );
            // Convert cv::Ptr to std::shared_ptr with a custom deleter
            return std::shared_ptr<SupervoxelSEEDS>(ptr.get(), [ptr](SupervoxelSEEDS*) mutable { ptr.release(); });
        },
        py::arg("width"), py::arg("height"), py::arg("depth"), py::arg("channels"),
        py::arg("num_superpixels"),
        py::arg("num_levels") = 2,
        py::arg("prior") = 2,
        py::arg("histogram_bins") = 15,
        py::arg("double_step") = false
    );
}