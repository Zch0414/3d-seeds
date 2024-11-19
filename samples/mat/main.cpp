#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Dimensions of the 3D matrix
    cv::Mat mat3D;
    // Create a 3D Mat with a single channel of 32-bit integer type
    int sizes[3] = {2, 4, 4};
    mat3D = cv::Mat(3, sizes, CV_32SC1);
    // Fill the matrix with values 0 to 26
    for (int i = 0; i < 32; ++i) {
        mat3D.data[i * sizeof(int)] = i;  // Populate each element in the 3D Mat
    }

    // Output the 3D matrix elements and their flattened indices
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                int index = z * 4 * 4 + y * 4 + x;
                std::cout << "mat3D[" << z << "][" << y << "][" << x << "] = " << mat3D.at<int>(z, y, x)
                          << " (Flattened index: " << index << ")" << std::endl;
            }
        }
    }

    return 0;
}