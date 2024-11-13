
#include "seeds3d_zch.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

int main()
{
    int depth = 48;
    int height = 192;
    int width = 192;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<> dis(INT_MIN, INT_MAX);

    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         for (int k = 0; k < depth; k++) {
    //             img_data.at<int>(i, j, k) = dis(gen)%255;  
    //         }
    //     }
    // }

    std::ifstream file("/Users/Zach/Zch/Research/seeds-3d/seeds-3d/seeds-3d/array_3d.bin", std::ios::binary);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }

    // Create a 1D vector to hold the data
    std::vector<float> data(depth * height * width);

    // Read the binary data into the vector
    file.read(reinterpret_cast<char*>(data.data()), depth * height * width * sizeof(float));
    file.close();
    
    Mat img_data = Mat(3, (int[]){depth, height, width}, CV_32FC1, data.data());
    // std::cout << "Value at (24, 96, 96): " << img_data.at<float>(24, 96, 96) << std::endl;
    
    double t = (double)getTickCount();
    Ptr<SuperpixelSEEDS3D> seeds;
    seeds = createSuperpixelSEEDS3D(192, 192, 48, 1, 432, 2, 2, 15);
    seeds->iterate(img_data, 20);
    t = ((double)getTickCount() - t) / getTickFrequency();
    printf("SEEDS segmentation took %i ms with %3i superpixels\n",
            (int) (t * 1000), seeds->getNumberOfSuperpixels());
    
    Mat labels;
    seeds->getLabels(labels);

    // int* labels_data = (int*)labels.data;
    // for (int i=0; i<192*192*48; i++)
    // {
    //     cout<<labels_data[i];
    //     if (i%20==0) cout<<endl;
    // }
    
    std::ofstream out_file("/Users/Zach/Zch/Research/seeds-3d/seeds-3d/seeds-3d/result.bin", std::ios::out | std::ios::binary);
    if (!out_file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return -1;
    }

    // Write the data directly from the Mat object
    out_file.write(reinterpret_cast<char*>((int*)labels.data), labels.total() * labels.elemSize());
    out_file.close();

    std::cout << "3D Mat data saved successfully!" << std::endl;
}