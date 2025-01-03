
#include "../../src/seeds.hpp"
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

static const char *window_name = "SEEDS Superpixels";

static bool init = false;

void trackbarChanged(int, void *)
{
    init = false;
}

int main()
{
    int width = 160;
    int height = 160;
    int depth = 160;

    std::ifstream file("/Users/Zach/Zch/Research/3d_seeds/3d_seeds/data/input.bin", std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }
    std::vector<float> data(depth * height * width);
    file.read(reinterpret_cast<char *>(data.data()), depth * height * width * sizeof(float));
    file.close();
    int _img_size[3] = {depth, height, width};
    Mat img_data = Mat(3, _img_size, CV_32FC1, data.data());
    Mat frame(height, width, CV_8UC1);

    namedWindow(window_name, 0);
    int num_iterations = 4;
    int prior = 2;
    bool double_step = false;
    int num_superpixels = 512;
    int num_levels = 4;
    int num_histogram_bins = 5;
    int idx = 1;
    createTrackbar("Slice", window_name, &idx, depth-1, trackbarChanged);
    createTrackbar("# Pixels", window_name, &num_superpixels, 1024, trackbarChanged);
    createTrackbar("# Levels", window_name, &num_levels, 10, trackbarChanged);
    createTrackbar("# Bins", window_name, &num_histogram_bins, 15, trackbarChanged);
    createTrackbar("Prior", window_name, &prior, 5, trackbarChanged);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Ptr<SupervoxelSEEDS> seeds;
    Mat mask;
    int display_mode = 0;
    for (;;)
    {
        if (!init)
        {
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    frame.at<uchar>(i, j) = static_cast<uchar>(data[idx * width * height + i * height + j]);
                }
            }
            seeds = createSupervoxelSEEDS(width, height, depth, 1, num_superpixels,
                                            num_levels, prior, num_histogram_bins, double_step);
            init = true;
        }
        Mat result = frame.clone();

        double t = (double)getTickCount();

        Mat input = img_data / 255.0f;
        seeds->iterate(input, num_iterations);

        t = ((double)getTickCount() - t) / getTickFrequency();
        printf("3D SEEDS segmentation took %i ms with %3i superpixels\n",
               (int)(t * 1000), seeds->getNumberOfSuperpixels());

        // /* retrieve the segmentation result */
        Mat labels;
        seeds->getLabels(labels);

        std::ofstream out_file("/Users/Zach/Zch/Research/3d_seeds/3d_seeds/data/result.bin", std::ios::out | std::ios::binary);
        if (!out_file) {
            std::cerr << "Error opening file for writing!" << std::endl;
            return -1;
        }

        // Write the data directly from the Mat object
        out_file.write(reinterpret_cast<char*>((int*)labels.data), labels.total() * labels.elemSize());
        out_file.close();
        std::cout << "3D Mat data saved successfully!" << std::endl;

        // // /* get the contours for displaying */
        seeds->getLabelContourMask(mask, false, idx);
        result.setTo(Scalar(0, 0, 255), mask);
        imshow(window_name, result);
        int c = waitKey(2500);
    }
}