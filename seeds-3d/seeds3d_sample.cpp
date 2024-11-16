
#include "seeds3d.hpp"
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

    int depth = 48;
    int height = 192;
    int width = 192;
    std::ifstream file("/Users/Zach/Zch/Research/seeds-3d/seeds-3d/array_3d.bin", std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }
    std::vector<float> data(depth * height * width);
    file.read(reinterpret_cast<char *>(data.data()), depth * height * width * sizeof(float));
    file.close();
    Mat img_data = Mat(3, (int[]){depth, height, width}, CV_32FC1, data.data());
    Mat frame(height, width, CV_8UC1);
    int idx = 30;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            frame.at<uchar>(i, j) = static_cast<uchar>(data[idx * width * height + i * height + j]);
        }
    }

    namedWindow(window_name, 0);
    int num_iterations = 4;
    int prior = 2;
    bool double_step = false;
    int num_superpixels = 400;
    int num_levels = 4;
    int num_histogram_bins = 5;
    createTrackbar("Number of Superpixels", window_name, &num_superpixels, 1000, trackbarChanged);
    createTrackbar("Smoothing Prior", window_name, &prior, 5, trackbarChanged);
    createTrackbar("Number of Levels", window_name, &num_levels, 10, trackbarChanged);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Ptr<SuperpixelSEEDS3D> seeds;
    Mat mask;
    int display_mode = 0;

    for (;;)
    {
        Mat result = frame.clone();
        if (!init)
        {
            seeds = createSuperpixelSEEDS3D(width, height, depth, 1, num_superpixels,
                                            num_levels, prior, num_histogram_bins, double_step);
            init = true;
        }

        double t = (double)getTickCount();

        Mat input = img_data / 255.0f;
        seeds->iterate(input, num_iterations);

        t = ((double)getTickCount() - t) / getTickFrequency();
        printf("SEEDS segmentation took %i ms with %3i superpixels\n",
               (int)(t * 1000), seeds->getNumberOfSuperpixels());

        // /* retrieve the segmentation result */
        Mat labels;
        seeds->getLabels(labels);

        // // /* get the contours for displaying */
        seeds->getLabelContourMask(mask, false, idx);
        result.setTo(Scalar(0, 0, 255), mask);

        // // /* display output */
        switch (display_mode)
        {
        case 0: // superpixel contours
            imshow(window_name, result);
            break;
        case 1: // mask
            imshow(window_name, mask);
            break;
            // case 2: //labels array
            // {
            //     // use the last x bit to determine the color. Note that this does not
            //     // guarantee that 2 neighboring superpixels have different colors.
            //     const int num_label_bits = 2;
            //     labels &= (1 << num_label_bits) - 1;
            //     labels *= 1 << (16 - num_label_bits);
            //     imshow(window_name, labels);
            // }
            break;
        }

        int c = waitKey(1000);
        if ((c & 255) == 'q' || c == 'Q' || (c & 255) == 27)
            break;
        else if ((c & 255) == ' ')
            display_mode = (display_mode + 1) % 3;
    }
}