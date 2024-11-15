#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
    std::string image_path = "/Users/Zach/Zch/Research/seeds-3d/00.jpg";
    Mat img = imread(image_path, IMREAD_COLOR);

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}