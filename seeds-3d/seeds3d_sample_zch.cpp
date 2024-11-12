
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

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

int main()
{
    int height = 192;
    int width = 192;
    int depth = 48;

    Mat img_data = Mat(3, (int[]){height, width, depth}, CV_32SC1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(INT_MIN, INT_MAX);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < depth; k++) {
                img_data.at<int>(i, j, k) = dis(gen)%255;  
            }
        }
    }


   
    Ptr<SuperpixelSEEDS3D> seeds;
    seeds = createSuperpixelSEEDS3D(192, 192, 48, 1, 125, 3, 5, 10, 0);
    double t = (double)getTickCount();
    seeds->iterate(img_data, 20);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "seeds3d took " << t << " ms";
    
    Mat labels_result;
    seeds->getLabels(labels_result);
    
    int* labels = (int*)labels_result.data;
    for (int i=0; i<192*192*48; i++)
    {
        cout<<labels[i];
        if (i%20==0) cout<<endl;
    }

}