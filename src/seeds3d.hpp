#ifdef __cplusplus
#ifndef __SEEDS3D_HPP__
#define __SEEDS3D_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{


class CV_EXPORTS_W SuperpixelSEEDS3D : public Algorithm
{
public:

    CV_WRAP virtual int getNumberOfSuperpixels() = 0;
    CV_WRAP virtual void iterate(InputArray img, int num_iterations=4) = 0;
    CV_WRAP virtual void getLabels(OutputArray labels_out) = 0;
    CV_WRAP virtual void getLabelContourMask(OutputArray image, bool thick_line = false, int idx=0) = 0;
    virtual ~SuperpixelSEEDS3D() {}
};

CV_EXPORTS_W Ptr<SuperpixelSEEDS3D> createSuperpixelSEEDS3D(
    int image_width, int image_height, int image_depth, int image_channels,
    int num_superpixels, int num_levels, int prior = 2,
    int histogram_bins=5, bool double_step = false);

}
}

#endif
#endif

