#ifdef __cplusplus
#ifndef __3D_SEEDS_HPP__
#define __3D_SEEDS_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{


class CV_EXPORTS_W SupervoxelSEEDS : public Algorithm
{
public:

    CV_WRAP virtual int getNumberOfSuperpixels() = 0;
    CV_WRAP virtual void iterate(InputArray img, int num_iterations=4) = 0;
    CV_WRAP virtual void getLabels(OutputArray labels_out) = 0;
    CV_WRAP virtual void getLabelContourMask(OutputArray image, bool thick_line = false, int idx=0) = 0;
    virtual ~SupervoxelSEEDS() {}
};

CV_EXPORTS_W Ptr<SupervoxelSEEDS> createSupervoxelSEEDS(
    int image_width, int image_height, int image_depth, int image_channels,
    int num_superpixels, int num_levels, int prior = 2,
    int histogram_bins=5, bool double_step = false);

}
}

#endif
#endif

