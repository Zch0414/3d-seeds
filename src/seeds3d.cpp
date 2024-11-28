/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/******************************************************************************\
*                            SEEDS3D Superpixels                              *
*  This code implements a 3D version of superpixel method described in:       *
*  M. Van den Bergh, X. Boix, G. Roig, B. de Capitani and L. Van Gool,        *
*  "SEEDS: Superpixels Extracted via Energy-Driven Sampling", ECCV 2012       *
\******************************************************************************/

#include "seeds3d.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
using namespace std;



//required confidence when double_step is used
#define REQ_CONF 0.1f
#define MINIMUM_NR_SUBLABELS 1
#define CV_MALLOC_ALIGN 16


// the type of the histogram and the T array
typedef float HISTN;


namespace cv {
namespace ximgproc {

class SuperpixelSEEDS3DImpl : public SuperpixelSEEDS3D
{
public:

    SuperpixelSEEDS3DImpl(int image_width, int image_height, int image_depth, int image_channels,
            int num_superpixels, int num_levels,  int prior = 2,
           int histogram_bins = 5,  bool double_step = false);

    ~SuperpixelSEEDS3DImpl();
    int getNumberOfSuperpixels() { return nrLabels(seeds_top_level); }
    void iterate(InputArray img, int num_iterations = 4);
    void getLabels(OutputArray labels_out);
    void getLabelContourMask(OutputArray image, bool thick_line = false, int idx=0);

private:
    /* initialization */
    void initialize(int num_superpixels, int num_levels);
    void initImage(InputArray img);
    void assignLabels();
    void computeHistograms(int until_level = -1);
    template<typename _Tp>
    inline void initImageBins(const Mat& img, int max_value);

    /* pixel operations */
    inline void update(int label_new, int image_idx, int label_old); //image_idx = z*width*hight+y*width+x
    inline void addPixel(int level, int label, int image_idx);
    inline void deletePixel(int level, int label, int image_idx);
    inline bool probability(int image_idx, int label1, int label2, int prior1, int prior2);
    // inline int threebyfour(int x, int y, int label);
    // inline int fourbythree(int x, int y, int label);
    inline int threebythreebyfour(int x, int y, int z, int label);
    inline int threebyfourbythree(int x, int y, int z, int label);
    inline int fourbythreebythree(int x, int y, int z, int label);
    inline void updateLabels();
    void updatePixels(); // main loop for pixel updating

    /* block operations */
    void addBlock(int level, int label, int sublevel, int sublabel);
    inline void addBlockToplevel(int label, int sublevel, int sublabel);
    void deleteBlockToplevel(int label, int sublevel, int sublabel);

    // intersection on label1A and intersection_delete on label1B
    // returns intA - intB
    float intersectConf(int level1, int label1A, int label1B, int level2, int label2);
    //main loop for block updates
    void updateBlocks(int level, float req_confidence = 0.0f);

    /* go to next block level */
    int goDownOneLevel();

    //make sure a superpixel stays connected (h=horizontal,v=vertical, f=forward,b=backward) // TODO
    // inline bool checkSplit_hf(int a11, int a12, int a21, int a22, int a31, int a32);
    // inline bool checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33);
    // inline bool checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23);
    // inline bool checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33);
    inline bool checkSplit_3d(
        int a111, int a112, int a121, int a122, int a131, int a132,
        int a211, int a212, int a221, int a222, int a231, int a232,
        int a311, int a312, int a321, int a322, int a331, int a332
    );

    //compute initial label for sublevels: level <= seeds_top_level
    //this is an equally sized grid with size nr_h[level]*nr_w[level]
    int computeLabel(int level, int x, int y, int z) {
        return std::min(z * nr_whd[3 * level + 2] / depth, nr_whd[3 * level + 2] - 1) * nr_whd[3 * level] * nr_whd[3 * level + 1] 
            +std::min(y * nr_whd[3 * level + 1] / height, nr_whd[3 * level + 1] - 1) * nr_whd[3 * level]
        + std::min(x * nr_whd[3 * level] / width, nr_whd[3 * level] - 1);
    }
    inline int nrLabels(int level) const {
        return nr_whd[3 * level] * nr_whd[3 * level + 1] * nr_whd[3 * level + 2];
    }

    int width, height, depth; //image size
    int nr_bins; //number of histogram bins per channel
    int nr_channels; //number of image channels
    bool forwardbackward;

    int seeds_nr_levels;
    int seeds_top_level; // == seeds_nr_levels-1 (const)
    int seeds_current_level; //start with level seeds_top_level-1, then go down
    bool seeds_double_step;
    int seeds_prior;

    // keep one labeling for each level
    vector<int> nr_whd; // [3*level]/[3*level+1]/[3*level+2] number of labels in x-direction/y-direction/z-direction

    /* pre-initialized arrays. they are not modified afterwards */
    int* labels_bottom; //labels of level==0
    vector<int*> parent_pre_init;

    unsigned int* image_bins; //[y*width + x] bin index (histogram) of each image pixel

    vector<int*> parent; //[level][label] = corresponding label of block with level+1
    int* labels; //output labels: labels of level==seeds_top_level
    unsigned int* nr_partitions; //[label] how many partitions label has on toplevel

    int histogram_size; //== pow(nr_bins, nr_channels)
    int histogram_size_aligned;
    vector<HISTN*> histogram; //[level][label * histogram_size_aligned + j]
    vector<HISTN*> T; //[level][label] how many pixels with this label

    /* OpenCV containers for our memory arrays. This makes sure memory is
     * allocated & released properly */
    Mat labels_mat;
    Mat labels_bottom_mat;
    Mat nr_partitions_mat;
    Mat image_bins_mat;
    vector<Mat> histogram_mat;
    vector<Mat> T_mat;
    vector<Mat> parent_mat;
    vector<Mat> parent_pre_init_mat;
};

CV_EXPORTS Ptr<SuperpixelSEEDS3D> createSuperpixelSEEDS3D(int image_width, int image_height, int image_depth,
        int image_channels, int num_superpixels, int num_levels, int prior, int histogram_bins,
        bool double_step)
{
    return makePtr<SuperpixelSEEDS3DImpl>(image_width, image_height, image_depth, image_channels,
            num_superpixels, num_levels, prior, histogram_bins, double_step);
}

SuperpixelSEEDS3DImpl::SuperpixelSEEDS3DImpl(int image_width, int image_height, int image_depth, int image_channels,
            int num_superpixels, int num_levels, int prior, int histogram_bins, bool double_step)
{
    width = image_width;
    height = image_height;
    depth = image_depth;
    nr_bins = histogram_bins;
    nr_channels = image_channels;
    seeds_double_step = double_step;
    seeds_prior = std::min(prior, 5);

    histogram_size = nr_bins;
    for (int i = 1; i < nr_channels; ++i)
        histogram_size *= nr_bins;
    histogram_size_aligned = (histogram_size
        + ((CV_MALLOC_ALIGN / sizeof(HISTN)) - 1)) & -static_cast<int>(CV_MALLOC_ALIGN / sizeof(HISTN));

    initialize(num_superpixels, num_levels);
}

SuperpixelSEEDS3DImpl::~SuperpixelSEEDS3DImpl()
{
}


void SuperpixelSEEDS3DImpl::iterate(InputArray img, int num_iterations)
{
    initImage(img);

    // block updates
    while (seeds_current_level >= 0)
    {
        if( seeds_double_step )
            updateBlocks(seeds_current_level, REQ_CONF);

        updateBlocks(seeds_current_level);
        seeds_current_level = goDownOneLevel();
    }
    updateLabels();

    for (int i = 0; i < num_iterations; ++i)
        updatePixels();
}
void SuperpixelSEEDS3DImpl::getLabels(OutputArray labels_out)
{
    labels_out.assign(labels_mat);
}

void SuperpixelSEEDS3DImpl::initialize(int num_superpixels, int num_levels)
{
    int sizes[3] = {depth, height, width};

    /* enforce parameter restrictions */
    if( num_superpixels < 10 )
        num_superpixels = 10;
    if( num_levels < 2 )
        num_levels = 2;
    int num_superpixels_h = (int)cbrtf((float)num_superpixels * (height / width) * (height / depth));
    int num_superpixels_w = num_superpixels_h * width / height;
    int num_superpixels_d = num_superpixels_h * depth / height;
    seeds_nr_levels = num_levels + 1;
    float seeds_wf, seeds_hf, seeds_df;
    do
    {
        --seeds_nr_levels;
        seeds_wf = (float)width / num_superpixels_w / (1<<(seeds_nr_levels-1));
        seeds_hf = (float)height / num_superpixels_h / (1<<(seeds_nr_levels-1));
        seeds_df = (float)depth / num_superpixels_d / (1<<(seeds_nr_levels-1));
    } while( seeds_wf < 1.f || seeds_hf < 1.f || seeds_df < 1.f);
    int seeds_w = (int)ceil(seeds_wf);
    int seeds_h = (int)ceil(seeds_hf);
    int seeds_d = (int)ceil(seeds_df);
    CV_Assert(seeds_nr_levels > 0);

    seeds_top_level = seeds_nr_levels - 1;
    image_bins_mat = Mat(3, sizes, CV_32SC1);
    image_bins = (unsigned int*)image_bins_mat.data;

    // init labels
    labels_mat = Mat(3, sizes, CV_32SC1);
    labels = (int*)labels_mat.data;
    labels_bottom_mat = Mat(3, sizes, CV_32SC1);
    labels_bottom = (int*)labels_bottom_mat.data;
    parent.resize(seeds_nr_levels);
    parent_pre_init.resize(seeds_nr_levels);
    nr_whd.resize(3 * seeds_nr_levels);
    int level = 0;
    int nr_seeds_w = (width / seeds_w);
    int nr_seeds_h = (height / seeds_h);
    int nr_seeds_d = (depth / seeds_d);
    nr_whd[3 * level] = nr_seeds_w;
    nr_whd[3 * level + 1] = nr_seeds_h;
    nr_whd[3 * level + 2] = nr_seeds_d;
    parent_mat.push_back(Mat(3, sizes, CV_32SC1));
    parent[level] = (int*)parent_mat.back().data;
    parent_pre_init_mat.push_back(Mat(3, sizes, CV_32SC1));
    parent_pre_init[level] = (int*)parent_pre_init_mat.back().data;
    for (level = 1; level < seeds_nr_levels; level++)
    {
        nr_seeds_w /= 2; // always partitioned in 2x2x2 sub-blocks
        nr_seeds_h /= 2;
        nr_seeds_d /= 2;
        int seeds_sizes[3] = {nr_seeds_d, nr_seeds_h, nr_seeds_w};
        parent_mat.push_back(Mat(3, seeds_sizes, CV_32SC1));
        parent[level] = (int*)parent_mat.back().data;
        parent_pre_init_mat.push_back(Mat(3, seeds_sizes, CV_32SC1));
        parent_pre_init[level] = (int*)parent_pre_init_mat.back().data;
        nr_whd[3 * level] = nr_seeds_w;
        nr_whd[3 * level + 1] = nr_seeds_h;
        nr_whd[3 * level + 2] = nr_seeds_d;

        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    parent_pre_init[level - 1][computeLabel(level - 1, x, y, z)] =
                            computeLabel(level, x, y, z); // set parent
                }
            }
        }    
    }
    int partitions_mat_sizes[3] = {nr_whd[3 * seeds_top_level + 2], nr_whd[3 * seeds_top_level + 1], nr_whd[3 * seeds_top_level]};
    nr_partitions_mat = Mat(3, partitions_mat_sizes, CV_32SC1);
    nr_partitions = (unsigned int*)nr_partitions_mat.data;

    //preinit the labels (these are not changed anymore later)
    int i = 0;
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                labels_bottom[i] = computeLabel(0, x, y, z);
                ++i;
            }
        }
    }

    // create histogram buffers
    histogram.resize(seeds_nr_levels);
    T.resize(seeds_nr_levels);
    histogram_mat.resize(seeds_nr_levels);
    T_mat.resize(seeds_nr_levels);
    for (level = 0; level < seeds_nr_levels; level++)
    {   
        int histogram_mat_sizes[3] = {nr_whd[3 * level + 2], nr_whd[3 * level + 1], nr_whd[3 * level]*histogram_size_aligned};
        histogram_mat[level] = Mat(3, histogram_mat_sizes, CV_32FC1);
        histogram[level] = (HISTN*)histogram_mat[level].data;
        int T_mat_sizes[3] = {nr_whd[3 * level + 2], nr_whd[3 * level + 1], nr_whd[3 * level]};
        T_mat[level] = Mat(3, T_mat_sizes, CV_32FC1);
        T[level] = (HISTN*)T_mat[level].data;
    }
}


template<typename _Tp>
void SuperpixelSEEDS3DImpl::initImageBins(const Mat& img, int max_value)
{
    int img_width = img.size[2];
    int img_height = img.size[1];
    int img_depth = img.size[0];
    int channels = img.channels();

    for (int z = 0; z < img_depth; ++z)
    {        
        for (int y = 0; y < img_height; ++y)
        {
            for (int x = 0; x < img_width; ++x)
            {
                const _Tp* ptr = img.ptr<_Tp>(z, y, x);
                int bin = 0;
                for (int i = 0; i < channels; ++i)
                {
                    bin = bin * nr_bins + (int) ptr[i] * nr_bins / max_value;
                }
                image_bins[z * img_width * img_height + y * img_width + x] = bin;
            }
        }
    }
}

/* specialization for float: max_value is assumed to be 1.0f */
template<>
void SuperpixelSEEDS3DImpl::initImageBins<float>(const Mat& img, int)
{
    int img_width = img.size[2];
    int img_height = img.size[1];
    int img_depth = img.size[0];
    int channels = img.channels();

    for (int z = 0; z < img_depth; ++z)
    {
        for (int y = 0; y < img_height; ++y)
        {
            for (int x = 0; x < img_width; ++x)
            {
                const float* ptr = img.ptr<float>(z, y, x);
                int bin = 0;
                for(int i=0; i<channels; ++i)
                {   
                    bin = bin * nr_bins + std::min((int)(ptr[i] * (float)nr_bins), nr_bins-1);
                }
                image_bins[z * img_width * img_height + y * img_width + x] = bin;
            }
        }
    }
}

void SuperpixelSEEDS3DImpl::initImage(InputArray img)
{
    Mat src;

    if ( img.isMat() )
    {
      // get Mat
      src = img.getMat();

      // image should be valid
      CV_Assert( !src.empty() );
    }
    else if ( img.isMatVector() )
    {
      vector<Mat> vec;
      // get vector Mat
      img.getMatVector( vec );

      // array should be valid
      CV_Assert( !vec.empty() );

      // merge into Mat
      merge( vec, src );
    }
    else
      CV_Error( Error::StsInternal, "Invalid InputArray." );

    int _depth = src.depth(); // differentiate from the depth dimension for image
    seeds_current_level = seeds_nr_levels - 2;
    forwardbackward = true;

    assignLabels();

    CV_Assert(src.size[2] == width && src.size[1] == height && src.size[0] == depth);
    CV_Assert(_depth == CV_8U || _depth == CV_16U || _depth == CV_32F);
    CV_Assert(src.channels() == nr_channels);

    // initialize the histogram bins from the image
    switch (_depth)
    {
    case CV_8U:
        initImageBins<uchar>(src, 1 << 8);
        break;
    case CV_16U:
        initImageBins<ushort>(src, 1 << 16);
        break;
    case CV_32F:
        initImageBins<float>(src, 1);
        break;
    }

    computeHistograms();
}

// adds labeling to all the blocks at all levels and sets the correct parents
void SuperpixelSEEDS3DImpl::assignLabels()
{
    /* each top level label is partitioned into 8 elements */
    int nr_labels_toplevel = nrLabels(seeds_top_level);
    for (int i = 0; i < nr_labels_toplevel; ++i)
        nr_partitions[i] = 8;

    for (int level = 1; level < seeds_nr_levels; level++)
    {
        memcpy(parent[level - 1], parent_pre_init[level - 1],
                sizeof(int) * nrLabels(level - 1));
    }
}

void SuperpixelSEEDS3DImpl::computeHistograms(int until_level)
{
    if( until_level == -1 )
        until_level = seeds_nr_levels - 1;
    until_level++;

    // clear histograms
    for (int level = 0; level < seeds_nr_levels; level++)
    {
        int nr_labels = nrLabels(level);
        memset(histogram[level], 0,
                sizeof(HISTN) * histogram_size_aligned * nr_labels);
        memset(T[level], 0, sizeof(HISTN) * nr_labels);
    }

    // build histograms on the first level by adding the pixels to the blocks
    for (int i = 0; i < width * height * depth; ++i)
        addPixel(0, labels_bottom[i], i);

    // build histograms on the upper levels by adding the histogram from the level below
    for (int level = 1; level < until_level; level++)
    {
        for (int label = 0; label < nrLabels(level - 1); label++)
        {
            addBlock(level, parent[level - 1][label], level - 1, label);
        }
    }
}

void SuperpixelSEEDS3DImpl::updateBlocks(int level, float req_confidence)
{
    int labelA;
    int labelB;
    int sublabel;
    bool done;
    int stepY = nr_whd[3 * level];
    int stepZ = nr_whd[3 * level] * nr_whd[3 * level + 1];

    // horizontal bidirectional block updating
    for (int z = 1; z < nr_whd[3 * level + 2] - 1; z ++)
    {
        for (int y = 1; y < nr_whd[3 * level + 1] - 1; y++)
        {
            for (int x = 1; x < nr_whd[3 * level] - 2; x++)
            {
                // choose a label at the current level
                sublabel = z * stepZ + y * stepY + x;
                // get the label at the top level (= superpixel label)
                labelA = parent[level][z * stepZ + y * stepY + x];
                // get the neighboring label at the top level (= superpixel label)
                labelB = parent[level][z * stepZ + y * stepY + x + 1];

                if( labelA == labelB )
                    continue;

                // get the surrounding labels at the top level, to check for splitting
                int a111 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x - 1)];
                int a112 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x)];
                int a121 = parent[level][(z - 1) * stepZ + (y) * stepY + (x - 1)];
                int a122 = parent[level][(z - 1) * stepZ + (y) * stepY + (x)];
                int a131 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x - 1)];
                int a132 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x)];

                int a211 = parent[level][(z) * stepZ + (y - 1) * stepY + (x - 1)];
                int a212 = parent[level][(z) * stepZ + (y - 1) * stepY + (x)];
                int a221 = parent[level][(z) * stepZ + (y) * stepY + (x - 1)];
                int a222 = parent[level][(z) * stepZ + (y) * stepY + (x)];
                int a231 = parent[level][(z) * stepZ + (y + 1) * stepY + (x - 1)];
                int a232 = parent[level][(z) * stepZ + (y + 1) * stepY + (x)];

                int a311 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x - 1)];
                int a312 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x)];
                int a321 = parent[level][(z + 1) * stepZ + (y) * stepY + (x - 1)];
                int a322 = parent[level][(z + 1) * stepZ + (y) * stepY + (x)];
                int a331 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x - 1)];
                int a332 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x)];
                done = false;

                if( 
                    nr_partitions[labelA] == 2 
                    || (
                        nr_partitions[labelA] > 2
                        && checkSplit_3d(
                            a111, a112, a121, a122, a131, a132, 
                            a211, a212, a221, a222, a231, a232,
                            a311, a312, a321, a322, a331, a332 
                        )
                    ) 
                )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelA, level, sublabel);
                        addBlockToplevel(labelB, level, sublabel);
                        done = true;
                    }
                }

                if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
                {
                    // try opposite direction
                    sublabel = z * stepZ + y * stepY + x + 1;
                    int a113 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x + 1)];
                    int a114 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x + 2)];
                    int a123 = parent[level][(z - 1) * stepZ + (y) * stepY + (x + 1)];
                    int a124 = parent[level][(z - 1) * stepZ + (y) * stepY + (x + 2)];
                    int a133 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a134 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x + 2)];

                    int a213 = parent[level][(z) * stepZ + (y - 1) * stepY + (x + 1)];
                    int a214 = parent[level][(z) * stepZ + (y - 1) * stepY + (x + 2)];
                    int a223 = parent[level][(z) * stepZ + (y) * stepY + (x + 1)];
                    int a224 = parent[level][(z) * stepZ + (y) * stepY + (x + 2)];
                    int a233 = parent[level][(z) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a234 = parent[level][(z) * stepZ + (y + 1) * stepY + (x + 2)];

                    int a313 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x + 1)];
                    int a314 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x + 2)];
                    int a323 = parent[level][(z + 1) * stepZ + (y) * stepY + (x + 1)];
                    int a324 = parent[level][(z + 1) * stepZ + (y) * stepY + (x + 2)];
                    int a333 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a334 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x + 2)];
                    if( 
                        nr_partitions[labelB] <= 2 // == 2
                        || (
                            nr_partitions[labelB] > 2     
                            && checkSplit_3d(
                                a134, a133, a124, a123, a114, a113,
                                a234, a233, a224, a223, a214, a213,
                                a334, a333, a324, a323, a314, a313
                            ) // counterclockwise rotate pi along z
                        ) 
                    )
                    {
                        // run algorithm as usual
                        float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                        if( conf > req_confidence )
                        {
                            deleteBlockToplevel(labelB, level, sublabel);
                            addBlockToplevel(labelA, level, sublabel);
                            x++;
                        }
                    }
                }
            }
        }
    }

    // vertical bidirectional
    for (int z = 1; z < nr_whd[3 * level + 2] - 1; z++)
    {
        for (int x = 1; x < nr_whd[3 * level] - 1; x++)
        {
            for (int y = 1; y < nr_whd[3 * level + 1] - 2; y++)
            {
                // choose a label at the current level
                sublabel = z * stepZ + y * stepY + x;
                // get the label at the top level (= superpixel label)
                labelA = parent[level][z * stepZ + y * stepY + x];
                // get the neighboring label at the top level (= superpixel label)
                labelB = parent[level][z * stepZ + (y + 1) * stepY + x];

                if( labelA == labelB )
                    continue;

                int a111 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x - 1)];
                int a112 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x)];
                int a113 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x + 1)];
                int a121 = parent[level][(z - 1) * stepZ + (y) * stepY + (x - 1)];
                int a122 = parent[level][(z - 1) * stepZ + (y) * stepY + (x)];
                int a123 = parent[level][(z - 1) * stepZ + (y) * stepY + (x + 1)];

                int a211 = parent[level][(z) * stepZ + (y - 1) * stepY + (x - 1)];
                int a212 = parent[level][(z) * stepZ + (y - 1) * stepY + (x)];
                int a213 = parent[level][(z) * stepZ + (y - 1) * stepY + (x + 1)];
                int a221 = parent[level][(z) * stepZ + (y) * stepY + (x - 1)];
                int a222 = parent[level][(z) * stepZ + (y) * stepY + (x)];
                int a223 = parent[level][(z) * stepZ + (y) * stepY + (x + 1)];

                int a311 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x - 1)];
                int a312 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x)];
                int a313 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x + 1)];
                int a321 = parent[level][(z + 1) * stepZ + (y) * stepY + (x - 1)];
                int a322 = parent[level][(z + 1) * stepZ + (y) * stepY + (x)];
                int a323 = parent[level][(z + 1) * stepZ + (y) * stepY + (x + 1)];
                done = false;
                
                if( 
                    nr_partitions[labelA] == 2 
                    || (nr_partitions[labelA] > 2
                        && checkSplit_3d(
                            a113, a123, a112, a122, a111, a121,
                            a213, a223, a212, a222, a211, a221,
                            a313, a323, a312, a322, a311, a321
                        ) // counterclockwise rotate pi/2 along z
                    ) 
                )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelA, level, sublabel);
                        addBlockToplevel(labelB, level, sublabel);
                        done = true;
                    }
                }

                if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
                {
                    // try opposite direction
                    sublabel = z * stepZ + (y + 1) * stepY + x;
                    int a131 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x - 1)];
                    int a132 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x)];
                    int a133 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a141 = parent[level][(z - 1) * stepZ + (y + 2) * stepY + (x - 1)];
                    int a142 = parent[level][(z - 1) * stepZ + (y + 2) * stepY + (x)];
                    int a143 = parent[level][(z - 1) * stepZ + (y + 2) * stepY + (x + 1)];

                    int a231 = parent[level][(z) * stepZ + (y + 1) * stepY + (x - 1)];
                    int a232 = parent[level][(z) * stepZ + (y + 1) * stepY + (x)];
                    int a233 = parent[level][(z) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a241 = parent[level][(z) * stepZ + (y + 2) * stepY + (x - 1)];
                    int a242 = parent[level][(z) * stepZ + (y + 2) * stepY + (x)];
                    int a243 = parent[level][(z) * stepZ + (y + 2) * stepY + (x + 1)];

                    int a331 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x - 1)];
                    int a332 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x)];
                    int a333 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x + 1)];
                    int a341 = parent[level][(z + 1) * stepZ + (y + 2) * stepY + (x - 1)];
                    int a342 = parent[level][(z + 1) * stepZ + (y + 2) * stepY + (x)];
                    int a343 = parent[level][(z + 1) * stepZ + (y + 2) * stepY + (x + 1)];
                    if( 
                        nr_partitions[labelB] <= 2 // == 2
                        || (
                            nr_partitions[labelB] > 2 
                            && checkSplit_3d(
                                a141, a131, a142, a132, a143, a133,
                                a241, a231, a242, a232, a243, a233,
                                a341, a331, a342, a332, a343, a333
                            ) // clockwise rotate pi/2 along z
                        ) 
                    )
                    {
                        // run algorithm as usual
                        float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                        if( conf > req_confidence )
                        {
                            deleteBlockToplevel(labelB, level, sublabel);
                            addBlockToplevel(labelA, level, sublabel);
                            y++;
                        }
                    }
                }
            }
        }
    }

    // depth bidirectional
    for (int y = 1; y < nr_whd[3 * level + 1] - 1; y++)
    {
        for (int x = 1; x < nr_whd[3 * level] - 1; x++)
        {
            for (int z = 1; z < nr_whd[3 * level + 2] - 2; z++)
            {
                // choose a label at the current level
                sublabel = z * stepZ + y * stepY + x;
                // get the label at the top level (= superpixel label)
                labelA = parent[level][z * stepZ + y * stepY + x];
                // get the neighboring label at the top level (= superpixel label)
                labelB = parent[level][(z + 1) * stepZ + y * stepY + x];

                if( labelA == labelB )
                    continue;

                int a111 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x - 1)];
                int a112 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x)];
                int a113 = parent[level][(z - 1) * stepZ + (y - 1) * stepY + (x + 1)];
                int a121 = parent[level][(z - 1) * stepZ + (y) * stepY + (x - 1)];
                int a122 = parent[level][(z - 1) * stepZ + (y) * stepY + (x)];
                int a123 = parent[level][(z - 1) * stepZ + (y) * stepY + (x + 1)];
                int a131 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x - 1)];
                int a132 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x)];
                int a133 = parent[level][(z - 1) * stepZ + (y + 1) * stepY + (x + 1)];

                int a211 = parent[level][(z) * stepZ + (y - 1) * stepY + (x - 1)];
                int a212 = parent[level][(z) * stepZ + (y - 1) * stepY + (x)];
                int a213 = parent[level][(z) * stepZ + (y - 1) * stepY + (x + 1)];
                int a221 = parent[level][(z) * stepZ + (y) * stepY + (x - 1)];
                int a222 = parent[level][(z) * stepZ + (y) * stepY + (x)];
                int a223 = parent[level][(z) * stepZ + (y) * stepY + (x + 1)];
                int a231 = parent[level][(z) * stepZ + (y + 1) * stepY + (x - 1)];
                int a232 = parent[level][(z) * stepZ + (y + 1) * stepY + (x)];
                int a233 = parent[level][(z) * stepZ + (y + 1) * stepY + (x + 1)];
                done = false;

                if( 
                    nr_partitions[labelA] == 2 
                    || (nr_partitions[labelA] > 2
                        && checkSplit_3d(
                            a113, a213, a123, a223, a133, a233,
                            a112, a212, a122, a222, a132, a232,
                            a111, a211, a121, a221, a131, a231
                        ) // conterclockwise rotate pi/2 along y
                    ) 
                )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelA, level, sublabel);
                        addBlockToplevel(labelB, level, sublabel);
                        done = true;
                    }
                }

                if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
                {
                    // try opposite direction
                    sublabel = (z + 1) * stepZ + y * stepY + x;
                    int a311 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x - 1)];
                    int a312 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x)];
                    int a313 = parent[level][(z + 1) * stepZ + (y - 1) * stepY + (x + 1)];
                    int a321 = parent[level][(z + 1) * stepZ + (y) * stepY + (x - 1)];
                    int a322 = parent[level][(z + 1) * stepZ + (y) * stepY + (x)];
                    int a323 = parent[level][(z + 1) * stepZ + (y) * stepY + (x + 1)];
                    int a331 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x - 1)];
                    int a332 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x)];
                    int a333 = parent[level][(z + 1) * stepZ + (y + 1) * stepY + (x + 1)];

                    int a411 = parent[level][(z + 2) * stepZ + (y - 1) * stepY + (x - 1)];
                    int a412 = parent[level][(z + 2) * stepZ + (y - 1) * stepY + (x)];
                    int a413 = parent[level][(z + 2) * stepZ + (y - 1) * stepY + (x + 1)];
                    int a421 = parent[level][(z + 2) * stepZ + (y) * stepY + (x - 1)];
                    int a422 = parent[level][(z + 2) * stepZ + (y) * stepY + (x)];
                    int a423 = parent[level][(z + 2) * stepZ + (y) * stepY + (x + 1)];
                    int a431 = parent[level][(z + 2) * stepZ + (y + 1) * stepY + (x - 1)];
                    int a432 = parent[level][(z + 2) * stepZ + (y + 1) * stepY + (x)];
                    int a433 = parent[level][(z + 2) * stepZ + (y + 1) * stepY + (x + 1)];
                    if( 
                        nr_partitions[labelB] <= 2 // == 2
                        || (
                            nr_partitions[labelB] > 2 
                            && checkSplit_3d(
                                a411, a311, a421, a321, a431, a331, 
                                a412, a312, a422, a322, a432, a332, 
                                a413, a313, a423, a323, a433, a333 
                            ) // clockwise rotate pi/2 along y
                        ) 
                    )
                    {
                        // run algorithm as usual
                        float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                        if( conf > req_confidence )
                        {
                            deleteBlockToplevel(labelB, level, sublabel);
                            addBlockToplevel(labelA, level, sublabel);
                            z++;
                        }
                    }
                }
            }
        }
    }
}

int SuperpixelSEEDS3DImpl::goDownOneLevel()
{
    int old_level = seeds_current_level;
    int alevel = seeds_current_level - 1;

    if( alevel < 0 )
        return -1;

    // reset nr_partitions
    memset(nr_partitions, 0, sizeof(int) * nrLabels(seeds_top_level));

    // go through labels of alevel
    int labels_alevel = nrLabels(alevel);
    //the lowest level (0) has 1 partition, all higher levels are
    //initially partitioned into 4
    int partitions = alevel ? 8 : 1;

    for (int label = 0; label < labels_alevel; ++label)
    {
        // assign parent = parent of old_label
        int& cur_parent = parent[alevel][label];
        int p = parent[old_level][cur_parent];
        cur_parent = p;

        nr_partitions[p] += partitions;
    }

    return alevel;
}

void SuperpixelSEEDS3DImpl::updatePixels()
{
    int labelA;
    int labelB;
    int priorA = 0;
    int priorB = 0;

    for (int z = 1; z < depth - 1; z++)
    {
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 2; x++)
            {

                labelA = labels[(z) * width * height + (y) * width + (x)]; // a222
                labelB = labels[(z) * width * height + (y) * width + (x + 1)]; // a223

                if( labelA != labelB )
                {
                    int a222 = labelA;
                    int a223 = labelB;
                    if( forwardbackward )
                    {
                        // horizontal bidirectional
                        int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                        int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                        int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                        int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                        int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                        int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];

                        int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                        int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                        int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                        // int a222 = labels[(z) * width * height + (y) * width + (x)];
                        int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                        int a232 = labels[(z) * width * height + (y + 1) * width + (x)];

                        int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                        int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                        int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                        int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                        int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                        int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                        if( 
                            checkSplit_3d(
                                a111, a112, a121, a122, a131, a132, 
                                a211, a212, a221, a222, a231, a232,
                                a311, a312, a321, a322, a331, a332 
                            ) 
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = threebythreebyfour(x, y, z, labelA);
                                priorB = threebythreebyfour(x, y, z, labelB);
                            }

                            if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                            {
                                update(labelB, z*width * height + y * width + x, labelA);
                            }
                            else
                            {
                                int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                                int a114 = labels[(z - 1) * width * height + (y - 1) * width + (x + 2)];
                                int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];
                                int a124 = labels[(z - 1) * width * height + (y) * width + (x + 2)];
                                int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];
                                int a134 = labels[(z - 1) * width * height + (y + 1) * width + (x + 2)];

                                int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                                int a214 = labels[(z) * width * height + (y - 1) * width + (x + 2)];
                                // int a223 = labels[(z) * width * height + (y) * width + (x + 1)];
                                int a224 = labels[(z) * width * height + (y) * width + (x + 2)];
                                int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                                int a234 = labels[(z) * width * height + (y + 1) * width + (x + 2)];

                                int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                                int a314 = labels[(z + 1) * width * height + (y - 1) * width + (x + 2)];
                                int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                                int a324 = labels[(z + 1) * width * height + (y) * width + (x + 2)];
                                int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];
                                int a334 = labels[(z + 1) * width * height + (y + 1) * width + (x + 2)];
                                if( 
                                    checkSplit_3d(
                                        a134, a133, a124, a123, a114, a113,
                                        a234, a233, a224, a223, a214, a213,
                                        a334, a333, a324, a323, a314, a313
                                    ) // counterclockwise rotate pi along z
                                )
                                {
                                    if( probability(z * width * height + y * width + x + 1, labelB, labelA, priorB, priorA) )
                                    {
                                        update(labelA, z * width * height + y * width + x + 1, labelB);
                                        x++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    { // forward backward
                        // horizontal bidirectional
                        int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                        int a114 = labels[(z - 1) * width * height + (y - 1) * width + (x + 2)];
                        int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];
                        int a124 = labels[(z - 1) * width * height + (y) * width + (x + 2)];
                        int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];
                        int a134 = labels[(z - 1) * width * height + (y + 1) * width + (x + 2)];

                        int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                        int a214 = labels[(z) * width * height + (y - 1) * width + (x + 2)];
                        // int a223 = labels[(z) * width * height + (y) * width + (x + 1)];
                        int a224 = labels[(z) * width * height + (y) * width + (x + 2)];
                        int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                        int a234 = labels[(z) * width * height + (y + 1) * width + (x + 2)];

                        int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                        int a314 = labels[(z + 1) * width * height + (y - 1) * width + (x + 2)];
                        int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                        int a324 = labels[(z + 1) * width * height + (y) * width + (x + 2)];
                        int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];
                        int a334 = labels[(z + 1) * width * height + (y + 1) * width + (x + 2)];
                        if( 
                            checkSplit_3d(
                                a134, a133, a124, a123, a114, a113,
                                a234, a233, a224, a223, a214, a213,
                                a334, a333, a324, a323, a314, a313
                            ) // counterclockwise rotate pi along z 
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = threebythreebyfour(x, y, z, labelA);
                                priorB = threebythreebyfour(x, y, z, labelB);
                            }

                            if( probability(z * width * height + y * width + x + 1, labelB, labelA, priorB, priorA) )
                            {
                                update(labelA, z * width * height + y * width + x + 1, labelB);
                                x++;
                            }
                            else
                            {
                                int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                                int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                                int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                                int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                                int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                                int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];

                                int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                                int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                                int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                                // int a222 = labels[(z) * width * height + (y) * width + (x)];
                                int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                                int a232 = labels[(z) * width * height + (y + 1) * width + (x)];

                                int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                                int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                                int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                                int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                                int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                                int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                                if( 
                                    checkSplit_3d(
                                        a111, a112, a121, a122, a131, a132, 
                                        a211, a212, a221, a222, a231, a232,
                                        a311, a312, a321, a322, a331, a332 
                                    ) 
                                )
                                {
                                    if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                                    {
                                        update(labelB, z * width * height + y * width + x, labelA);
                                    }
                                }
                            }
                        }
                    }
                } // labelA != labelB
            } // for x
        } // for y
    } // for z

    for (int z = 1; z < depth - 1; z++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 2; y++)
            {

                labelA = labels[(z) * width * height + (y) * width + (x)]; // a222
                labelB = labels[(z) * width * height + (y + 1) * width + (x)]; // a232
                if( labelA != labelB )
                {
                    int a222 = labelA;
                    int a232 = labelB;

                    if( forwardbackward )
                    {
                        // vertical bidirectional
                        int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                        int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                        int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                        int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                        int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                        int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];

                        int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                        int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                        int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                        int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                        // int a222 = labels[(z) * width * height + (y) * width + (x)];
                        int a223 = labels[(z) * width * height + (y) * width + (x + 1)];

                        int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                        int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                        int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                        int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                        int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                        int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                        if( 
                            checkSplit_3d(
                                a113, a123, a112, a122, a111, a121,
                                a213, a223, a212, a222, a211, a221,
                                a313, a323, a312, a322, a311, a321
                            ) // counterclockwise rotate pi/2 along z
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = threebyfourbythree(x, y, z, labelA);
                                priorB = threebyfourbythree(x, y, z, labelB);
                            }

                            if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                            {
                                update(labelB, z * width * height + y * width + x, labelA);
                            }
                            else
                            {
                                int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                                int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];
                                int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];
                                int a141 = labels[(z - 1) * width * height + (y + 2) * width + (x - 1)];
                                int a142 = labels[(z - 1) * width * height + (y + 2) * width + (x)];
                                int a143 = labels[(z - 1) * width * height + (y + 2) * width + (x + 1)];

                                int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                                // int a232 = labels[(z) * width * height + (y + 1) * width + (x)];
                                int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                                int a241 = labels[(z) * width * height + (y + 2) * width + (x - 1)];
                                int a242 = labels[(z) * width * height + (y + 2) * width + (x)];
                                int a243 = labels[(z) * width * height + (y + 2) * width + (x + 1)];

                                int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                                int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                                int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];
                                int a341 = labels[(z + 1) * width * height + (y + 2) * width + (x - 1)];
                                int a342 = labels[(z + 1) * width * height + (y + 2) * width + (x)];
                                int a343 = labels[(z + 1) * width * height + (y + 2) * width + (x + 1)];
                                if( 
                                    checkSplit_3d(
                                        a141, a131, a142, a132, a143, a133,
                                        a241, a231, a242, a232, a243, a233,
                                        a341, a331, a342, a332, a343, a333
                                    ) // clockwise rotate pi/2 along z
                                )
                                {
                                    if( probability(z * width * height + (y + 1) * width + x, labelB, labelA, priorB, priorA) )
                                    {
                                        update(labelA, z * width * height + (y + 1) * width + x, labelB);
                                        y++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    { // forwardbackward
                        // vertical bidirectional
                        int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                        int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];
                        int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];
                        int a141 = labels[(z - 1) * width * height + (y + 2) * width + (x - 1)];
                        int a142 = labels[(z - 1) * width * height + (y + 2) * width + (x)];
                        int a143 = labels[(z - 1) * width * height + (y + 2) * width + (x + 1)];

                        int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                        // int a232 = labels[(z) * width * height + (y + 1) * width + (x)];
                        int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                        int a241 = labels[(z) * width * height + (y + 2) * width + (x - 1)];
                        int a242 = labels[(z) * width * height + (y + 2) * width + (x)];
                        int a243 = labels[(z) * width * height + (y + 2) * width + (x + 1)];

                        int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                        int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                        int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];
                        int a341 = labels[(z + 1) * width * height + (y + 2) * width + (x - 1)];
                        int a342 = labels[(z + 1) * width * height + (y + 2) * width + (x)];
                        int a343 = labels[(z + 1) * width * height + (y + 2) * width + (x + 1)];
                        if( 
                            checkSplit_3d(
                                a141, a131, a142, a132, a143, a133,
                                a241, a231, a242, a232, a243, a233,
                                a341, a331, a342, a332, a343, a333
                            ) // clockwise rotate pi/2 along z
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = threebyfourbythree(x, y, z, labelA);
                                priorB = threebyfourbythree(x, y, z, labelB);
                            }

                            if( probability(z * width * height + (y + 1) * width + x, labelB, labelA, priorB, priorA) )
                            {
                                update(labelA, z * width * height + (y + 1) * width + x, labelB);
                                y++;
                            }
                            else
                            {
                                int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                                int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                                int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                                int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                                int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                                int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];

                                int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                                int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                                int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                                int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                                // int a222 = labels[(z) * width * height + (y) * width + (x)];
                                int a223 = labels[(z) * width * height + (y) * width + (x + 1)];

                                int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                                int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                                int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                                int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                                int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                                int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                                if( 
                                    checkSplit_3d(
                                        a113, a123, a112, a122, a111, a121,
                                        a213, a223, a212, a222, a211, a221,
                                        a313, a323, a312, a322, a311, a321
                                    ) // counterclockwise rotate pi/2 along z 
                                )
                                {
                                    if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                                    {
                                        update(labelB, z * width * height + y * width + x, labelA);
                                    }
                                }
                            }
                        }
                    }
                } // labelA != labelB
            } // for y
        } // for x
    } // for z 

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int z = 1; z < depth - 2; z++)
            {

                labelA = labels[(z) * width * height + (y) * width + (x)]; // a222
                labelB = labels[(z + 1) * width * height + (y) * width + (x)]; // a322
                if( labelA != labelB )
                {
                    int a222 = labelA;
                    int a322 = labelB;

                    if( forwardbackward )
                    {
                        // depth bidirectional
                        int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                        int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                        int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                        int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                        int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                        int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];
                        int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                        int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];
                        int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];

                        int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                        int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                        int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                        int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                        // int a222 = labels[(z) * width * height + (y) * width + (x)];
                        int a223 = labels[(z) * width * height + (y) * width + (x + 1)];
                        int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                        int a232 = labels[(z) * width * height + (y + 1) * width + (x)];
                        int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                        if( 
                            checkSplit_3d(
                                a113, a213, a123, a223, a133, a233,
                                a112, a212, a122, a222, a132, a232,
                                a111, a211, a121, a221, a131, a231
                            ) // conterclockwise rotate pi/2 along y
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = fourbythreebythree(x, y, z, labelA);
                                priorB = fourbythreebythree(x, y, z, labelB);
                            }

                            if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                            {
                                update(labelB, z * width * height + y * width + x, labelA);
                            }
                            else
                            {
                                int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                                int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                                int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                                int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                                // int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                                int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                                int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                                int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                                int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];

                                int a411 = labels[(z + 2) * width * height + (y - 1) * width + (x - 1)];
                                int a412 = labels[(z + 2) * width * height + (y - 1) * width + (x)];
                                int a413 = labels[(z + 2) * width * height + (y - 1) * width + (x + 1)];
                                int a421 = labels[(z + 2) * width * height + (y) * width + (x - 1)];
                                int a422 = labels[(z + 2) * width * height + (y) * width + (x)];
                                int a423 = labels[(z + 2) * width * height + (y) * width + (x + 1)];
                                int a431 = labels[(z + 2) * width * height + (y + 1) * width + (x - 1)];
                                int a432 = labels[(z + 2) * width * height + (y + 1) * width + (x)];
                                int a433 = labels[(z + 2) * width * height + (y + 1) * width + (x + 1)];
                                if( 
                                    checkSplit_3d(
                                        a411, a311, a421, a321, a431, a331, 
                                        a412, a312, a422, a322, a432, a332, 
                                        a413, a313, a423, a323, a433, a333 
                                    ) // clockwise rotate pi/2 along y 
                                )
                                {
                                    if( probability((z + 1) * width * height + (y) * width + x, labelB, labelA, priorB, priorA) )
                                    {
                                        update(labelA, (z + 1) * width * height + (y) * width + x, labelB);
                                        z++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    { // forwardbackward
                        // depth bidirectional
                        int a311 = labels[(z + 1) * width * height + (y - 1) * width + (x - 1)];
                        int a312 = labels[(z + 1) * width * height + (y - 1) * width + (x)];
                        int a313 = labels[(z + 1) * width * height + (y - 1) * width + (x + 1)];
                        int a321 = labels[(z + 1) * width * height + (y) * width + (x - 1)];
                        // int a322 = labels[(z + 1) * width * height + (y) * width + (x)];
                        int a323 = labels[(z + 1) * width * height + (y) * width + (x + 1)];
                        int a331 = labels[(z + 1) * width * height + (y + 1) * width + (x - 1)];
                        int a332 = labels[(z + 1) * width * height + (y + 1) * width + (x)];
                        int a333 = labels[(z + 1) * width * height + (y + 1) * width + (x + 1)];

                        int a411 = labels[(z + 2) * width * height + (y - 1) * width + (x - 1)];
                        int a412 = labels[(z + 2) * width * height + (y - 1) * width + (x)];
                        int a413 = labels[(z + 2) * width * height + (y - 1) * width + (x + 1)];
                        int a421 = labels[(z + 2) * width * height + (y) * width + (x - 1)];
                        int a422 = labels[(z + 2) * width * height + (y) * width + (x)];
                        int a423 = labels[(z + 2) * width * height + (y) * width + (x + 1)];
                        int a431 = labels[(z + 2) * width * height + (y + 1) * width + (x - 1)];
                        int a432 = labels[(z + 2) * width * height + (y + 1) * width + (x)];
                        int a433 = labels[(z + 2) * width * height + (y + 1) * width + (x + 1)];
                        if( 
                            checkSplit_3d(
                                a411, a311, a421, a321, a431, a331, 
                                a412, a312, a422, a322, a432, a332, 
                                a413, a313, a423, a323, a433, a333 
                            ) // clockwise rotate pi/2 along y 
                        )
                        {
                            if( seeds_prior )
                            {
                                priorA = fourbythreebythree(x, y, z, labelA);
                                priorB = fourbythreebythree(x, y, z, labelB);
                            }

                            if( probability((z + 1) * width * height + (y) * width + x, labelB, labelA, priorB, priorA) )
                            {
                                update(labelA, (z + 1) * width * height + (y) * width + x, labelB);
                                z++;
                            }
                            else
                            {
                                int a111 = labels[(z - 1) * width * height + (y - 1) * width + (x - 1)];
                                int a112 = labels[(z - 1) * width * height + (y - 1) * width + (x)];
                                int a113 = labels[(z - 1) * width * height + (y - 1) * width + (x + 1)];
                                int a121 = labels[(z - 1) * width * height + (y) * width + (x - 1)];
                                int a122 = labels[(z - 1) * width * height + (y) * width + (x)];
                                int a123 = labels[(z - 1) * width * height + (y) * width + (x + 1)];
                                int a131 = labels[(z - 1) * width * height + (y + 1) * width + (x - 1)];
                                int a132 = labels[(z - 1) * width * height + (y + 1) * width + (x)];
                                int a133 = labels[(z - 1) * width * height + (y + 1) * width + (x + 1)];

                                int a211 = labels[(z) * width * height + (y - 1) * width + (x - 1)];
                                int a212 = labels[(z) * width * height + (y - 1) * width + (x)];
                                int a213 = labels[(z) * width * height + (y - 1) * width + (x + 1)];
                                int a221 = labels[(z) * width * height + (y) * width + (x - 1)];
                                // int a222 = labels[(z) * width * height + (y) * width + (x)];
                                int a223 = labels[(z) * width * height + (y) * width + (x + 1)];
                                int a231 = labels[(z) * width * height + (y + 1) * width + (x - 1)];
                                int a232 = labels[(z) * width * height + (y + 1) * width + (x)];
                                int a233 = labels[(z) * width * height + (y + 1) * width + (x + 1)];
                                if( 
                                    checkSplit_3d(
                                        a113, a213, a123, a223, a133, a233,
                                        a112, a212, a122, a222, a132, a232,
                                        a111, a211, a121, a221, a131, a231
                                    ) // conterclockwise rotate pi/2 along y 
                                )
                                {
                                    if( probability(z * width * height + y * width + x, labelA, labelB, priorA, priorB) )
                                    {
                                        update(labelB, z * width * height + y * width + x, labelA);
                                    }
                                }
                            }
                        }
                    }
                } // labelA != labelB
            } // for z
        } // for x
    } // for y
    forwardbackward = !forwardbackward;

    // update border pixels
    for (int z = 0; z < depth; z++)
    {
        for (int x = 0; x < width; x++)
        {
            labelA = labels[z * width * height + x];
            labelB = labels[z * width * height + width + x];
            if( labelA != labelB )
                update(labelB, z * width * height + x, labelA);
            labelA = labels[z * width * height + (height - 1) * width + x];
            labelB = labels[z * width * height + (height - 2) * width + x];
            if( labelA != labelB )
                update(labelB, z * width * height + (height - 1) * width + x, labelA);
        }
    }
    for (int z = 0; z < depth; z++)
    {
        for (int y = 0; y < height; y++)
        {
            labelA = labels[z * width * height + y * width];
            labelB = labels[z * width * height + y * width + 1];
            if( labelA != labelB )
                update(labelB, z * width * height + y * width, labelA);
            labelA = labels[z * width * height + y * width + width - 1];
            labelB = labels[z * width * height + y * width + width - 2];
            if( labelA != labelB )
                update(labelB, z * width * height + y * width + width - 1, labelA);
        }
    }
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            labelA = labels[y * width + x];
            labelB = labels[width * height + y * width + x];
            if( labelA != labelB )
                update(labelB, y * width + x, labelA);
            labelA = labels[(depth - 1) * width * height + y * width + x];
            labelB = labels[(depth - 2) * width * height + y * width + x];
            if( labelA != labelB )
                update(labelB, (depth - 1) * width * height + y * width + x, labelA);
        }
    }
}

void SuperpixelSEEDS3DImpl::update(int label_new, int image_idx, int label_old)
{
    //change the label of a single pixel
    deletePixel(seeds_top_level, label_old, image_idx);
    addPixel(seeds_top_level, label_new, image_idx);
    labels[image_idx] = label_new;
}

void SuperpixelSEEDS3DImpl::addPixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]++;
    T[level][label]++;
}

void SuperpixelSEEDS3DImpl::deletePixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]--;
    T[level][label]--;
}

void SuperpixelSEEDS3DImpl::addBlock(int level, int label, int sublevel, int sublabel)
{
    parent[sublevel][sublabel] = label;

    HISTN* h_label = &histogram[level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //add the (sublevel, sublabel) block to the block (level, label)
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_add_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; n++)
        h_label[n] += h_sublabel[n];

    T[level][label] += T[sublevel][sublabel];
}

void SuperpixelSEEDS3DImpl::addBlockToplevel(int label, int sublevel, int sublabel)
{
    addBlock(seeds_top_level, label, sublevel, sublabel);
    nr_partitions[label]++;
}

void SuperpixelSEEDS3DImpl::deleteBlockToplevel(int label, int sublevel, int sublabel)
{
    HISTN* h_label = &histogram[seeds_top_level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //do the reverse operation of add_block_toplevel
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_sub_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
        h_label[n] -= h_sublabel[n];

    T[seeds_top_level][label] -= T[sublevel][sublabel];

    nr_partitions[label]--;
}

void SuperpixelSEEDS3DImpl::updateLabels()
{
    for (int i = 0; i < width * height * depth; ++i)
        labels[i] = parent[0][labels_bottom[i]];
}

bool SuperpixelSEEDS3DImpl::probability(int image_idx, int label1, int label2,
        int prior1, int prior2)
{
    unsigned int color = image_bins[image_idx];
    float P_label1 = histogram[seeds_top_level][label1 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label2];
    float P_label2 = histogram[seeds_top_level][label2 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label1];

    if( seeds_prior )
    {
        float p;
        if( prior2 != 0 )
            p = (float) prior1 / prior2;
        else //pathological case
            p = 1.f;
        switch( seeds_prior )
        {
        case 5: p *= p;
            /* fallthrough */
        case 4: p *= p;
            /* fallthrough */
        case 3: p *= p;
            /* fallthrough */
        case 2:
            p *= p;
            P_label1 *= T[seeds_top_level][label2];
            P_label2 *= T[seeds_top_level][label1];
            /* fallthrough */
        case 1:
            P_label1 *= p;
            break;
        }
    }

    return (P_label2 > P_label1);
}

int SuperpixelSEEDS3DImpl::threebythreebyfour(int x, int y, int z, int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * (z-1) x x x x (z) x x x x (z+1) x x x x
     *       x x x x     x O o x       x x x x
     *       x x x x     x x x x       x x x x
     */

#if CV_SSSE3
    __m128i addp = _mm_set1_epi32(1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp);
    countp = _mm_add_epi32(count3, countp);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;

    // z
    count += (labels[z * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y - 1) * width + x] == label);
    count += (labels[z * width * height + (y - 1) * width + x + 1] == label);
    count += (labels[z * width * height + (y - 1) * width + x + 2] == label);

    count += (labels[z * width * height + y * width + x - 1] == label);
    count += (labels[z * width * height + y * width + x + 2] == label);

    count += (labels[z * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y + 1) * width + x] == label);
    count += (labels[z * width * height + (y + 1) * width + x + 1] == label);
    count += (labels[z * width * height + (y + 1) * width + x + 2] == label);

    // z - 1
    count += (labels[(z - 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x + 1] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x + 2] == label);

    count += (labels[(z - 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + y * width + x] == label);
    count += (labels[(z - 1) * width * height + y * width + x + 1] == label);
    count += (labels[(z - 1) * width * height + y * width + x + 2] == label);

    count += (labels[(z - 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x + 1] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x + 2] == label);

    // z + 1
    count += (labels[(z + 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x + 1] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x + 2] == label);

    count += (labels[(z + 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + y * width + x] == label);
    count += (labels[(z + 1) * width * height + y * width + x + 1] == label);
    count += (labels[(z + 1) * width * height + y * width + x + 2] == label);

    count += (labels[(z + 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x + 1] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x + 2] == label);

    return count;
#endif
}

int SuperpixelSEEDS3DImpl::threebyfourbythree(int x, int y, int z, int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * (z-1) x x x (z) x x x (z+1) x x x
     *       x x x     x O x       x x x
     *       x x x     x o x       x x x
     *       x x x     x x x       x x x
     */

#if CV_SSSE3
    __m128i addp = _mm_set1_epi32(1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp);
    countp = _mm_add_epi32(count3, countp);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;

    // z
    count += (labels[z * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y - 1) * width + x] == label);
    count += (labels[z * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[z * width * height + y * width + x - 1] == label);
    count += (labels[z * width * height + y * width + x + 1] == label);

    count += (labels[z * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[z * width * height + (y + 2) * width + x - 1] == label);
    count += (labels[z * width * height + (y + 2) * width + x] == label);
    count += (labels[z * width * height + (y + 2) * width + x + 1] == label);

    // z - 1
    count += (labels[(z - 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[(z - 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + y * width + x] == label);
    count += (labels[(z - 1) * width * height + y * width + x + 1] == label);

    count += (labels[(z - 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[(z - 1) * width * height + (y + 2) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y + 2) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y + 2) * width + x + 1] == label);

    // z + 1
    count += (labels[(z + 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + y * width + x] == label);
    count += (labels[(z + 1) * width * height + y * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + (y + 2) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y + 2) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y + 2) * width + x + 1] == label);

    return count;
#endif
}

int SuperpixelSEEDS3DImpl::fourbythreebythree(int x, int y, int z, int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * (y-1) x x x (y) x x x (y+1) x x x
     *       x x x     x O x       x x x
     *       x x x     x o x       x x x
     *       x x x     x x x       x x x
     */

#if CV_SSSE3
    __m128i addp = _mm_set1_epi32(1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp);
    countp = _mm_add_epi32(count3, countp);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;

    // y
    count += (labels[(z - 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + y * width + x] == label);
    count += (labels[(z - 1) * width * height + y * width + x + 1] == label);

    count += (labels[z * width * height + y * width + x - 1] == label);
    count += (labels[z * width * height + y * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + y * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + y * width + x + 1] == label);

    count += (labels[(z + 2) * width * height + y * width + x - 1] == label);
    count += (labels[(z + 2) * width * height + y * width + x] == label);
    count += (labels[(z + 2) * width * height + y * width + x + 1] == label);

    // y - 1
    count += (labels[(z - 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[z * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y - 1) * width + x] == label);
    count += (labels[z * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y - 1) * width + x + 1] == label);

    count += (labels[(z + 2) * width * height + (y - 1) * width + x - 1] == label);
    count += (labels[(z + 2) * width * height + (y - 1) * width + x] == label);
    count += (labels[(z + 2) * width * height + (y - 1) * width + x + 1] == label);

    // y + 1
    count += (labels[(z - 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z - 1) * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[z * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[z * width * height + (y + 1) * width + x] == label);
    count += (labels[z * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[(z + 1) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z + 1) * width * height + (y + 1) * width + x + 1] == label);

    count += (labels[(z + 2) * width * height + (y + 1) * width + x - 1] == label);
    count += (labels[(z + 2) * width * height + (y + 1) * width + x] == label);
    count += (labels[(z + 2) * width * height + (y + 1) * width + x + 1] == label);

    return count;
#endif
}

// int SuperpixelSEEDSImpl::threebyfour(int x, int y, int label)
// {
//     /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
//      * neighborhood (x=counted, o,O=ignored, O=(x,y)):
//      * x x x x
//      * x O o x
//      * x x x x
//      */

// #if CV_SSSE3
//     __m128i addp = _mm_set1_epi32(1);
//     __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
//     __m128i labelp = _mm_set1_epi32(label);
//     /* 1. row */
//     __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
//     __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
//     __m128i countp = _mm_and_si128(mask1, addp);
//     /* 2. row */
//     __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
//     __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
//     __m128i count1 = _mm_and_si128(mask2, addp_middle);
//     countp = _mm_add_epi32(countp, count1);
//     /* 3. row */
//     __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
//     __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
//     __m128i count3 = _mm_and_si128(mask3, addp);
//     countp = _mm_add_epi32(count3, countp);

//     countp = _mm_hadd_epi32(countp, countp);
//     countp = _mm_hadd_epi32(countp, countp);
//     return _mm_cvtsi128_si32(countp);
// #else
//     int count = 0;
//     count += (labels[(y - 1) * width + x - 1] == label);
//     count += (labels[(y - 1) * width + x] == label);
//     count += (labels[(y - 1) * width + x + 1] == label);
//     count += (labels[(y - 1) * width + x + 2] == label);

//     count += (labels[y * width + x - 1] == label);
//     count += (labels[y * width + x + 2] == label);

//     count += (labels[(y + 1) * width + x - 1] == label);
//     count += (labels[(y + 1) * width + x] == label);
//     count += (labels[(y + 1) * width + x + 1] == label);
//     count += (labels[(y + 1) * width + x + 2] == label);

//     return count;
// #endif
// }

// int SuperpixelSEEDSImpl::fourbythree(int x, int y, int label)
// {
//     /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
//      * neighborhood (x=counted, o,O=ignored, O=(x,y)):
//      * x x x o
//      * x O o x
//      * x o o x
//      * x x x o
//      */

// #if CV_SSSE3
//     __m128i addp_border = _mm_set_epi32(0, 1, 1, 1);
//     __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
//     __m128i labelp = _mm_set1_epi32(label);
//     /* 1. row */
//     __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
//     __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
//     __m128i countp = _mm_and_si128(mask1, addp_border);
//     /* 2. row */
//     __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
//     __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
//     __m128i count1 = _mm_and_si128(mask2, addp_middle);
//     countp = _mm_add_epi32(countp, count1);
//     /* 3. row */
//     __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
//     __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
//     __m128i count3 = _mm_and_si128(mask3, addp_middle);
//     countp = _mm_add_epi32(count3, countp);
//     /* 4. row */
//     __m128i data4 = _mm_loadu_si128((__m128i*) (labels + (y+2)*width + x -1));
//     __m128i mask4 = _mm_cmpeq_epi32(data4, labelp);
//     __m128i count4 = _mm_and_si128(mask4, addp_border);
//     countp = _mm_add_epi32(countp, count4);

//     countp = _mm_hadd_epi32(countp, countp);
//     countp = _mm_hadd_epi32(countp, countp);
//     return _mm_cvtsi128_si32(countp);
// #else
//     int count = 0;
//     count += (labels[(y - 1) * width + x - 1] == label);
//     count += (labels[(y - 1) * width + x] == label);
//     count += (labels[(y - 1) * width + x + 1] == label);

//     count += (labels[y * width + x - 1] == label);
//     count += (labels[y * width + x + 2] == label);

//     count += (labels[(y + 1) * width + x - 1] == label);
//     count += (labels[(y + 1) * width + x + 2] == label);

//     count += (labels[(y + 2) * width + x - 1] == label);
//     count += (labels[(y + 2) * width + x] == label);
//     count += (labels[(y + 2) * width + x + 1] == label);

//     return count;
// #endif
// }

float SuperpixelSEEDS3DImpl::intersectConf(int level1, int label1A, int label1B,
        int level2, int label2)
{
    float sumA = 0, sumB = 0;
    float* h1A = &histogram[level1][label1A * histogram_size_aligned];
    float* h1B = &histogram[level1][label1B * histogram_size_aligned];
    float* h2 = &histogram[level2][label2 * histogram_size_aligned];
    const float count1A = T[level1][label1A];
    const float count2 = T[level2][label2];
    const float count1B = T[level1][label1B] - count2;

    /* this calculates several things:
     * - normalized intersection of a histogram. which is equal to:
     *   sum i over bins ( min(histogram1_i / T1_i, histogram2_i / T2_i) )
     * - intersection A = intersection of (level1, label1A) and (level2, label2)
     * - intersection B =
     *     intersection of (level1, label1B) - (level2, label2) and (level2, label2)
     *   where (level1, label1B) - (level2, label2)
     *     is the subtraction of 2 histograms (-> delete_block method)
     * - returns the difference between the 2 intersections: intA - intB
     */

    int n = 0;
#if CV_SSSE3
    __m128 count1Ap = _mm_set1_ps(count1A);
    __m128 count2p = _mm_set1_ps(count2);
    __m128 count1Bp = _mm_set1_ps(count1B);
    __m128 sumAp = _mm_set1_ps(0.0f);
    __m128 sumBp = _mm_set1_ps(0.0f);

    const int loop_end = histogram_size - 3;
    for(; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time

        // normal
        __m128 h1Ap = _mm_load_ps(h1A + n);
        __m128 h1Bp = _mm_load_ps(h1B + n);
        __m128 h2p = _mm_load_ps(h2 + n);

        __m128 h1ApC2 = _mm_mul_ps(h1Ap, count2p);
        __m128 h2pC1A = _mm_mul_ps(h2p, count1Ap);
        __m128 maskA = _mm_cmple_ps(h1ApC2, h2pC1A);
        __m128 sum1AddA = _mm_and_ps(maskA, h1ApC2);
        __m128 sum2AddA = _mm_andnot_ps(maskA, h2pC1A);
        sumAp = _mm_add_ps(sumAp, sum1AddA);
        sumAp = _mm_add_ps(sumAp, sum2AddA);

        // del
        __m128 diffp = _mm_sub_ps(h1Bp, h2p);
        __m128 h1BpC2 = _mm_mul_ps(diffp, count2p);
        __m128 h2pC1B = _mm_mul_ps(h2p, count1Bp);
        __m128 maskB = _mm_cmple_ps(h1BpC2, h2pC1B);
        __m128 sum1AddB = _mm_and_ps(maskB, h1BpC2);
        __m128 sum2AddB = _mm_andnot_ps(maskB, h2pC1B);
        sumBp = _mm_add_ps(sumBp, sum1AddB);
        sumBp = _mm_add_ps(sumBp, sum2AddB);
    }
    // merge results (quite expensive)
    float sum1Asse;
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    _mm_store_ss(&sum1Asse, sumAp);

    float sum1Bsse;
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    _mm_store_ss(&sum1Bsse, sumBp);

    sumA += sum1Asse;
    sumB += sum1Bsse;
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
    {
        // normal intersect
        if( h1A[n] * count2 < h2[n] * count1A ) sumA += h1A[n] * count2;
        else sumA += h2[n] * count1A;

        // intersect_del
        float diff = h1B[n] - h2[n];
        if( diff * count2 < h2[n] * count1B ) sumB += diff * count2;
        else sumB += h2[n] * count1B;
    }

    float intA = sumA / (count1A * count2);
    float intB = sumB / (count1B * count2);
    return intA - intB;
}

bool SuperpixelSEEDS3DImpl::checkSplit_3d(
    int a111, int a112, int a121, int a122, int a131, int a132,
    int a211, int a212, int a221, int a222, int a231, int a232,
    int a311, int a312, int a321, int a322, int a331, int a332)
{   

    if( (a222 != a221) && (a222 == a212) && (a222 == a232) ) return false;
    if( (a222 != a211) && (a222 == a212) && (a222 == a221) ) return false;
    if( (a222 != a231) && (a222 == a221) && (a222 == a232) ) return false;

    if( (a222 != a221) && (a222 == a122) && (a222 == a322) ) return false;
    if( (a222 != a121) && (a222 == a122) && (a222 == a221) ) return false;
    if( (a222 != a321) && (a222 == a221) && (a222 == a322) ) return false;

    if( (a222 != a112) && (a222 == a122) && (a222 == a212) ) return false;
    if( (a222 != a132) && (a222 == a122) && (a222 == a232) ) return false;
    if( (a222 != a312) && (a222 == a212) && (a222 == a322) ) return false;
    if( (a222 != a332) && (a222 == a232) && (a222 == a322) ) return false;

    return true;
}
// bool SuperpixelSEEDS3DImpl::checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33)
// {
//     if( (a22 != a23) && (a22 == a12) && (a22 == a32) ) return false;
//     if( (a22 != a13) && (a22 == a12) && (a22 == a23) ) return false;
//     if( (a22 != a33) && (a22 == a32) && (a22 == a23) ) return false;
//     return true;

// }
// bool SuperpixelSEEDS3DImpl::checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23)
// {
//     if( (a22 != a12) && (a22 == a21) && (a22 == a23) ) return false;
//     if( (a22 != a11) && (a22 == a21) && (a22 == a12) ) return false;
//     if( (a22 != a13) && (a22 == a23) && (a22 == a12) ) return false;
//     return true;
// }
// bool SuperpixelSEEDS3DImpl::checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33)
// {
//     if( (a22 != a32) && (a22 == a21) && (a22 == a23) ) return false;
//     if( (a22 != a31) && (a22 == a21) && (a22 == a32) ) return false;
//     if( (a22 != a33) && (a22 == a23) && (a22 == a32) ) return false;
//     return true;
// }

void SuperpixelSEEDS3DImpl::getLabelContourMask(OutputArray image, bool thick_line, int idx)
{
    image.create(height, width, CV_8UC1);
    Mat dst = image.getMat();
    dst.setTo(Scalar(0));

    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int neighbors = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                    int index = y * width + x;
                    int mainindex = j * width + k;
                    if( labels[idx*width*height+mainindex] != labels[idx*height*width+index] )
                    {
                        if( thick_line || !*dst.ptr<uchar>(y, x) )
                            neighbors++;
                    }
                }
            }
            if( neighbors > 1 )
                *dst.ptr<uchar>(j, k) = (uchar)255;
        }
    }
}

} // namespace ximgproc
} // namespace cv
