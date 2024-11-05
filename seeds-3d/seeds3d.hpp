#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unordered_set>
#include <queue>
using namespace std;

struct point {
    int x, y, z;
    point(int x, int y, int z) : x(x), y(y), z(z) {}
    bool operator==(const point &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace cv
{

    class seeds3d
    {
    public:
        seeds3d(int img_height, int img_width, int depth, int num_superpixels, int num_histogram, int seeds_prior);
        void seeds3d_func(vector<vector<vector<float>>> img, int num_interations);
        void initImage(vector<vector<vector<float>>> img);
        int computeLabels(tuple<int, int, int> idx);
        tuple<int, int, int> computeSuperPixelSize();
        int computeColor(float input);
        void changePixelLabel(tuple<int, int, int> idx, int new_label);
        bool computeProbability(tuple<int, int, int> idx, int label_a, int label_b, int prior_a, int prior_b);
        int computePrior_x(tuple<int, int, int> idx, int label);
        int computePrior_y(tuple<int, int, int> idx, int label);
        int computePrior_z(tuple<int, int, int> idx, int label);
        void update();
        bool checkSpilit(tuple<int, int, int> idx);
        bool isConnected(const point &a, const point &b);
        bool isConnectedSpace(const vector<point> &points);
        ~seeds3d();

    private:
        int img_height, img_width, img_depth, num_superpixels, num_histogram,
            num_x, num_y, num_z, superpixel_height, superpixel_width, superpixel_depth, seeds_prior;
        vector<vector<vector<int>>> labels, colors;
        vector<vector<int>> histogram;
        vector<int> T;
    };

}



namespace std {
    template <>
    struct hash<point> {
        size_t operator()(const point& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1) ^ (hash<int>()(p.z) << 2);
        }
    };
}