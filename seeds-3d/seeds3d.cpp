#include "seeds3d.hpp"
#include "precomp.hpp"

namespace cv
{

    seeds3d::seeds3d(int img_width, int img_height, int img_depth, int num_superpixels, int num_histogram, int seeds_prior)
    {
        this->img_width = img_width;
        this->img_height = img_height;
        this->img_depth = img_depth;
        this->num_superpixels = num_superpixels;
        this->num_histogram = num_histogram;
        this->seeds_prior = seeds_prior;
        vector<vector<vector<int>>> label_tmp(img_width, vector<vector<int>>(img_height, vector<int>(img_depth, -1)));
        this->labels = label_tmp;
        vector<vector<int>> histogram_tmp(num_superpixels, vector<int>(num_histogram, -1));
        this->histogram = histogram_tmp;
        vector<int> T_tmp = vector<int>(num_superpixels, -1);
        this->T = T_tmp;
        vector<vector<vector<int>>> color_tmp(img_width, vector<vector<int>>(img_height, vector<int>(img_depth, -1)));
        this->colors = color_tmp;
    }

    void seeds3d::initImage(vector<vector<vector<float>>> img)
    {
        computeSuperPixelSize();
        for (int x = 0; x < img_width; x++)
        {
            for (int y = 0; y < img_height; y++)
            {
                for (int z = 0; z < img_depth; z++)
                {
                    int color = computeColor(img[x][y][z]);
                    colors[x][y][z] = color;
                    int label = computeLabels({x, y, z});
                    labels[x][y][z] = label;
                    T[label]++;
                    histogram[label][color]++;
                }
            }
        }
    }

    int seeds3d::computeLabels(tuple<int, int, int> idx)
    {
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int label = min(num_z - 1, int(floor((z / superpixel_depth)))) * (num_x * num_y) + min(num_y - 1, int(floor(y / superpixel_height))) * num_x +
                    min(num_x - 1, int(floor(x / superpixel_width)));
        return label;
    }

    tuple<int, int, int> seeds3d::computeSuperPixelSize()
    {
        double product = img_width * img_height * img_depth;
        double p_tmp = cbrt(num_superpixels / product);
        num_x = min(img_width, max(1, int(floor(p_tmp * img_width))));
        num_y = min(img_height, max(1, int(floor(p_tmp * img_height))));
        num_z = min(img_depth, max(1, int(floor(num_superpixels / num_x / num_y))));
        superpixel_width = int(floor(img_width / num_x));
        superpixel_height = int(floor(img_height / num_y));
        superpixel_depth = int(floor(img_depth / num_z));
        return {superpixel_width, superpixel_height, superpixel_depth};
    }

    int seeds3d::computeColor(float input)
    {
        return min(num_histogram - 1, int(input / (float(1.0 / num_histogram))));
    }

    int seeds3d::computePrior_x(tuple<int, int, int> idx, int label)
    {
        /*   xy (z & z+1 & z-1)
        0 0 0 0
        0 a b 0
        0 0 0 0
        */
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int count = 0;

        count += (labels[x - 1][y + 1][z] == label);
        count += (labels[x][y + 1][z] == label);
        count += (labels[x + 1][y + 1][z] == label);
        count += (labels[x + 2][y + 1][z] == label);

        count += (labels[x - 1][y][z] == label);
        count += (labels[x + 2][y][z] == label);

        count += (labels[x - 1][y - 1][z] == label);
        count += (labels[x][y - 1][z] == label);
        count += (labels[x + 1][y - 1][z] == label);
        count += (labels[x + 2][y - 1][z] == label);

        count += (labels[x - 1][y + 1][z - 1] == label);
        count += (labels[x][y + 1][z - 1] == label);
        count += (labels[x + 1][y + 1][z - 1] == label);
        count += (labels[x + 2][y + 1][z - 1] == label);

        count += (labels[x - 1][y][z - 1] == label);
        count += (labels[x][y][z - 1] == label);
        count += (labels[x + 1][y][z - 1] == label);
        count += (labels[x + 2][y][z - 1] == label);

        count += (labels[x - 1][y - 1][z - 1] == label);
        count += (labels[x][y - 1][z - 1] == label);
        count += (labels[x + 1][y - 1][z - 1] == label);
        count += (labels[x + 2][y - 1][z - 1] == label);

        count += (labels[x - 1][y + 1][z + 1] == label);
        count += (labels[x][y + 1][z + 1] == label);
        count += (labels[x + 1][y + 1][z + 1] == label);
        count += (labels[x + 2][y + 1][z + 1] == label);

        count += (labels[x - 1][y][z + 1] == label);
        count += (labels[x][y][z + 1] == label);
        count += (labels[x + 1][y][z + 1] == label);
        count += (labels[x + 2][y][z + 1] == label);

        count += (labels[x - 1][y - 1][z + 1] == label);
        count += (labels[x][y - 1][z + 1] == label);
        count += (labels[x + 1][y - 1][z + 1] == label);
        count += (labels[x + 2][y - 1][z + 1] == label);

        return count;
    }

    int seeds3d::computePrior_y(tuple<int, int, int> idx, int label)
    {
        /* xy (z & z-1 & z+1)
        0 0 0
        0 b 0
        0 a 0
        0 0 0
        */
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int count = 0;

        count += (labels[x - 1][y - 1][z] == label);
        count += (labels[x][y - 1][z] == label);
        count += (labels[x + 1][y - 1][z] == label);

        count += (labels[x - 1][y][z] == label);
        count += (labels[x + 1][y][z] == label);

        count += (labels[x - 1][y + 1][z] == label);
        count += (labels[x + 1][y + 1][z] == label);

        count += (labels[x - 1][y + 2][z] == label);
        count += (labels[x][y + 2][z] == label);
        count += (labels[x + 1][y + 2][z] == label);

        count += (labels[x - 1][y - 1][z - 1] == label);
        count += (labels[x][y - 1][z - 1] == label);
        count += (labels[x + 1][y - 1][z - 1] == label);

        count += (labels[x - 1][y][z - 1] == label);
        count += (labels[x][y][z - 1] == label);
        count += (labels[x + 1][y][z - 1] == label);

        count += (labels[x - 1][y + 1][z - 1] == label);
        count += (labels[x][y + 1][z - 1] == label);
        count += (labels[x + 1][y + 1][z - 1] == label);

        count += (labels[x - 1][y + 2][z - 1] == label);
        count += (labels[x][y + 2][z - 1] == label);
        count += (labels[x + 1][y + 2][z - 1] == label);

        count += (labels[x - 1][y - 1][z + 1] == label);
        count += (labels[x][y - 1][z + 1] == label);
        count += (labels[x + 1][y - 1][z + 1] == label);

        count += (labels[x - 1][y][z + 1] == label);
        count += (labels[x][y][z + 1] == label);
        count += (labels[x + 1][y][z + 1] == label);

        count += (labels[x - 1][y + 1][z + 1] == label);
        count += (labels[x][y + 1][z + 1] == label);
        count += (labels[x + 1][y + 1][z + 1] == label);

        count += (labels[x - 1][y + 2][z + 1] == label);
        count += (labels[x][y + 2][z + 1] == label);
        count += (labels[x + 1][y + 2][z + 1] == label);

        return count;
    }

    int seeds3d::computePrior_z(tuple<int, int, int> idx, int label)
    {
        /* yz (x & x-1 & x+1)
        0 0 0
        0 b 0
        0 a 0
        0 0 0
        */
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int count = 0;

        count += (labels[x][y - 1][z - 1] == label);
        count += (labels[x][y][z - 1] == label);
        count += (labels[x][y + 1][z - 1] == label);

        count += (labels[x][y - 1][z] == label);
        count += (labels[x][y + 1][z] == label);

        count += (labels[x][y - 1][z + 1] == label);
        count += (labels[x][y + 1][z + 1] == label);

        count += (labels[x][y - 1][z + 2] == label);
        count += (labels[x][y][z + 2] == label);
        count += (labels[x][y + 1][z + 2] == label);

        count += (labels[x - 1][y - 1][z - 1] == label);
        count += (labels[x - 1][y][z - 1] == label);
        count += (labels[x - 1][y + 1][z - 1] == label);

        count += (labels[x - 1][y - 1][z] == label);
        count += (labels[x - 1][y][z] == label);
        count += (labels[x - 1][y + 1][z] == label);

        count += (labels[x - 1][y - 1][z + 1] == label);
        count += (labels[x - 1][y][z + 1] == label);
        count += (labels[x - 1][y + 1][z + 1] == label);

        count += (labels[x - 1][y - 1][z + 2] == label);
        count += (labels[x - 1][y][z + 2] == label);
        count += (labels[x - 1][y + 1][z + 2] == label);

        count += (labels[x + 1][y - 1][z - 1] == label);
        count += (labels[x + 1][y][z - 1] == label);
        count += (labels[x + 1][y + 1][z - 1] == label);

        count += (labels[x + 1][y - 1][z] == label);
        count += (labels[x + 1][y][z] == label);
        count += (labels[x + 1][y + 1][z] == label);

        count += (labels[x + 1][y - 1][z + 1] == label);
        count += (labels[x + 1][y][z + 1] == label);
        count += (labels[x + 1][y + 1][z + 1] == label);

        count += (labels[x + 1][y - 1][z + 2] == label);
        count += (labels[x + 1][y][z + 2] == label);
        count += (labels[x + 1][y + 1][z + 2] == label);

        return count;
    }

    bool seeds3d::computeProbability(tuple<int, int, int> idx, int label_a, int label_b, int prior_a, int prior_b)
    {
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int color = colors[x][y][z];
        float Probability_a = histogram[label_a][color] * T[label_b];
        float Probability_b = histogram[label_b][color] * T[label_a];

        if (seeds_prior)
        {
            float p;
            if (prior_b != 0)
                p = (float)prior_a / prior_b;
            else
                p = 1.f;
            switch (seeds_prior)
            {
            case 5:
                p *= p;
            case 4:
                p *= p;
            case 3:
                p *= p;
            case 2:
                p *= p;
                Probability_a *= T[label_b];
                Probability_b *= T[label_a];
            case 1:
                Probability_a *= p;
                break;
            }
        }
        return (Probability_b > Probability_a);
    }

    void seeds3d::changePixelLabel(tuple<int, int, int> idx, int new_label)
    {
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int old_label = labels[x][y][z];
        T[old_label]--;
        T[new_label]++;
        int color = colors[x][y][z];
        histogram[old_label][color]--;
        histogram[new_label][color]++;
        labels[x][y][z] = new_label;
    }

    void seeds3d::update()
    {
        for (int z = 1; z < img_depth - 1; z++)
        {
            for (int y = 1; y < img_height - 1; y++)
            {
                for (int x = 1; x < img_width - 2; x++)
                {
                    int label_a = labels[x][y][z], label_b = labels[x + 1][y][z], prior_a = 0, prior_b = 0;
                    if (label_a == label_b)
                    {
                        continue;
                    }
                    tuple<int, int, int> idx_a = {x, y, z}, idx_b = {x + 1, y, z};
                    if (seeds_prior)
                    {
                        prior_a = computePrior_x(idx_a, label_a);
                        prior_b = computePrior_x(idx_a, label_b);
                    }
                    int P_a = computeProbability(idx_a, label_a, label_b, prior_a, prior_b);
                    if (P_a)
                    {
                        changePixelLabel(idx_a, label_b);
                    }
                    else if (!P_a)
                    {
                        // todo check
                        int P_b = computeProbability(idx_b, label_b, label_a, prior_b, prior_a);
                        if (P_b)
                        {
                            changePixelLabel(idx_b, label_a);
                        }
                    }
                }
            }
        }

        for (int z = 1; z < img_depth - 1; z++)
        {
            for (int x = 1; x < img_width - 1; x++)
            {
                for (int y = 1; y < img_height - 2; y++)
                {
                    int label_a = labels[x][y][z], label_b = labels[x][y + 1][z], prior_a = 0, prior_b = 0;
                    if (label_a == label_b)
                    {
                        continue;
                    }
                    tuple<int, int, int> idx_a = {x, y, z}, idx_b = {x, y + 1, z};
                    if (seeds_prior)
                    {
                        prior_a = computePrior_y(idx_a, label_a);
                        prior_b = computePrior_y(idx_a, label_b);
                    }
                    int P_a = computeProbability(idx_a, label_a, label_b, prior_a, prior_b);
                    if (P_a)
                    {
                        changePixelLabel(idx_a, label_b);
                    }
                    else if (!P_a)
                    {
                        // todo check
                        int P_b = computeProbability(idx_b, label_b, label_a, prior_b, prior_a);
                        if (P_b)
                        {
                            changePixelLabel(idx_b, label_a);
                        }
                    }
                }
            }
        }

        for (int x = 1; x < img_width - 1; x++)
        {
            for (int y = 1; y < img_height - 1; y++)
            {
                for (int z = 1; z < img_depth - 2; z++)
                {
                    int label_a = labels[x][y][z], label_b = labels[x][y][z + 1], prior_a = 0, prior_b = 0;
                    if (label_a == label_b)
                    {
                        continue;
                    }
                    tuple<int, int, int> idx_a = {x, y, z}, idx_b = {x, y, z + 1};
                    if (seeds_prior)
                    {
                        prior_a = computePrior_z(idx_a, label_a);
                        prior_b = computePrior_z(idx_a, label_b);
                    }
                    int P_a = computeProbability(idx_a, label_a, label_b, prior_a, prior_b);
                    if (P_a && checkSpilit(idx_a))
                    {
                        changePixelLabel(idx_a, label_b);
                    }
                    else if (!P_a)
                    {
                        // todo check
                        int P_b = computeProbability(idx_b, label_b, label_a, prior_b, prior_a);
                        if (P_b && checkSpilit(idx_b))
                        {
                            changePixelLabel(idx_b, label_a);
                        }
                    }
                }
            }
        }
    }

    void seeds3d::seeds3d_func(vector<vector<vector<float>>> img, int num_interations)
    {
        double t = (double)getTickCount();
        initImage(img);
        for (int e = 0; e < num_interations; e++)
        {
            update();
        }
        t = ((double)getTickCount() - t) / getTickFrequency();
        cout << "seeds3d took " << t << " ms";
    }

    bool seeds3d::isConnected(const point &a, const point &b)
    {
        int diff = 0;
        if (a.x != b.x)
            diff++;
        if (a.y != b.y)
            diff++;
        if (a.z != b.z)
            diff++;
        return diff == 1;
    }

    bool seeds3d::isConnectedSpace(const vector<point> &points)
    {
        if (points.empty())
            return true;

        unordered_set<point> visited;

        queue<point> q;
        q.push(points[0]);
        visited.insert(points[0]);


        while (!q.empty())
        {
            point current = q.front();
            q.pop();

            for (const auto &neighbor : points)
            {
                if (visited.find(neighbor) == visited.end() && isConnected(current, neighbor))
                {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        return visited.size() == points.size();
    }

    bool seeds3d::checkSpilit(tuple<int, int, int> idx)
    {
        return true;
        vector<point> points;
        int x = get<0>(idx);
        int y = get<1>(idx);
        int z = get<2>(idx);
        int label = labels[x][y][z];
        for (int x = 0; x < img_width; x++)
        {
            for (int y = 0; y < img_height; y++)
            {
                for (int z = 0; z < img_depth; z++)
                {
                    if (labels[x][y][z] == label)
                    {
                        point p_tmp(x, y, z);
                        points.push_back(p_tmp);
                    }
                }
            }
        }
        return isConnectedSpace(points);
    }

    seeds3d::~seeds3d()
    {
    }
}