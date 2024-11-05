#include "seeds3d.hpp"
// #include <boost/program_options.hpp>
using namespace cv;
// namespace po = boost::program_options;

int main(int argc, char *argv[])
{

    /*
        测试文件
        三维图像数值随机 0.0f - 1.0f
        -d 三维图像维度
        -h 箱数量
        -p prior
        -i 迭代次数

    */

    vector<int> dim = {192, 192, 48};
    int num_histogram = 10, seeds_prior = 5, num_interations = 2;

    // try
    // {
    //     po::options_description desc(
    //     " seeds 3d 
    //     测试文件
    //     三维图像数值随机 0.0f - 1.0f
    //     -d 三维图像维度
    //     -h 箱数量
    //     -p prior
    //     -i 迭代次数     "

    //      );

    //     desc.add_options()
    //     ("help", "produce help message")
    //     ("d", po::value<std::vector<int>>(&dim)->default_value({192, 192, 48}), "set the dim value")
    //     ("-h", po::value<int>(&num_histogram)->default_value(10), "set the num_histogram value")
    //     ("-p", po::value<int>(&seeds_prior)->default_value(5), "set the seeds_prior value")
    //     ("-i", po::value<int>(&num_interations)->default_value(2), "set the num_interations value");

    //     po::variables_map vm;
    //     po::store(po::parse_command_line(argc, argv, desc), vm);
    //     po::notify(vm);

    //     if (vm.count("help"))
    //     {
    //         std::cout << desc << std::endl;
    //         return 0;
    //     }

    // }
    // catch (const po::error &e)
    // {
    //     std::cerr << "Error parsing options: " << e.what() << std::endl;
    //     return 1;
    // }

    int dim_a = dim[0];
    int dim_b = dim[1];
    int dim_c = dim[2];
    seeds3d a(dim_a, dim_b, dim_c, 100, 10, 5);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<std::vector<std::vector<float>>> vec(dim_a,
                                                     std::vector<std::vector<float>>(dim_b,
                                                                                     std::vector<float>(dim_c)));

    for (int i = 0; i < dim_a; ++i)
    {
        for (int j = 0; j < dim_b; ++j)
        {
            for (int k = 0; k < dim_c; ++k)
            {
                vec[i][j][k] = dis(gen);
            }
        }
    }

    a.seeds3d_func(vec, num_interations);


}
