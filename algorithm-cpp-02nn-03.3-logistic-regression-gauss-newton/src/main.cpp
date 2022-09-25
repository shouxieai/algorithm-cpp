#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <iomanip>
#include <stdarg.h>
#include <random>
#include "matrix.hpp"

using namespace std;

namespace Application{

    static default_random_engine global_random_engine;

    struct Item{
        float area;
        float distance;
        float label;
    };

    namespace logger{

        #define INFO(...)  Application::logger::__printf(__FILE__, __LINE__, __VA_ARGS__)

        void __printf(const char* file, int line, const char* fmt, ...){

            va_list vl;
            va_start(vl, fmt);

            // None   = 0,     // 无颜色配置
            // Black  = 30,    // 黑色
            // Red    = 31,    // 红色
            // Green  = 32,    // 绿色
            // Yellow = 33,    // 黄色
            // Blue   = 34,    // 蓝色
            // Rosein = 35,    // 品红
            // Cyan   = 36,    // 青色
            // White  = 37     // 白色
            /* 格式是： \e[颜色号m文字\e[0m   */
            printf("\e[32m[%s:%d]:\e[0m ", file, line);
            vprintf(fmt, vl);
            printf("\n");
        }
    };

    namespace io{

        /* 通过csv文件加载数据 */
        vector<Item> load_data(const string& file){

            vector<Item> output;
            fstream ifile(file, ios::binary | ios::in);

            string line;
            getline(ifile, line);

            while(getline(ifile, line)){
                int p0 = line.find(",");
                int p1 = line.find(",", p0 + 1);
                Item item;
                item.area     = atof(line.substr(0, p0).c_str());
                item.distance = atof(line.substr(p0+1, p1).c_str());
                item.label    = atof(line.substr(p1+1).c_str());

                // cout << item.area << ", " << item.distance << ", " << item.label << endl;
                output.emplace_back(item);
            }
            return output;
        }
    };

    namespace statistics{

        /* 计算数据的均值和标准差 */
        tuple<Item, Item> compute_mean_std(const vector<Item>& items){

            Item mean{0, 0}, stdval{0, 0};

            for(auto& item : items){
                mean.area  += item.area;
                mean.distance += item.distance;
            }
            mean.area  /= items.size();
            mean.distance /= items.size();

            for(auto& item : items){
                stdval.area  += std::pow(item.area - mean.area, 2.0f);
                stdval.distance += std::pow(item.distance - mean.distance, 2.0f);;
            }
            stdval.area  = std::sqrt(stdval.area / items.size());
            stdval.distance = std::sqrt(stdval.distance / items.size());
            return make_tuple(mean, stdval);
        }
        
        tuple<Matrix, Matrix> datas_to_matrix(const vector<Item>& datas){

            Matrix data_matrix(datas.size(), 3);   // 1, area, distance
            Matrix label_matrix(datas.size(), 1);
            for(int i = 0; i < datas.size(); ++i){

                auto& item = datas[i];
                data_matrix(i, 0) = 1;
                data_matrix(i, 1) = item.area;
                data_matrix(i, 2) = item.distance;
                label_matrix(i, 0) = item.label;
            }
            return make_tuple(data_matrix, label_matrix);
        }
    };

    namespace random{

        Matrix create_normal_distribution_matrix(int rows, int cols, float mean=0.0f, float stddev=1.0f){

            normal_distribution<float> norm(mean, stddev);
            Matrix out(rows, cols);
            auto p = out.ptr();
            for(int i = 0; i < rows * cols; ++i)
                *p++ = norm(global_random_engine);
            return out;
        }
    };

    namespace nn{
        auto sigmoid = [](float x){
            return 1 / (1 + std::exp(-x));
        };
    };
    
    int run(){

        auto datas = io::load_data("shanghai.csv");

        Item mean, stdval;
        tie(mean, stdval) = statistics::compute_mean_std(datas);

        /* 对数据进行减去均值除以标准差，使得均值为0，标准差为1 */
        for(auto& item : datas){
            item.area     = (item.area - mean.area) / stdval.area;
            item.distance = (item.distance - mean.distance) / stdval.distance;
        }

        Matrix datas_matrix, label_matrix;
        tie(datas_matrix, label_matrix) = statistics::datas_to_matrix(datas);

        // 使用元素操作，把每一个元素都设置为-1/1
        label_matrix = label_matrix.element_wise([](float v){return v == 0 ? -1 : 1;});
        //Matrix theta = random::create_normal_distribution_matrix(3, 1);
        Matrix theta(3, 1, {0, 0.1, 0.1});
        int batch_size = datas.size();
        float lr = 1;

        for(int iter = 0; iter < 10; ++iter){

            auto r = datas_matrix.gemm(theta) - label_matrix;
            auto J = datas_matrix;
            auto grad = J.gemm(J, true).inv().gemm(J, false, true).gemm(r);
            theta = theta - grad * lr;
        }

        float b      = theta(0, 0);
        float k_area = theta(1, 0);
        float k_dist = theta(2, 0);

        INFO("模型参数：k_area = %f, k_dist = %f, b = %f", k_area, k_dist, b);
        INFO("数据集：area_mean = %f, dist_mean = %f, area_std = %f, dist_std = %f", 
            mean.area, mean.distance, stdval.area, stdval.distance
        );

        INFO(""
            "k_area = %f\n"
            "k_distance = %f\n"
            "b = %f\n"
            "area_mean = %f\n"
            "area_std = %f\n"
            "distance_mean = %f\n"
            "distance_std = %f"
        , k_area, k_dist, b, mean.area, stdval.area, mean.distance, stdval.distance);

        float area          = 100;
        float distance      = 2000;
        float norm_area     = (area - mean.area) / stdval.area;
        float norm_dist     = (distance - mean.distance) / stdval.distance;
        float logistic      = k_area * norm_area + k_dist * norm_dist + b;
        
        INFO("logistic = %f", logistic);
        INFO("在上海，对于房屋面积 %.0f 平方米，距离地铁 %.0f 米的住户而言，他们觉得生活是 【%s】 的.",
            area, distance, logistic >= 0 ? "幸福" : "并不幸福"
        );
        return 0;
    }
};

int main(){
    return Application::run();
}