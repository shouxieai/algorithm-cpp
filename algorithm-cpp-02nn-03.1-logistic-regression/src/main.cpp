#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <iomanip>
#include <stdarg.h>

using namespace std;

namespace Application{

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
    };

    namespace nn{

        double sigmoid(double x){
            return 1 / (1 + std::exp(-x));
        }
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
        
        float k_area = 0.1;
        float k_dist = 0.1;
        float b = 0;
        float lr = 0.1;

        for(int iter = 0; iter < 1000; ++iter){

            float loss = 0;
            float delta_k_area = 0;
            float delta_k_dist = 0;
            float delta_b      = 0;

            for(auto& item : datas){
                float predict = k_area * item.area + k_dist * item.distance + b;
                double logistic = nn::sigmoid(predict);
                float L = -(std::log(logistic) * item.label + std::log(1 - logistic) * (1 - item.label));

                delta_k_area += (logistic - item.label) * item.area;
                delta_k_dist += (logistic - item.label) * item.distance;
                delta_b      += (logistic - item.label);
                loss += L;
            }

            if(iter % 100 == 0)
                cout << "Iter " << iter << ", Loss: " << setprecision(3) << loss << endl;

            k_area = k_area - lr * delta_k_area;
            k_dist = k_dist - lr * delta_k_dist;
            b      = b      - lr * delta_b;
        }
    
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
        float predict       = k_area * norm_area + k_dist * norm_dist + b;
        float logistic      = nn::sigmoid(predict);

        INFO("在上海，对于房屋面积 %.0f 平方米，距离地铁 %.0f 米的住户而言，他们觉得生活是 【%s】 的.",
            area, distance, logistic > 0.5 ? "幸福" : "并不幸福"
        );
        return 0;
    }
};

int main(){
    return Application::run();
}