#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <iomanip>

using namespace std;

namespace Application{

    struct Item{
        float year;
        float price;
    };

    namespace io{

        /* 通过csv文件加载数据 */
        vector<Item> load_data(const string& file){

            vector<Item> output;
            fstream ifile(file, ios::binary | ios::in);

            string line;
            getline(ifile, line);

            while(getline(ifile, line)){
                int p = line.find(",");
                Item item;
                item.year  = atof(line.substr(0, p).c_str());
                item.price = atof(line.substr(p + 1).c_str());

                //cout << item.year << ", " << item.price << endl;
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
                mean.year  += item.year;
                mean.price += item.price;
            }
            mean.year  /= items.size();
            mean.price /= items.size();

            for(auto& item : items){
                stdval.year  += std::pow(item.year - mean.year, 2.0f);
                stdval.price += std::pow(item.price - mean.price, 2.0f);;
            }
            stdval.year  = std::sqrt(stdval.year / items.size());
            stdval.price = std::sqrt(stdval.price / items.size());
            return make_tuple(mean, stdval);
        }
    };
    
    int run(){

        auto datas = io::load_data("shanghai.csv");

        Item mean, stdval;
        tie(mean, stdval) = statistics::compute_mean_std(datas);

        /* 对数据进行减去均值除以标准差，使得均值为0，标准差为1 */
        for(auto& item : datas){
            item.year  = (item.year - mean.year) / stdval.year;
            item.price = (item.price - mean.price) / stdval.price;
        }
        
        float k_identity = 0.1;
        float k_sin      = 0.1;
        float k_cos      = 0.1;
        float lambda     = 1e-5;
        float b = 0;
        float lr = 0.01;

        for(int iter = 0; iter < 1000; ++iter){

            float loss = 0;
            float delta_k_identity = 0;
            float delta_k_sin      = 0;
            float delta_k_cos      = 0;
            float delta_b          = 0;

            for(auto& item : datas){
                float predict = k_identity * item.year + k_sin * std::sin(item.year) + k_cos * std::cos(item.year) + b;
                float L = 0.5 * std::pow(predict - item.price, 2.0f) + lambda * (k_identity*k_identity + k_sin*k_sin + k_cos*k_cos + b*b);

                delta_k_identity += (predict - item.price) * item.year + k_identity * lambda;
                delta_k_sin      += (predict - item.price) * std::sin(item.year) + k_sin * lambda;
                delta_k_cos      += (predict - item.price) * std::cos(item.year) + k_cos * lambda;
                delta_b          += (predict - item.price) + b * lambda;

                loss += L;
            }

            if(iter % 100 == 0)
                cout << "Iter " << iter << ", Loss: " << setprecision(3) << loss << endl;

            k_identity = k_identity - lr * delta_k_identity;
            k_sin      = k_sin      - lr * delta_k_sin;
            k_cos      = k_cos      - lr * delta_k_cos;
            b          = b          - lr * delta_b;
        }
    
        printf(
            "模型参数：k_identity = %f, k_sin = %f, k_cos = %f, b = %f\n"
            "数据集：xm = %f, ym = %f, xs = %f, ys = %f\n", 
            k_identity,  k_sin, k_cos, b, mean.year, mean.price, stdval.year, stdval.price
        );

        printf("参数：\nk1, k2, k3, b, xm, ym, xstd, ystd = %f, %f, %f, %f, %f, %f, %f, %f\n", k_identity,  k_sin, k_cos, b, mean.year, mean.price, stdval.year, stdval.price);

        float year          = 2023;
        float x             = (year - mean.year) / stdval.year;
        float predict       = x * k_identity + std::sin(x) * k_sin + std::cos(x) * k_cos + b;
        float predict_price = predict * stdval.price + mean.price;

        printf("预计 %d 年，上海房价将会是：%.3f 元\n", (int)year, predict_price);
        return 0;
    }
};

int main(){
    return Application::run();
}