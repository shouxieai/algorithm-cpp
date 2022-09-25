#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <iomanip>
#include <stdarg.h>
#include <memory.h>
#include <random>
#include <algorithm>
#include <chrono>
#include "matrix.hpp"

using namespace std;

namespace Application{

    static default_random_engine global_random_engine;

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

        struct __attribute__((packed)) mnist_labels_header_t{
            unsigned int magic_number;
            unsigned int number_of_items;
        };

        struct __attribute__((packed)) mnist_images_header_t{
            unsigned int magic_number;
            unsigned int number_of_images;
            unsigned int number_of_rows;
            unsigned int number_of_columns;
        };

        unsigned int inverse_byte(unsigned int v){
            unsigned char* p = (unsigned char*)&v;
            std::swap(p[0], p[3]);
            std::swap(p[1], p[2]);
            return v;
        }

        /* 加载mnist数据集 */
        tuple<Matrix, Matrix> load_data(const string& image_file, const string& label_file){

            Matrix images, labels;
            fstream fimage(image_file, ios::binary | ios::in);
            fstream flabel(label_file, ios::binary | ios::in);

            mnist_images_header_t images_header;
            mnist_labels_header_t labels_header;
            fimage.read((char*)&images_header, sizeof(images_header));
            flabel.read((char*)&labels_header, sizeof(labels_header));

            images_header.number_of_images = inverse_byte(images_header.number_of_images);
            labels_header.number_of_items  = inverse_byte(labels_header.number_of_items);

            images.resize(images_header.number_of_images, 28 * 28);
            labels.resize(labels_header.number_of_items, 10);

            std::vector<unsigned char> buffer(images.rows() * images.cols());
            fimage.read((char*)buffer.data(), buffer.size());

            for(int i = 0; i < buffer.size(); ++i)
                images.ptr()[i] = (buffer[i] / 255.0f - 0.1307f) / 0.3081f;
                //images.ptr()[i] = (buffer[i] - 127.5f) / 127.5f;

            buffer.resize(labels.rows());
            flabel.read((char*)buffer.data(), buffer.size());
            for(int i = 0; i < buffer.size(); ++i)
                labels.ptr(i)[buffer[i]] = 1;   // onehot
            return make_tuple(images, labels);
        }

        void print_image(const float* ptr, int rows, int cols){

            for(int i = 0;i < rows * cols; ++i){

                //int pixel = ptr[i] * 127.5 + 127.5;
                int pixel = (ptr[i] * 0.3081f + 0.1307f) * 255.0f;
                if(pixel < 20)
                    printf("--- ");
                else
                    printf("%03d ", pixel);

                if((i + 1) % cols == 0)
                    printf("\n");
            }
        }

        bool save_model(const string& file, const vector<Matrix>& model){

            ofstream out(file, ios::binary | ios::out);
            if(!out.is_open()){
                INFO("Open %s failed.", file.c_str());
                return false;
            }

            unsigned int header_file[] = {0x3355FF11, model.size()};
            out.write((char*)header_file, sizeof(header_file));

            for(auto& m : model){
                int header[] = {m.rows(), m.cols()};
                out.write((char*)header, sizeof(header));
                out.write((char*)m.ptr(), m.numel() * sizeof(float));
            }
            return out.good();
        }

        bool load_model(const string& file, vector<Matrix>& model){

            ifstream in(file, ios::binary | ios::in);
            if(!in.is_open()){
                INFO("Open %s failed.", file.c_str());
                return false;
            }

            unsigned int header_file[2];
            in.read((char*)header_file, sizeof(header_file));

            if(header_file[0] != 0x3355FF11){
                INFO("Invalid model file: %s", file.c_str());
                return false;
            }

            model.resize(header_file[1]);
            for(int i = 0; i < model.size(); ++i){
                auto& m = model[i];
                int header[2];
                in.read((char*)header, sizeof(header));
                m.resize(header[0], header[1]);
                in.read((char*)m.ptr(), m.numel() * sizeof(float));
            }
            return in.good();
        }
    };

    namespace data{

        int argmax(float* ptr, int size){
            return std::max_element(ptr, ptr + size) - ptr;
        }

        Matrix choice_rows(const Matrix& m, const vector<int>& indexs, int begin=0, int size=-1){

            if(size == -1) size = indexs.size();
            Matrix out(size, m.cols());
            for(int i = 0; i < size; ++i){
                int mrow = indexs[i + begin];
                int orow = i;
                memcpy(out.ptr(orow), m.ptr(mrow), sizeof(float) * m.cols());
            }
            return out;
        }

        Matrix reduce_sum_by_row(const Matrix& value){
            Matrix out(1, value.cols());
            auto optr = out.ptr();
            auto vptr = value.ptr();
            for(int i = 0; i < value.numel(); ++i, ++vptr)
                optr[i % value.cols()] += *vptr;
            return out;
        }
    };

    namespace tools{

        vector<int> range(int end){
            vector<int> out(end);
            for(int i = 0; i < end; ++i)
                out[i] = i;
            return out;
        }

        double timenow(){
            return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        }
    };

    namespace nn{

        Matrix relu(const Matrix& input){
            Matrix out(input);
            for(int i = 0; i < out.numel(); ++i)
                out.ptr()[i] = std::max(0.0f, out.ptr()[i]);
            return out;
        }

        Matrix drelu(const Matrix& grad, const Matrix& x){
            Matrix out = grad;
            auto optr = out.ptr();
            auto xptr = x.ptr();
            for(int i = 0; i < out.numel(); ++i, ++optr, ++xptr){
                if(*xptr <= 0)
                    *optr = 0;
            }
            return out;
        }

        Matrix sigmoid(const Matrix& input){
            Matrix out(input);
            float eps = 1e-5;
            for(int i = 0; i < out.numel(); ++i){
                float& x = out.ptr()[i];

                /* 处理sigmoid数值稳定性问题 */
                if(x < 0){
                    x = exp(x) / (1 + exp(x));
                }else{
                    x = 1 / (1 + exp(-x));
                }

                /* 保证x不会等于0或者等于1 */
                x = std::max(0.0f + eps, std::min(1.0f - eps, x));
            }
            return out;
        }

        float compute_loss(const Matrix& probability, const Matrix& onehot_labels){

            float eps = 1e-5;
            float sum_loss  = 0;
            auto pred_ptr   = probability.ptr();
            auto onehot_ptr = onehot_labels.ptr();
            int numel       = probability.numel();
            for(int i = 0; i < numel; ++i, ++pred_ptr, ++onehot_ptr){
                auto y = *onehot_ptr;
                auto p = *pred_ptr;
                p = max(min(p, 1 - eps), eps);
                sum_loss += -(y * log(p) + (1 - y) * log(1 - p));
            }
            return sum_loss / probability.rows();
        }

        float eval_test_accuracy(const Matrix& probability, const Matrix& labels){

            int success = 0;
            for(int i = 0; i < probability.rows(); ++i){
                auto row_ptr = probability.ptr(i);
                int predict_label = std::max_element(row_ptr, row_ptr + probability.cols()) - row_ptr;
                if(labels.ptr(i)[predict_label] == 1)
                    success++;
            }
            return success / (float)probability.rows();
        }
    };

    namespace random{

        Matrix create_normal_distribution_matrix(int rows, int cols, float mean=0.0f, float stddev=1.0f){

            normal_distribution<float> norm(mean, stddev);
            Matrix out(rows, cols);
            for(int i = 0; i < rows; ++i){
                for(int j = 0; j < cols; ++j)
                    out.ptr(i)[j] = norm(global_random_engine);
            }
            return out;
        }
    };

    namespace optimizer{

        struct SGDMomentum{
            vector<Matrix> delta_momentums;

            // 提供对应的参数params，和对应的梯度grads，进行参数的更新
            void update_params(const vector<Matrix*>& params, const vector<Matrix*>& grads, float lr, float momentum=0.9){

                if(delta_momentums.size() != params.size())
                    delta_momentums.resize(params.size());

                for(int i =0 ; i < params.size(); ++i){
                    auto& delta_momentum = delta_momentums[i];
                    auto& param          = *params[i];
                    auto& grad           = *grads[i];

                    if(delta_momentum.numel() == 0)
                        delta_momentum.resize(param.rows(), param.cols());
                    
                    delta_momentum = momentum * delta_momentum - lr * grad;
                    param          = param + delta_momentum;
                }
            }
        };
    };
    
    int run(){

        Matrix trainimages, trainlabels;
        Matrix valimage, vallabels;
        tie(trainimages, trainlabels) = io::load_data("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
        tie(valimage, vallabels)      = io::load_data("mnist/t10k-images.idx3-ubyte",  "mnist/t10k-labels.idx1-ubyte");
        
        int num_images  = trainimages.rows();
        int num_input   = trainimages.cols();
        int num_hidden  = 1024;
        int num_output  = 10;
        int num_epoch   = 10;
        float lr        = 1e-1;
        int batch_size  = 256;
        float momentum  = 0.9f;
        int num_batch_per_epoch = num_images / batch_size;
        auto image_indexs       = tools::range(num_images);

        // 凯明初始化，fan_in + fan_out
        // W1 B1
        Matrix input_to_hidden  = random::create_normal_distribution_matrix(num_input,  num_hidden, 0, 2.0f / sqrt((float)(num_input + num_hidden)));
        Matrix hidden_bias(1, num_hidden);

        // W2 B2
        Matrix hidden_to_output = random::create_normal_distribution_matrix(num_hidden, num_output, 0, 1.0f / sqrt((float)(num_hidden + num_output)));
        Matrix output_bias(1, num_output);

        optimizer::SGDMomentum optim;
        auto t0 = tools::timenow();
        int total_batch = 0;
        for(int epoch = 0; epoch < num_epoch; ++epoch){

            if(epoch == 8){
                lr *= 0.1;
            }

            // 打乱索引
            // 0, 1, 2, 3, 4, 5 ... 59999
            // 199, 20, 1, 9, 10, 6, ..., 111
            std::shuffle(image_indexs.begin(), image_indexs.end(), global_random_engine);
            
            // 开始循环所有的batch
            for(int ibatch = 0; ibatch < num_batch_per_epoch; ++ibatch, ++total_batch){

                // 前向过程
                // trainimages -> X(60000, 784)
                // idx = image_indexs[0:256] -> 乱的
                // X = trainimages[idx]
                auto x           = data::choice_rows(trainimages,   image_indexs, ibatch * batch_size, batch_size);
                auto y           = data::choice_rows(trainlabels,   image_indexs, ibatch * batch_size, batch_size);
                auto hidden      = x.gemm(input_to_hidden) + hidden_bias;
                auto hidden_act  = nn::relu(hidden);
                auto output      = hidden_act.gemm(hidden_to_output) + output_bias;
                auto probability = nn::sigmoid(output);
                float loss       = nn::compute_loss(probability, y);

                // 反向过程
                // C = AB
                // dA = G * BT
                // dB = AT * G
                // loss部分求导，loss对output求导
                auto doutput           = (probability - y) * (1 / (float)batch_size);

                // 第二个Linear求导
                auto doutput_bias      = data::reduce_sum_by_row(doutput);
                auto dhidden_to_output = hidden_act.gemm(doutput, true);
                auto dhidden_act       = doutput.gemm(hidden_to_output, false, true);

                // 第一个Linear输出求导
                auto dhidden           = nn::drelu(dhidden_act, hidden);

                // 第一个Linear求导
                auto dinput_to_hidden  = x.gemm(dhidden, true);
                auto dhidden_bias      = data::reduce_sum_by_row(dhidden);

                // 调用优化器来调整更新参数
                optim.update_params(
                    {&input_to_hidden,  &hidden_bias,  &hidden_to_output,  &output_bias},
                    {&dinput_to_hidden, &dhidden_bias, &dhidden_to_output, &doutput_bias},
                    lr, momentum
                );

                if((total_batch + 1) % 50 == 0){
                    auto t1 = tools::timenow();
                    auto batchs_time = t1 - t0;
                    t0 = t1;
                    INFO("Epoch %.2f / %d, Loss: %f, LR: %f [ %.2f ms / 50 batch ]", epoch + ibatch / (float)num_batch_per_epoch, num_epoch, loss, lr, batchs_time);
                }
            }

            // 模型对测试集进行测试，并打印精度
            auto test_hidden      = nn::relu(valimage.gemm(input_to_hidden) + hidden_bias);
            auto test_probability = nn::sigmoid(test_hidden.gemm(hidden_to_output) + output_bias);
            float accuracy        = nn::eval_test_accuracy(test_probability, vallabels);
            float test_loss       = nn::compute_loss(test_probability, vallabels);
            INFO("Test Accuracy: %.2f %%, Loss: %f", accuracy * 100, test_loss);
        }

        INFO("Save to model.bin .");
        io::save_model("model.bin", {input_to_hidden, hidden_bias, hidden_to_output, output_bias});

        for(int i = 0; i < valimage.rows(); ++i){

            auto input = data::choice_rows(valimage, {i});
            auto test_hidden      = nn::relu(input.gemm(input_to_hidden) + hidden_bias);
            auto test_probability = nn::sigmoid(test_hidden.gemm(hidden_to_output) + output_bias);

            int ilabel = data::argmax(test_probability.ptr(), test_probability.cols());
            float prob = test_probability.ptr()[ilabel];

            io::print_image(input.ptr(), 28, 28);
            INFO("Predict %d, Confidence = %f", ilabel, prob);

            printf("Pass [Enter] to next.");
            getchar();
        }
        return 0;
    }
};

int main(){
    return Application::run();
}