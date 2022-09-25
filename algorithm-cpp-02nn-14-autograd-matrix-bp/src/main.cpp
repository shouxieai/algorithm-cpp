
#include "autodiff.hpp"
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

        class Optimizer{
        public:
            virtual void step() = 0;
            virtual void zero_grad(){
                for(auto& param : params_)
                    param.gradient().fill_(0);
            }

            virtual float lr() const{return lr_;}
            virtual void set_lr(float newlr){lr_ = newlr;}

        protected:
            vector<Parameter> params_;
            float lr_ = 0;
        };

        class SGDMomentum : public Optimizer{
        public:
            SGDMomentum(const vector<Parameter>& params, float lr, float momentum=0.9){
                params_   = params;
                lr_       = lr;
                momentum_ = momentum;
                delta_momentums_.resize(params_.size());
            }

            // 提供对应的参数params，和对应的梯度grads，进行参数的更新
            void step(){

                for(int i =0 ; i < params_.size(); ++i){
                    auto& delta_momentum = delta_momentums_[i];
                    auto& param          = params_[i].value();
                    auto& grad           = params_[i].gradient();

                    if(delta_momentum.empty())
                        delta_momentum.resize(param.rows(), param.cols());
                    
                    delta_momentum = momentum_ * delta_momentum - lr_ * grad;
                    param          = param + delta_momentum;
                }
            }
        
        private:
            vector<Matrix> delta_momentums_;
            float momentum_ = 0;
        };

        class AdamW : public Optimizer{
        public:
            AdamW(const vector<Parameter>& params, float lr=1e-3, float beta1=0.9, float beta2=0.999, float l2_regularization=0){
                params_   = params;
                lr_       = lr;
                beta1_    = beta1;
                beta2_    = beta2;
                l2_regularization_ = l2_regularization;
                m_.resize(params_.size());
                v_.resize(params_.size());
            }

            void step(){
                
                t_ ++;
                for(int i = 0; i < params_.size(); ++i){
                    auto& m      = m_[i];
                    auto& v      = v_[i];
                    auto& param  = params_[i].value();
                    auto& grad   = params_[i].gradient();

                    if(m.empty())
                        m.resize(param.rows(), param.cols());

                    if(v.empty())
                        v.resize(param.rows(), param.cols());
                    
                    m = beta1_ * m + (1 - beta1_) * grad;
                    v = beta2_ * v + (1 - beta2_) * grad.power(2.0f);
                    auto mt = m / (1 - std::pow(beta1_, t_));
                    auto vt = v / (1 - std::pow(beta2_, t_));
                    param -= lr_ * mt / (vt.sqrt() + eps_) + l2_regularization_ * param;
                }
            }

        private:
            vector<Matrix> m_, v_;
            float t_ = 0;
            float eps_ = 1e-7;
            float momentum_ = 0;
            float beta1_ = 0;
            float beta2_ = 0;
            float l2_regularization_ = 0;
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
        auto input_to_hidden = Parameter(random::create_normal_distribution_matrix(num_input,  num_hidden, 0, 2.0f / sqrt((float)(num_input + num_hidden))));
        auto hidden_bias = Parameter(Matrix(1, num_hidden));
        auto hidden_to_output = Parameter(random::create_normal_distribution_matrix(num_hidden, num_output, 0, 1.0f / sqrt((float)(num_hidden + num_output))));
        auto output_bias = Parameter(Matrix(1, num_output));
        
        optimizer::SGDMomentum optim({
            input_to_hidden, hidden_bias, hidden_to_output, output_bias
        }, lr, momentum);
        // optimizer::AdamW optim({
        //     input_to_hidden, hidden_bias, hidden_to_output, output_bias
        // }, 0.01);

        int total_batch = 0;
        auto t0 = tools::timenow();
        MatMul mul;
        ReLU relu;
        SigmoidCrossEntropyLoss lossfn;
        for(int epoch = 0; epoch < num_epoch; ++epoch){

            if(epoch == 8)
                optim.set_lr(optim.lr() * 0.1);

            // 打乱索引
            std::shuffle(image_indexs.begin(), image_indexs.end(), global_random_engine);
            
            // 开始循环所有的batch
            for(int ibatch = 0; ibatch < num_batch_per_epoch; ++ibatch, ++total_batch){

                // 前向过程
                auto x           = trainimages.slice(image_indexs, ibatch * batch_size, batch_size);
                auto y           = trainlabels.slice(image_indexs, ibatch * batch_size, batch_size);
                auto hidden      = relu(mul(x, input_to_hidden) + hidden_bias);
                auto output      = hidden.gemm(hidden_to_output) + output_bias;
                auto loss        = lossfn(output, y);
                auto lossval     = loss.forward().item();

                optim.zero_grad();
                loss.backward();
                optim.step();

                if((total_batch + 1) % 50 == 0){
                    auto t1 = tools::timenow();
                    auto batchs_time = t1 - t0;
                    t0 = t1;
                    INFO("Epoch %.2f / %d, Loss: %f, LR: %f [ %.2f ms / 50 batch ]", epoch + ibatch / (float)num_batch_per_epoch, num_epoch, lossval, optim.lr(), batchs_time);
                }
            }

            //模型对测试集进行测试，并打印精度
            auto test_hidden      = (valimage.gemm(input_to_hidden) + hidden_bias.value()).relu();
            auto test_probability = (test_hidden.gemm(hidden_to_output) + output_bias.value()).sigmoid();
            float accuracy        = nn::eval_test_accuracy(test_probability, vallabels);
            float test_loss       = lossfn(test_probability, vallabels).forward().item();
            INFO("Test Accuracy: %.2f %%, Loss: %f", accuracy * 100, test_loss);
        }

        INFO("Save to model.bin .");
        io::save_model("model.bin", {input_to_hidden, hidden_bias, hidden_to_output, output_bias});

        for(int i = 0; i < valimage.rows(); ++i){

            auto input            = valimage.slice({i});
            auto test_hidden      = (input.gemm(input_to_hidden) + hidden_bias.value()).relu();
            auto test_probability = (test_hidden.gemm(hidden_to_output) + output_bias.value()).sigmoid();

            int ilabel = test_probability.argmax(0);
            float prob = test_probability(0, ilabel);

            io::print_image(input.ptr(), 28, 28);
            INFO("Predict %d, Confidence = %f", ilabel, prob);

            printf("Prass [Enter] to next.");
            getchar();
        }
        return 0;
    }
};

int main(){
    return Application::run();
}