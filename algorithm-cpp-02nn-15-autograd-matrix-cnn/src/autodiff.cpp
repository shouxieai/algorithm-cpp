#include "autodiff.hpp"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <random>
#include <iostream>

using namespace std;


default_random_engine& get_random_engine(){
    static default_random_engine global_random_engine;
    return global_random_engine;
}

Matrix create_normal_distribution_matrix(const vector<int>& shape, float mean, float stddev){

    normal_distribution<float> norm(mean, stddev);
    Matrix out(shape);
    auto& engine = get_random_engine();
    auto p = out.ptr();
    for(int i = 0; i < out.numel(); ++i)
        *p++ = norm(engine);
    return out;
}

class MatrixContainer : public ExpressionContainer{
public:
    MatrixContainer(const Matrix& value, bool requires_grad = false){value_ = value;requires_grad_=requires_grad;}
    virtual const char* type() override{return "Matrix";}
    virtual Matrix forward() override{return value_;}
    virtual void backward(const Matrix& gradient) override{if(requires_grad_) gradient_ += gradient;}
    const Matrix& gradient() const{return gradient_;}
    const Matrix& value() const{return value_;}
    Matrix& gradient(){return gradient_;}
    Matrix& value(){return value_;}

private:
    Matrix value_;
    Matrix gradient_{{1, 1}, {0}};
};


class AddContainer : public ExpressionContainer{
public:
    void assign(const Expression& left, const Expression& right);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    std::shared_ptr<ExpressionContainer> left_;
    std::shared_ptr<ExpressionContainer> right_;
    int broadcast_ = 0;
};

class SubContainer : public ExpressionContainer{
public:
    void assign(const Expression& left, const Expression& right);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    std::shared_ptr<ExpressionContainer> left_;
    std::shared_ptr<ExpressionContainer> right_;
};

class MultiplyContainer : public ExpressionContainer{
public:
    void assign(const Expression& left, const Expression& right);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix left_save_;
    Matrix right_save_;
    std::shared_ptr<ExpressionContainer> left_;
    std::shared_ptr<ExpressionContainer> right_;
};

class ViewContainer : public ExpressionContainer{
public:
    ViewContainer(const vector<int>& shape);
    void assign(const Expression& value);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    std::vector<int> shape_, old_shape_;
    std::shared_ptr<ExpressionContainer> value_;
};

class PowerContainer : public ExpressionContainer{
public:
    void assign(const Expression& value);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix value_save_;
    std::shared_ptr<ExpressionContainer> value_;
};

class MatMulContainer : public ExpressionContainer{
public:
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;
    void assign(const Expression& left, const Expression& right);

private:
    Matrix left_save_;
    Matrix right_save_;
    std::shared_ptr<ExpressionContainer> left_;
    std::shared_ptr<ExpressionContainer> right_;
};

class LinearContainer : public ExpressionContainer{
public:
    LinearContainer(int input, int output, bool bias);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;
    void assign(const Expression& x);

    Parameter& weight();
    Parameter& bias();

private:
    Matrix x_save_;
    std::shared_ptr<ExpressionContainer> x_;
    std::shared_ptr<Parameter> weight_, bias_;
    int input_ = 0;
    int output_ = 0;
    bool hasbias_ = true;
};

class Conv2dContainer : public ExpressionContainer{
public:
    Conv2dContainer(int input, int output, int ksize, int stride, int padding, bool bias);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;
    void assign(const Expression& x);

    Parameter& weight();
    Parameter& bias();

private:
    std::vector<int> x_shape_;
    Matrix column_, output_save_, grad_save_;
    std::shared_ptr<ExpressionContainer> x_;
    std::shared_ptr<Parameter> weight_, bias_;
    int input_ = 0;
    int output_ = 0;
    int ksize_ = 0;
    int stride_ = 0;
    int padding_ = 0;
    bool hasbias_ = true;
    int oh_ = 0, ow_ = 0;
};

class ReLUContainer : public ExpressionContainer{
public:
    void assign(const Expression& left);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix save_forward_;
    std::shared_ptr<ExpressionContainer> left_;
};

class SigmoidContainer : public ExpressionContainer{
public:
    void assign(const Expression& left);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix save_forward_;
    std::shared_ptr<ExpressionContainer> left_;
};

class LogContainer : public ExpressionContainer{
public:
    void assign(const Expression& left);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix save_forward_;
    std::shared_ptr<ExpressionContainer> left_;
};

class SigmoidCrossEntropyLossContainer : public ExpressionContainer{
public:
    void assign(const Expression& predict, const Expression& label);
    virtual const char* type() override;
    virtual Matrix forward() override;
    virtual void backward(const Matrix& gradient) override;

private:
    Matrix save_gradient_;
    std::shared_ptr<ExpressionContainer> predict_;
    std::shared_ptr<ExpressionContainer> label_;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Parameter::Parameter(){
    data_.reset(new MatrixContainer(Matrix(), true));
}

Parameter::Parameter(float value){
    data_.reset(new MatrixContainer(Matrix({1, 1}, {value}), true));
}

Parameter::Parameter(const vector<float>& value){
    data_.reset(new MatrixContainer(Matrix({(int)value.size(), 1}, value), true));
}

Parameter::Parameter(const Matrix& value){
    data_.reset(new MatrixContainer(value, true));
}

bool Parameter::requires_grad() const{
    return data_->requires_grad();
}

const Matrix& Parameter::gradient() const{
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->gradient();
}

const Matrix& Parameter::value() const{
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->value();
}

Matrix& Parameter::gradient(){
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->gradient();
}

Parameter::operator Matrix&(){
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->value();
}

Parameter::operator const Matrix&()const{
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->value();
}

Matrix& Parameter::value(){
    MatrixContainer* ptr = (MatrixContainer*)data_.get();
    return ptr->value();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void AddContainer::assign(const Expression& left, const Expression& right){
    left_ = left.data();
    right_ = right.data();
    requires_grad_ = left_->requires_grad() || right_->requires_grad();
}

const char* AddContainer::type(){
    return "Add";
}

Matrix AddContainer::forward(){

    auto lm = left_->forward();
    auto rm = right_->forward();
    Matrix* pa, *pb;
    tie(pa, pb, broadcast_) = Matrix::check_broadcast(&lm, &rm);
    if(pa != &lm) broadcast_ += 4;  // 0无广播，1右边列向量，2右边行向量, 3右边是标量, 4无广播，5左边列向量，6左边行向量, 7左边是标量
    return lm + rm;
}

void AddContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;

    Matrix g = gradient;
    Matrix reduce = gradient;
    if(broadcast_ == 1 || broadcast_ == 5){
        // 列向量
        reduce = g.reduce_sum_by_col();
    }else if(broadcast_ == 2 || broadcast_ == 6){
        // 行向量
        reduce = g.reduce_sum_by_row();
    }else if(broadcast_ == 3 || broadcast_ == 7){
        reduce = g.reduce_sum_all();
    }

    if(broadcast_ > 3)
        std::swap(g, reduce);
    
    if(left_->requires_grad())
        left_->backward(g);

    if(right_->requires_grad())
        right_->backward(reduce);
}

Add::Add(){
    data_.reset(new AddContainer());
}

Add& Add::operator()(const Expression& left, const Expression& right){
    std::dynamic_pointer_cast<AddContainer>(data_)->assign(left, right);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SubContainer::assign(const Expression& left, const Expression& right){
    left_ = left.data();
    right_ = right.data();
    requires_grad_ = left_->requires_grad() || right_->requires_grad();
}

const char* SubContainer::type(){
    return "Sub";
}

Matrix SubContainer::forward(){
    return left_->forward() - right_->forward();
}

void SubContainer::backward(const Matrix& gradient){

    if(left_->requires_grad())
        left_->backward(gradient);
    
    if(right_->requires_grad())
        right_->backward(-gradient);
}

Sub::Sub(){
    data_.reset(new SubContainer());
}

Sub& Sub::operator()(const Expression& left, const Expression& right){
    std::dynamic_pointer_cast<SubContainer>(data_)->assign(left, right);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiplyContainer::assign(const Expression& left, const Expression& right){
    left_ = left.data();
    right_ = right.data();
    requires_grad_ = left_->requires_grad() || right_->requires_grad();
}

const char* MultiplyContainer::type(){
    return "Multiply";
}

Matrix MultiplyContainer::forward(){
    left_save_  = left_->forward();
    right_save_ = right_->forward();
    return left_save_ * right_save_;
}

void MultiplyContainer::backward(const Matrix& gradient){

    if(left_->requires_grad())
        left_->backward(right_save_ * gradient);

    if(right_->requires_grad())
        right_->backward(left_save_ * gradient);
}

Multiply::Multiply(){
    data_.reset(new MultiplyContainer());
}

Multiply& Multiply::operator()(const Expression& left, const Expression& right){
    std::dynamic_pointer_cast<MultiplyContainer>(data_)->assign(left, right);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ViewContainer::ViewContainer(const vector<int>& shape){
    shape_ = shape;
}

void ViewContainer::assign(const Expression& value){
    value_ = value.data();
    requires_grad_ = value_->requires_grad();
}

const char* ViewContainer::type(){
    return "View";
}

Matrix ViewContainer::forward(){

    auto x = value_->forward();
    old_shape_ = x.shape();
    return x.view(shape_);
}

void ViewContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;
    value_->backward(gradient.view(old_shape_));
}

View::View(const vector<int>& shape){
    data_.reset(new ViewContainer(shape));
}

View& View::operator()(const Expression& value){
    std::dynamic_pointer_cast<ViewContainer>(data_)->assign(value);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PowerContainer::assign(const Expression& value){
    value_ = value.data();
    requires_grad_ = value_->requires_grad();
}

const char* PowerContainer::type(){
    return "Power";
}

Matrix PowerContainer::forward(){
    value_save_ = value_->forward();
    return value_save_.power(2.0f);
}

void PowerContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;
    value_->backward(value_save_ * gradient * 2.0f);
}

Power::Power(){
    data_.reset(new PowerContainer());
}

Power& Power::operator()(const Expression& value){
    std::dynamic_pointer_cast<PowerContainer>(data_)->assign(value);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MatMulContainer::assign(const Expression& left, const Expression& right){
    left_  = left.data();
    right_ = right.data();
    requires_grad_ = left_->requires_grad() || right_->requires_grad();
}

const char* MatMulContainer::type(){
    return "MatMul";
}

Matrix MatMulContainer::forward(){
    left_save_  = left_->forward();
    right_save_ = right_->forward();
    return left_save_.gemm(right_save_);
}

void MatMulContainer::backward(const Matrix& gradient){

    if(left_->requires_grad())
        left_->backward(gradient.gemm(right_save_, false, true));
    
    if(right_->requires_grad())
        right_->backward(left_save_.gemm(gradient, true));
}

MatMul::MatMul(){
    data_.reset(new MatMulContainer());
}

MatMul& MatMul::operator()(const Expression& left, const Expression& right){
    std::dynamic_pointer_cast<MatMulContainer>(data_)->assign(left, right);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LinearContainer::LinearContainer(int input, int output, bool bias){
    input_ = input;
    output_ = output;
    hasbias_ = bias;

    float fan_in_fan_out = 1.0f / sqrt((float)(input_ + output_));
    weight_.reset(new Parameter(create_normal_distribution_matrix({input_, output_}, 0, fan_in_fan_out)));
    if(hasbias_) bias_.reset(new Parameter(Matrix({1, output_})));
}

Parameter& LinearContainer::weight(){
    return *weight_.get();
}

Parameter& LinearContainer::bias(){
    return *bias_.get();
}

void LinearContainer::assign(const Expression& x){
    x_ = x.data();
    requires_grad_ = true;
}

const char* LinearContainer::type(){
    return "Linear";
}

Matrix LinearContainer::forward(){
    x_save_  = x_->forward();
    auto x = x_save_.gemm(weight_->value());
    if(hasbias_)
        x += bias_->value();
    return x;
}

void LinearContainer::backward(const Matrix& gradient){

    if(x_->requires_grad())
        x_->backward(gradient.gemm(weight_->value(), false, true));
    
    if(weight_->requires_grad())
        weight_->data()->backward(x_save_.gemm(gradient, true));

    if(hasbias_ && bias_->requires_grad())
        bias_->data()->backward(gradient.reduce_sum_by_row());
}

Parameter& Linear::weight(){
    return std::dynamic_pointer_cast<LinearContainer>(data_)->weight();
}

Parameter& Linear::bias(){
    return std::dynamic_pointer_cast<LinearContainer>(data_)->bias();
}

Linear::Linear(int input, int output, bool bias){
    data_.reset(new LinearContainer(input, output, bias));
    params_ = {this->weight(), this->bias()};
}

Linear& Linear::operator()(const Expression& x){
    std::dynamic_pointer_cast<LinearContainer>(data_)->assign(x);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Conv2dContainer::Conv2dContainer(int input, int output, int ksize, int stride, int padding, bool bias){
    input_ = input;
    output_ = output;
    ksize_ = ksize;
    stride_ = stride;
    padding_ = padding;
    hasbias_ = bias;

    float fan_in_fan_out = 2.0f / sqrt((float)(input_ + output_));
    weight_.reset(new Parameter(create_normal_distribution_matrix({output_, input_, ksize_, ksize_}, 0, fan_in_fan_out)));
    if(hasbias_) bias_.reset(new Parameter(
        create_normal_distribution_matrix({1, output_}, 0, 1)
    ));
}

Parameter& Conv2dContainer::weight(){
    return *weight_.get();
}

Parameter& Conv2dContainer::bias(){
    return *bias_.get();
}

void Conv2dContainer::assign(const Expression& x){
    x_ = x.data();
    requires_grad_ = true;
}

const char* Conv2dContainer::type(){
    return "Conv2d";
}

Matrix Conv2dContainer::forward(){

    auto xval = x_->forward();
    x_shape_  = xval.shape();

    int xb, xc, xh, xw;
    xb = x_shape_[0];
    xc = x_shape_[1];
    xh = x_shape_[2];
    xw = x_shape_[3];

    ow_ = (x_shape_[3] + padding_ * 2 - ksize_) / stride_ + 1;
    oh_ = (x_shape_[2] + padding_ * 2 - ksize_) / stride_ + 1;
    int col_w = ow_ * oh_;
    int col_h = ksize_ * ksize_ * input_;
    column_.resize({xb, col_h, col_w});
    output_save_.resize({xb, output_, oh_, ow_});

    for(int ib = 0; ib < xb; ++ib){
        auto subcolumn = column_.reference_d0(ib);
        for(int ic = 0; ic < xc; ++ic){
            for(int oy = 0; oy < oh_; ++oy){
                for(int ox = 0; ox < ow_; ++ox){
                    int column_x = ox + oy * ow_;
                    for(int ky = 0; ky < ksize_; ++ky){
                        int column_idx =  (ky + ic * ksize_) * ksize_;
                        int iy = oy * stride_ + ky - padding_;
                        if(iy < 0 || iy >= xh) continue;

                        for(int kx = 0; kx < ksize_; ++kx){
                            int column_y = column_idx + kx;
                            int ix = ox * stride_ + kx - padding_;
                            if(ix < 0 || ix >= xw) continue;
                            subcolumn(column_y, column_x) = xval(ib, ic, iy, ix);
                        }
                    }
                }
            }
        }

        auto bout  = weight_->value().view({output_, -1}).gemm(subcolumn).view({output_, oh_, ow_});
        auto pout  = bout.ptr();
        auto osptr = output_save_.ptr(ib);

        if(hasbias_){
            auto pbias = this->bias_->value().ptr();
            for(int ot = 0; ot < output_; ++ot, ++pbias){
                for(int j = 0; j < ow_ * oh_; ++j)
                    *osptr++ = *pout++ + *pbias;
            }
        }else{
            memcpy(osptr, pout, sizeof(float) * output_ * oh_ * ow_);
        }
    }
    return output_save_;
}

void Conv2dContainer::backward(const Matrix& gradient){

    int xb, xc, xh, xw;
    xb = x_shape_[0];
    xc = x_shape_[1];
    xh = x_shape_[2];
    xw = x_shape_[3];

    if(weight_->requires_grad()){
        for(int ib = 0; ib < xb; ++ib){
            auto subcolumn = column_.reference_d0(ib);
            Matrix g_ib = gradient.reference_d0(ib).view({output_, -1});
            auto grad = g_ib.gemm(subcolumn, false, true).view(weight_->value().shape());
            weight_->data()->backward(grad);
        }
    }

    if(hasbias_ && bias_->requires_grad()){
        
        Matrix bias_grad({output_});
        auto po = bias_grad.ptr();
        auto pg = gradient.ptr();
        size_t area = gradient.count_of(2);
        for(int i = 0; i < gradient.size(0); ++i){
            for(int ic = 0; ic < gradient.size(1); ++ic){
                auto& bias_value = po[ic];
                for(int j = 0; j < area; ++j)
                    bias_value += *pg++;
            }
        }
        bias_->data()->backward(bias_grad);
    }

    if(x_->requires_grad()){

        grad_save_.resize(x_shape_);
        grad_save_.fill_zero_();
        
        auto kcol = this->weight().value().view({output_, -1});
        for(int ib = 0; ib < xb; ++ib){

            Matrix g_ib = gradient.reference_d0(ib).view({output_, -1});
            auto dcolumn = kcol.gemm(g_ib, true);

            for(int ic = 0; ic < xc; ++ic){
                for(int oy = 0; oy < oh_; ++oy){
                    for(int ox = 0; ox < ow_; ++ox){
                        int column_x = ox + oy * ow_;
                        for(int ky = 0; ky < ksize_; ++ky){
                            int column_idx =  (ky + ic * ksize_) * ksize_;
                            int iy = oy * stride_ + ky - padding_;
                            if(iy < 0 || iy >= xh) continue;

                            for(int kx = 0; kx < ksize_; ++kx){
                                int column_y = column_idx + kx;
                                int ix = ox * stride_ + kx - padding_;

                                if(ix < 0 || ix >= xw) continue;
                                grad_save_(ib, ic, iy, ix) += dcolumn(column_y, column_x);
                            }
                        }
                    }
                }
            }
        }
        x_->backward(grad_save_);
    }
}

Parameter& Conv2d::weight(){
    return std::dynamic_pointer_cast<Conv2dContainer>(data_)->weight();
}

Parameter& Conv2d::bias(){
    return std::dynamic_pointer_cast<Conv2dContainer>(data_)->bias();
}

Conv2d::Conv2d(int input, int output, int ksize, int stride, int padding, bool bias){
    data_.reset(new Conv2dContainer(input, output, ksize, stride, padding, bias));
    params_ = {this->weight(), this->bias()};
}

Conv2d& Conv2d::operator()(const Expression& x){
    std::dynamic_pointer_cast<Conv2dContainer>(data_)->assign(x);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReLUContainer::assign(const Expression& left){
    left_ = left.data();
    requires_grad_ = left_->requires_grad();
}

const char* ReLUContainer::type(){
    return "ReLU";
}

Matrix ReLUContainer::forward(){

    save_forward_ = left_->forward().copy();

    auto p = save_forward_.ptr();
    for(int i = 0; i < save_forward_.numel(); ++i, ++p)
        *p = std::max(0.0f, *p);
    return save_forward_;
}

void ReLUContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;
    Matrix out = gradient.copy();
    auto psave = save_forward_.ptr();
    auto pout  = out.ptr();
    for(int i = 0; i < save_forward_.numel(); ++i){
        if(psave[i] <= 0)
            pout[i] = 0;
    }
    left_->backward(out);
}

ReLU::ReLU(){
    data_.reset(new ReLUContainer());
}

ReLU& ReLU::operator()(const Expression& left){
    std::dynamic_pointer_cast<ReLUContainer>(data_)->assign(left);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SigmoidContainer::assign(const Expression& left){
    left_ = left.data();
    requires_grad_ = left_->requires_grad();
}

Matrix SigmoidContainer::forward(){

    save_forward_ = left_->forward().copy();
    for(int i = 0; i < save_forward_.numel(); ++i){
        auto& x = save_forward_.ptr()[i];
        // if(x < 0){
        //     x = exp(x) / (1.0f + exp(x));
        // }else{
        //     x = 1.0f / (1.0f + exp(-x));
        // }
        x = 1.0f / (1.0f + exp(-x));
    }
    return save_forward_;
}

void SigmoidContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;
    Matrix out = gradient.copy();
    for(int i = 0; i < save_forward_.numel(); ++i){
        auto& x = save_forward_.ptr()[i];
        out.ptr()[i] *= x * (1 - x);
    }
    left_->backward(out);
}

const char* SigmoidContainer::type(){
    return "Sigmoid";
}

Sigmoid::Sigmoid(){
    data_.reset(new SigmoidContainer());
}

Sigmoid& Sigmoid::operator()(const Expression& left){
    std::dynamic_pointer_cast<SigmoidContainer>(data_)->assign(left);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LogContainer::assign(const Expression& left){
    left_ = left.data();
    requires_grad_ = left_->requires_grad();
}

Matrix LogContainer::forward(){

    save_forward_ = left_->forward().copy();
    for(int i = 0; i < save_forward_.numel(); ++i){
        auto& x = save_forward_.ptr()[i];
        x = log(x);
    }
    return save_forward_;
}

void LogContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;

    Matrix out = gradient.copy();
    for(int i = 0; i < save_forward_.numel(); ++i){
        auto& x = save_forward_.ptr()[i];
        out.ptr()[i] /= x;
    }
    left_->backward(out);
}

const char* LogContainer::type(){
    return "Log";
}

Log::Log(){
    data_.reset(new LogContainer());
}

Log& Log::operator()(const Expression& left){
    std::dynamic_pointer_cast<LogContainer>(data_)->assign(left);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SigmoidCrossEntropyLossContainer::assign(const Expression& predict, const Expression& label){
    predict_ = predict.data();
    label_ = label.data();
    requires_grad_ = predict_->requires_grad() || label_->requires_grad();
}

Matrix SigmoidCrossEntropyLossContainer::forward(){

    auto fprob = predict_->forward().sigmoid();
    auto flabel = label_->forward();

    if(requires_grad_)
        save_gradient_ = (fprob - flabel) * (1.0f / fprob.size(0));
    
    float eps = 1e-5;
    float sum_loss  = 0;
    auto pred_ptr   = fprob.ptr();
    auto onehot_ptr = flabel.ptr();
    int numel       = fprob.numel();
    for(int i = 0; i < numel; ++i, ++pred_ptr, ++onehot_ptr){
        auto y = *onehot_ptr;
        auto p = *pred_ptr;
        p = max(min(p, 1 - eps), eps);
        sum_loss += -(y * log(p) + (1 - y) * log(1 - p));
    }
    return Matrix({1, 1}, {sum_loss / fprob.size(0)});
}

void SigmoidCrossEntropyLossContainer::backward(const Matrix& gradient){

    if(!requires_grad_) return;
    predict_->backward(save_gradient_ * gradient);
}

const char* SigmoidCrossEntropyLossContainer::type(){
    return "SigmoidCrossEntropyLoss";
}

SigmoidCrossEntropyLoss::SigmoidCrossEntropyLoss(){
    data_.reset(new SigmoidCrossEntropyLossContainer());
}

SigmoidCrossEntropyLoss& SigmoidCrossEntropyLoss::operator()(const Expression& predict, const Expression& label){
    std::dynamic_pointer_cast<SigmoidCrossEntropyLossContainer>(data_)->assign(predict, label);
    return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Expression operator+(const Expression& a, const Expression& b){
    return Add()(a, b);
}

Expression operator+(float a, const Expression& b){
    return Add()(Matrix({1, 1}, {a}), b);
}

Expression operator+(const Expression& a, float b){
    return Add()(a, Matrix({1, 1}, {b}));
}

Expression operator-(const Expression& a, const Expression& b){
    return Sub()(a, b);
}

Expression operator-(float a, const Expression& b){
    return Sub()(Matrix({1, 1}, {a}), b);
}

Expression operator-(const Expression& a, float b){
    return Sub()(a, Matrix({1, 1}, {b}));
}

Expression operator*(const Expression& a, const Expression& b){
    return Multiply()(a, b);
}

Expression operator*(float a, const Expression& b){
    return Multiply()(Matrix({1, 1}, {a}), b);
}

Expression operator*(const Expression& a, float b){
    return Multiply()(a, Matrix({1, 1}, {b}));
}

Expression Expression::view(const std::vector<int>& shape){
    return View(shape)(*this);
}

Expression Expression::power(){
    return Power()(*this);
}

Expression Expression::gemm(const Expression& other){
    return MatMul()(*this, other);
}

Expression::Expression(const Matrix& value){
    data_.reset(new MatrixContainer(value, false));
}

Matrix Expression::forward(){
    auto output = data_->forward();
    forward_output_shape_ = output.shape();
    return output;
}

void Expression::backward(){

    Matrix grad(forward_output_shape_);
    grad.fill_(1);
    data_->backward(grad);
}
