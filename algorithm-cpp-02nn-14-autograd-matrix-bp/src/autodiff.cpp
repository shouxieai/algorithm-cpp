#include "autodiff.hpp"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>

using namespace std;


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
    Matrix gradient_{{0}};
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

Parameter::Parameter(float value){
    data_.reset(new MatrixContainer(Matrix({value}), true));
}

Parameter::Parameter(const vector<float>& value){
    data_.reset(new MatrixContainer(value, true));
}

Parameter::Parameter(const Matrix& value){
    data_.reset(new MatrixContainer(value, true));
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
        save_gradient_ = (fprob - flabel) * (1.0f / fprob.rows());
    
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
    return Matrix({sum_loss / fprob.rows()});
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
    return Add()(Matrix({a}), b);
}

Expression operator+(const Expression& a, float b){
    return Add()(a, Matrix({b}));
}

Expression operator-(const Expression& a, const Expression& b){
    return Sub()(a, b);
}

Expression operator-(float a, const Expression& b){
    return Sub()(Matrix({a}), b);
}

Expression operator-(const Expression& a, float b){
    return Sub()(a, Matrix({b}));
}

Expression operator*(const Expression& a, const Expression& b){
    return Multiply()(a, b);
}

Expression operator*(float a, const Expression& b){
    return Multiply()(Matrix({a}), b);
}

Expression operator*(const Expression& a, float b){
    return Multiply()(a, Matrix({b}));
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
    return data_->forward();
}

void Expression::backward(){
    data_->backward(Matrix({1.0f}));
}
