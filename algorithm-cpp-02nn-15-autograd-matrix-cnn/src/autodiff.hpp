#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP


#include <memory>
#include <vector>
#include <functional>
#include <random>
#include "matrix.hpp"

class ExpressionContainer{
public:
    virtual const char* type() = 0;
    virtual Matrix forward() = 0;
    virtual void backward(const Matrix& gradient) = 0;
    virtual bool requires_grad() const{return requires_grad_;}

protected:
    bool requires_grad_ = false;
};

class Parameter;
class Expression{
public:
    Expression(){};
    Expression(const Matrix& value);

    const char* type(){return data_ ? data_->type() : "nullptr";}
    Matrix forward();
    void backward();

    Expression view(const std::vector<int>& shape);
    Expression power();
    Expression gemm(const Expression& other);
    std::shared_ptr<ExpressionContainer> data() const{return data_;}
    std::vector<Parameter> params() const{return params_;};

protected:
    std::shared_ptr<ExpressionContainer> data_;
    std::vector<int> forward_output_shape_;
    std::vector<Parameter> params_;
};

class Parameter : public Expression{
public:
    Parameter();
    Parameter(float value);
    Parameter(const std::vector<float>& value);
    Parameter(const Matrix& value);
    const Matrix& gradient() const;
    const Matrix& value() const;
    Matrix& gradient();
    Matrix& value();
    operator Matrix&();
    operator const Matrix&()const;
    virtual bool requires_grad() const;
};

class Add : public Expression{
public:
    Add();
    Add& operator()(const Expression& left, const Expression& right);
};

class Sub : public Expression{
public:
    Sub();
    Sub& operator()(const Expression& left, const Expression& right);
};

class Multiply : public Expression{
public:
    Multiply();
    Multiply& operator()(const Expression& left, const Expression& right);
};

class View : public Expression{
public:
    View(const std::vector<int>& shape);
    View& operator()(const Expression& x);
};

class Power : public Expression{
public:
    Power();
    Power& operator()(const Expression& value);
};

class MatMul : public Expression{
public:
    MatMul();
    MatMul& operator()(const Expression& left, const Expression& right);
};

class Linear : public Expression{
public:
    Linear(int input, int output, bool bias=true);
    Linear& operator()(const Expression& x);

    Parameter& weight();
    Parameter& bias();
};

class Conv2d : public Expression{
public:
    Conv2d(int input, int output, int ksize, int stride, int padding, bool bias=true);
    Conv2d& operator()(const Expression& x);

    Parameter& weight();
    Parameter& bias();
};

class ReLU : public Expression{
public:
    ReLU();
    ReLU& operator()(const Expression& left);
};

class Sigmoid : public Expression{
public:
    Sigmoid();
    Sigmoid& operator()(const Expression& left);
};

class Log : public Expression{
public: 
    Log();
    Log& operator()(const Expression& left);
};

class SigmoidCrossEntropyLoss : public Expression{
public:
    SigmoidCrossEntropyLoss();
    SigmoidCrossEntropyLoss& operator()(const Expression& predict, const Expression& label);
};

Expression operator+(const Expression& a, const Expression& b);
Expression operator+(float a, const Expression& b);
Expression operator+(const Expression& a, float b);
Expression operator-(const Expression& a, const Expression& b);
Expression operator-(float a, const Expression& b);
Expression operator-(const Expression& a, float b);
Expression operator*(const Expression& a, const Expression& b);
Expression operator*(float a, const Expression& b);
Expression operator*(const Expression& a, float b);
std::default_random_engine& get_random_engine();
Matrix create_normal_distribution_matrix(const std::vector<int>& shape, float mean=0.0f, float stddev=1.0f);

#endif // AUTODIFF_HPP