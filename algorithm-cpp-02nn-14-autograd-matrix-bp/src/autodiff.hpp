#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP


#include <memory>
#include <vector>
#include <functional>
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

class Expression{
public:
    Expression(){};
    Expression(const Matrix& value);

    const char* type(){return data_ ? data_->type() : "nullptr";}
    Matrix forward();
    void backward();

    Expression power();
    Expression gemm(const Expression& other);
    std::shared_ptr<ExpressionContainer> data() const{return data_;}

protected:
    std::shared_ptr<ExpressionContainer> data_;
};

class Parameter : public Expression{
public:
    Parameter(float value);
    Parameter(const std::vector<float>& value);
    Parameter(const Matrix& value);
    const Matrix& gradient() const;
    const Matrix& value() const;
    Matrix& gradient();
    Matrix& value();
    operator Matrix&();
    operator const Matrix&()const;
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


#endif // AUTODIFF_HPP