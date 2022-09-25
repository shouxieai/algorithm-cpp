
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <memory>

using namespace std;

class ExpressionContainer{
public:
    virtual const char* type() = 0;
    virtual float forward() = 0;
    virtual void backward(float gradient) = 0;
};

class Expression{
public:

    operator float(){
        return forward();
    }

    float forward(){
        return data_->forward();
    }

    void backward(){
        data_->backward(1.0f);
    }

    Expression power();

    shared_ptr<ExpressionContainer> data_;
};

class ScalarContainer : public ExpressionContainer{
public:
    ScalarContainer(float value){
        value_ = value;
    }

    virtual const char* type() override{
        return "Scalar";
    }

    virtual float forward() override{
        return value_;
    }

    virtual void backward(float gradient) override{
        gradient_ += gradient;
    }

    float value_ = 0;
    float gradient_ = 0;
};

class Scalar : public Expression{
public:
    Scalar(float value){
        data_.reset(new ScalarContainer(value));
    }
    
    float gradient(){
        return ((ScalarContainer*)data_.get())->gradient_;
    }
};

class AddContainer : public ExpressionContainer{
public:
    AddContainer(const Expression& left, const Expression& right){
        left_ = left.data_;
        right_ = right.data_;
    }

    virtual const char* type() override{
        return "Add";
    }

    virtual float forward() override{
        return left_->forward() + right_->forward();
    }

    virtual void backward(float gradient) override{
        left_->backward(gradient);
        right_->backward(gradient);
    }

    shared_ptr<ExpressionContainer> left_;
    shared_ptr<ExpressionContainer> right_;
};

class Add : public Expression{
public:
    Add(const Expression& left, const Expression& right){
        data_.reset(new AddContainer(left, right));
    }
};

class SubContainer : public ExpressionContainer{
public:
    SubContainer(const Expression& left, const Expression& right){
        left_ = left.data_;
        right_ = right.data_;
    }

    virtual const char* type() override{
        return "Sub";
    }

    virtual float forward() override{
        return left_->forward() - right_->forward();
    }

    virtual void backward(float gradient) override{
        left_->backward(gradient);
        right_->backward(-gradient);
    }

    shared_ptr<ExpressionContainer> left_;
    shared_ptr<ExpressionContainer> right_;
};

class Sub : public Expression{
public:
    Sub(const Expression& left, const Expression& right){
        data_.reset(new SubContainer(left, right));
    }
};

class MultiplyContainer : public ExpressionContainer{
public:
    MultiplyContainer(const Expression& left, const Expression& right){
        left_ = left.data_;
        right_ = right.data_;
    }

    virtual const char* type() override{
        return "Multiply";
    }

    virtual float forward() override{
        return left_->forward() * right_->forward();
    }

    virtual void backward(float gradient) override{
        left_->backward(right_->forward() * gradient);
        right_->backward(left_->forward() * gradient);
    }

    shared_ptr<ExpressionContainer> left_;
    shared_ptr<ExpressionContainer> right_;
};

class Multiply : public Expression{
public:
    Multiply(const Expression& left, const Expression& right){
        data_.reset(new MultiplyContainer(left, right));
    }
};

class PowerContainer : public ExpressionContainer{
public:
    PowerContainer(const Expression& value){
        value_ = value.data_;
    }

    virtual const char* type() override{
        return "Power";
    }

    virtual float forward() override{
        return pow(value_->forward(), 2.0f);
    }

    virtual void backward(float gradient) override{
        value_->backward(value_->forward() * 2 * gradient);
    }

    shared_ptr<ExpressionContainer> value_;
};

class Power : public Expression{
public:
    Power(const Expression& value){
        data_.reset(new PowerContainer(value));
    }
};

Expression operator+(const Expression& a, const Expression& b){
    return Add(a, b);
}

Expression operator+(float a, const Expression& b){
    return Add(Scalar(a), b);
}

Expression operator+(const Expression& a, float b){
    return Add(a, Scalar(b));
}

Expression operator-(const Expression& a, const Expression& b){
    return Sub(a, b);
}

Expression operator-(float a, const Expression& b){
    return Sub(Scalar(a), b);
}

Expression operator-(const Expression& a, float b){
    return Sub(a, Scalar(b));
}

Expression operator*(const Expression& a, const Expression& b){
    return Multiply(a, b);
}

Expression operator*(float a, const Expression& b){
    return Multiply(Scalar(a), b);
}

Expression operator*(const Expression& a, float b){
    return Multiply(a, Scalar(b));
}

Expression Expression::power(){
    return Power(*this);
}

int main(){

    float x = 9;
    float t = x / 2.0f;
    while(pow(t*t - x, 2.0f) > 1e-5){

        Scalar a(t);
        auto value = (a.power() - x).power();
        value.backward();

        t = t - 0.01 * a.gradient();
        printf("%f\n", t);
    }
    printf("t = %f\n", t);
    return 0;
}