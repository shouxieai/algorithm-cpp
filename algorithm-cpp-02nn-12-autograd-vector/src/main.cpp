
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace std;

class Tensor{
public:
    Tensor(){}
    Tensor(float value)                      {value_.resize(1);value_[0] = value;}
    Tensor(const std::vector<float>& values) {value_ = values;}

    std::vector<float>&       value()                     {return value_;}
    const std::vector<float>& value() const               {return value_;}
    const float&              operator[](int index) const {return value_[index];}
    float&                    operator[](int index)       {return value_[index];}
    size_t                    size() const                {return value_.size();}

    Tensor operator*(const Tensor& other){
        Tensor output(this->value());
        for(int i = 0; i < size(); ++i)
            output[i] *= other.size() == 1 ? other[0] : other[i];
        return output;
    }

    Tensor operator-(const Tensor& other){
        Tensor output(this->value());
        for(int i = 0; i < size(); ++i)
            output[i] -= other.size() == 1 ? other[0] : other[i];
        return output;
    }

    Tensor operator+(const Tensor& other){

        Tensor output(this->value());
        for(int i = 0; i < size(); ++i)
            output[i] += other.size() == 1 ? other[0] : other[i];
        return output;
    }

    Tensor& operator+=(const Tensor& other){

        if(value_.empty()){
            value_ = other.value_;
            return *this;
        }

        for(int i = 0; i < size(); ++i)
            value_[i] += other.size() == 1 ? other[0] : other[i];
        return *this;
    }

    Tensor& operator-=(const Tensor& other){

        if(value_.empty()){
            value_.resize(other.size());
            for(int i = 0; i < other.size(); ++i)
                value_[i] = -other[i];
            return *this;
        }

        for(int i = 0; i < size(); ++i)
            value_[i] -= other.size() == 1 ? other[0] : other[i];
        return *this;
    }

    Tensor& operator*=(const Tensor& other){

        if(value_.empty()){
            value_.resize(other.size());
            return *this;
        }

        for(int i = 0; i < size(); ++i)
            value_[i] *= other.size() == 1 ? other[0] : other[i];
        return *this;
    }

    Tensor operator-()const{
        Tensor output(this->value());
        for(int i = 0; i < output.size(); ++i)
            output[i] = -output[i];
        return output;
    }

    Tensor power() const{
        Tensor output(this->value());
        for(int i = 0; i < output.size(); ++i)
            output[i] = pow(output[i], 2.0f);
        return output;
    }

private:
    std::vector<float> value_;
};

class ExpressionContainer{
public:
    virtual const char* type() = 0;
    virtual Tensor forward() = 0;
    virtual void backward(const Tensor& gradient) = 0;
};

class Expression{
public:
    Expression(){};
    Expression(const Tensor& value);

    const char* type(){return data_ ? data_->type() : "nullptr";}
    Tensor forward();
    void backward();
    Expression power();
    shared_ptr<ExpressionContainer> data() const{return data_;}

protected:
    shared_ptr<ExpressionContainer> data_;
};

class TensorContainer : public ExpressionContainer{
public:
    TensorContainer(const Tensor& value){
        value_ = value;
    }

    virtual const char* type() override{
        return "Tensor";
    }

    virtual Tensor forward() override{
        return value_;
    }

    virtual void backward(const Tensor& gradient) override{
        gradient_ += gradient;
    }

    const Tensor& gradient() const{
        return gradient_;
    }

    const Tensor& value() const{
        return value_;
    }

private:
    Tensor value_;
    Tensor gradient_;
};

class Variable : public Expression{
public:
    Variable(float value){
        data_.reset(new TensorContainer(value));
    }

    Variable(const vector<float>& value){
        data_.reset(new TensorContainer(value));
    }

    Variable(const Tensor& value){
        data_.reset(new TensorContainer(value));
    }

    const Tensor& gradient() const{
        TensorContainer* ptr = (TensorContainer*)data_.get();
        return ptr->gradient();
    }

    const Tensor& value() const{
        TensorContainer* ptr = (TensorContainer*)data_.get();
        return ptr->value();
    }
};

class AddContainer : public ExpressionContainer{
public:
    AddContainer(const Expression& left, const Expression& right){
        left_ = left.data();
        right_ = right.data();
    }

    virtual const char* type() override{
        return "Add";
    }

    virtual Tensor forward() override{
        return left_->forward() + right_->forward();
    }

    virtual void backward(const Tensor& gradient) override{
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
        left_ = left.data();
        right_ = right.data();
    }

    virtual const char* type() override{
        return "Sub";
    }

    virtual Tensor forward() override{
        return left_->forward() - right_->forward();
    }

    virtual void backward(const Tensor& gradient) override{
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
        left_ = left.data();
        right_ = right.data();
    }

    virtual const char* type() override{
        return "Multiply";
    }

    virtual Tensor forward() override{
        return left_->forward() * right_->forward();
    }

    virtual void backward(const Tensor& gradient) override{
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
        value_ = value.data();
    }

    virtual const char* type() override{
        return "Power";
    }

    virtual Tensor forward() override{
        return value_->forward().power();
    }

    virtual void backward(const Tensor& gradient) override{
        value_->backward(value_->forward() * gradient * Tensor(2.0f));
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
    return Add(Variable(a), b);
}

Expression operator+(const Expression& a, float b){
    return Add(a, Variable(b));
}

Expression operator-(const Expression& a, const Expression& b){
    return Sub(a, b);
}

Expression operator-(float a, const Expression& b){
    return Sub(Variable(a), b);
}

Expression operator-(const Expression& a, float b){
    return Sub(a, Variable(b));
}

Expression operator*(const Expression& a, const Expression& b){
    return Multiply(a, b);
}

Expression operator*(float a, const Expression& b){
    return Multiply(Variable(a), b);
}

Expression operator*(const Expression& a, float b){
    return Multiply(a, Variable(b));
}

Expression Expression::power(){
    return Power(*this);
}

Expression::Expression(const Tensor& value){
    data_.reset(new TensorContainer(value));
}

Tensor Expression::forward(){
    return data_->forward();
}

void Expression::backward(){
    data_->backward(1.0f);
}

int main(){

    Variable a({1, 2, 5});
    auto c = (a * Tensor({2, 7, 1})).power() * 1.5;
    auto t = c.forward();
    for(int i = 0; i < t.size(); ++i){
        printf("%f\n", t[i]);
    }

    printf("c.type = %s\n", c.type());
    c.backward();

    for(int i = 0; i < a.gradient().size(); ++i){
        printf("gradient: %f\n", a.gradient()[i]);
    }
    return 0;
}