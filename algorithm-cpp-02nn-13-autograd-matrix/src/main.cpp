
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <memory>
#include <vector>
#include <functional>

using namespace std;

struct TensorData{
    vector<float> data;
    int rows = 0;
    int cols = 0;
};

class Expression;
class Tensor{
public:
    Tensor(){
        data_.reset(new TensorData());
    }

    Tensor(float value){
        data_.reset(new TensorData());
        data_->data.resize(1);
        data_->data[0] = value;
        data_->rows = 1; 
        data_->cols=1;
    }

    Tensor(const std::vector<float>& values) {
        data_.reset(new TensorData());
        data_->data = values;
        data_->rows = values.size();
        data_->cols = 1;
    }
    
    Tensor(int rows, int cols, const std::vector<float>& values={}) {
        data_.reset(new TensorData());
        data_->data = values;
        data_->rows = rows;
        data_->cols = cols;
        if(data_->data.size() != rows * cols)
            data_->data.resize(rows * cols);
    }

    bool                     empty() const                {return data_->data.empty();}
    int                       rows() const                {return data_->rows;}
    int                       cols() const                {return data_->cols;}
    std::vector<float>&       value()                     {return data_->data;}
    const std::vector<float>& value() const               {return data_->data;}
    float*                    data()                      {return data_->data.data();}
    const float*              data() const                {return data_->data.data();}
    const float&              operator[](int index) const {return data_->data[index];}
    float&                    operator[](int index)       {return data_->data[index];}
    const float&              operator()(int index) const {return data_->data[index];}
    float&                    operator()(int index)       {return data_->data[index];}
    const float&              operator()(int ir, int ic) const {return data_->data[ir * cols() + ic];}
    float&                    operator()(int ir, int ic)       {return data_->data[ir * cols() + ic];}
    size_t                    numel() const               {return data_->data.size();}

    Tensor copy() const {
        Tensor output;
        *output.data_.get() = *this->data_.get();
        return output;
    }

    Tensor operator*(float other) const{
        Tensor output = copy();
        compute_scalar(other, &output, [](float a, float b){return a * b;});
        return output;
    }

    Tensor operator-(float other) const{
        Tensor output = copy();
        compute_scalar(other, &output, [](float a, float b){return a - b;});
        return output;
    }

    Tensor operator+(float other) const{
        Tensor output = copy();
        compute_scalar(other, &output, [](float a, float b){return a + b;});
        return output;
    }

    Tensor& operator+=(float other){
        compute_scalar(other, this, [](float a, float b){return a + b;});
        return *this;
    }

    Tensor& operator-=(float other){
        compute_scalar(other, this, [](float a, float b){return a - b;});
        return *this;
    }

    Tensor& operator*=(float other){
        compute_scalar(other, this, [](float a, float b){return a * b;});
        return *this;
    }

    Tensor operator*(const Tensor& other){
        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵
        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);

        Tensor output = a->copy();
        compute(&output, b, broadcast, [](float a, float b){return a*b;});        
        return output;
    }

    Tensor operator-(const Tensor& other){
        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵

        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);

        Tensor output = a->copy();
        compute(&output, b, broadcast, [](float a, float b){return a-b;});    
        return output;
    }

    Tensor operator+(const Tensor& other){

        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵

        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);
         
        Tensor output = a->copy();
        compute(&output, b, broadcast, [](float a, float b){return a+b;});
        return output;
    }

    Tensor& operator+=(const Tensor& other){

        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵

        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);
         
        Tensor output = *a;
        if(this != a)output = a->copy();

        compute(&output, b, broadcast, [](float a, float b){return a+b;});
        if(this != a)swap(output.data_, this->data_);
        return *this;
    }

    Tensor& operator-=(const Tensor& other){

        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵

        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);
         
        Tensor output = *a;
        if(this != a)output = a->copy();

        compute(&output, b, broadcast, [](float a, float b){return a-b;});
        if(this != a)swap(output.data_, this->data_);            
        return *this;
    }

    Tensor& operator*=(const Tensor& other){

        Tensor* a = this;             // 大矩阵 
        Tensor* b = (Tensor*)&other;  // 小矩阵

        int broadcast = 0;
        tie(a, b, broadcast) = verify_param(a, b);
         
        Tensor output = *a;
        if(this != a)output = a->copy();

        compute(&output, b, broadcast, [](float a, float b){return a*b;});
        if(this != a)swap(output.data_, this->data_);
        return *this;
    }

    Tensor operator-()const{
        Tensor output = this->copy();
        for(int i = 0; i < output.numel(); ++i)
            output[i] = -output[i];
        return output;
    }

    Tensor power() const{
        Tensor output = this->copy();
        for(int i = 0; i < output.numel(); ++i)
            output[i] = pow(output[i], 2.0f);
        return output;
    }

    Tensor T() const{
        Tensor output = this->copy();
        swap(output.data_->rows, output.data_->cols);

        for(int i = 0; i < rows(); ++i)
            for(int j = 0; j < cols(); ++j)
                output(j, i) = (*this)(i, j);
        return output;
    }

    Tensor matmul(const Tensor& other) const{
        if(this->cols() != other.rows()){
            printf("Invalid matrix multiply %dx%d -- %dx%d\n", this->rows(), this->cols(), other.rows(), other.cols());
            return Tensor();
        }

        Tensor output(this->rows(), other.cols());
        for(int i = 0; i < rows(); ++i){
            for(int j = 0; j < other.cols(); ++j){

                float value = 0;
                for(int k = 0; k < cols(); ++k)
                    value += (*this)(i, k) * other(k, j);
                
                output(i, j) = value;
            }
        }
        return output;
    }

private:
    tuple<Tensor*, Tensor*, int> verify_param(Tensor* a, Tensor* b){
        if(a->empty() || b->empty()){
            printf("Compute operator+= for empty tensor\n");
            return make_tuple(nullptr, nullptr, 0);
        }

        int broadcast = 0; // 0无广播，1右边列向量，2右边行向量, 3是标量
        if(a->numel() != b->numel()){
            if(a->numel() < b->numel()){
                // this是小矩阵，other是广播到的大矩阵
                std::swap(a, b);
            }

            if(a->numel() % b->numel() != 0){
                printf("invalid numel %dx%d * %dx%d\n", this->rows(), this->cols(), b->rows(), b->cols());
                return make_tuple(nullptr, nullptr, 0);
            }

            if(b->cols() == 1 && b->rows() == 1){
                broadcast = 3;
            }else if(b->cols() == 1){
                broadcast = 1;
            }else if(b->rows() == 1){
                broadcast = 2;
            }else{
                printf("Invalid broadcast for %d x %d\n", b->rows(), b->cols());
            }
        }
        return make_tuple(a, b, broadcast);
    }

    void compute(Tensor* a, Tensor* b, int broadcast, const function<float(float, float)>& op){

        float* odata = a->data();
        float* bdata = b->data();
        if(broadcast == 0){
            for(int i = 0; i < a->numel(); ++i)
                odata[i] = op(odata[i], bdata[i]);
        }else if(broadcast == 1){
            for(int i = 0; i < a->rows(); ++i){
                float bvalue = bdata[i];
                for(int j = 0; j < a->cols(); ++j)
                    *odata++ = op(*odata, bvalue);
            }
        }else if(broadcast == 2){
            for(int i = 0; i < a->numel(); ++i){
                float* bvalue = bdata;
                for(int j = 0; j < a->cols(); ++j)
                    *odata++ = op(*odata, *bvalue++);
            }
        }else if(broadcast == 3){
            for(int i = 0; i < a->numel(); ++i)
                odata[i] = op(odata[i], *bdata);
        }
    }

    void compute_scalar(float value, Tensor* ptensor, const function<float(float, float)>& op) const{
        for(int i = 0; i < ptensor->numel(); ++i) (*ptensor)(i) = op((*ptensor)(i), value);
    }

private:
    shared_ptr<TensorData> data_;
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
    Expression matmul(const Expression& other);
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
    Tensor gradient_{0};
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
        left_save_  = left_->forward();
        right_save_ = right_->forward();
        return left_save_ * right_save_;
    }

    virtual void backward(const Tensor& gradient) override{
        left_->backward(right_save_ * gradient);
        right_->backward(left_save_ * gradient);
    }

    Tensor left_save_;
    Tensor right_save_;
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
        value_save_ = value_->forward();
        return value_save_.power();
    }

    virtual void backward(const Tensor& gradient) override{
        value_->backward(value_save_ * gradient * Tensor(2.0f));
    }

    Tensor value_save_;
    shared_ptr<ExpressionContainer> value_;
};

class Power : public Expression{
public:
    Power(const Expression& value){
        data_.reset(new PowerContainer(value));
    }
};

class MatMulContainer : public ExpressionContainer{
public:
    MatMulContainer(const Expression& left, const Expression& right){
        left_ = left.data();
        right_ = right.data();
    }

    virtual const char* type() override{
        return "MatMul";
    }

    virtual Tensor forward() override{
        left_save_  = left_->forward();
        right_save_ = right_->forward();
        return left_save_.matmul(right_save_);
    }

    virtual void backward(const Tensor& gradient) override{
        left_->backward(gradient.matmul(right_save_.T()));
        right_->backward(left_save_.T().matmul(gradient));
    }

    Tensor left_save_;
    Tensor right_save_;
    shared_ptr<ExpressionContainer> left_;
    shared_ptr<ExpressionContainer> right_;
};

class MatMul : public Expression{
public:
    MatMul(const Expression& left, const Expression& right){
        data_.reset(new MatMulContainer(left, right));
    }
};

Expression operator+(const Expression& a, const Expression& b){
    return Add(a, b);
}

Expression operator+(float a, const Expression& b){
    return Add(Tensor(a), b);
}

Expression operator+(const Expression& a, float b){
    return Add(a, Tensor(b));
}

Expression operator-(const Expression& a, const Expression& b){
    return Sub(a, b);
}

Expression operator-(float a, const Expression& b){
    return Sub(Tensor(a), b);
}

Expression operator-(const Expression& a, float b){
    return Sub(a, Tensor(b));
}

Expression operator*(const Expression& a, const Expression& b){
    return Multiply(a, b);
}

Expression operator*(float a, const Expression& b){
    return Multiply(Tensor(a), b);
}

Expression operator*(const Expression& a, float b){
    return Multiply(a, Tensor(b));
}

Expression Expression::power(){
    return Power(*this);
}

Expression Expression::matmul(const Expression& other){
    return MatMul(*this, other);
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

    Variable a(Tensor(1, 3, {1, 2, 5}));
    auto c = 3.5f * a.matmul(Tensor(3, 1, {2, 7, 1})) + 1.5f;
    auto d = Tensor(5.0f) * 5.0f;
    auto t = c.forward();
    for(int i = 0; i < t.numel(); ++i){
        printf("%f\n", t[i]);
    }

    printf("c.type = %s\n", c.type());
    c.backward();
 
    for(int i = 0; i < a.gradient().numel(); ++i){
        printf("gradient: %f\n", a.gradient()[i]);
    }
    return 0;
}