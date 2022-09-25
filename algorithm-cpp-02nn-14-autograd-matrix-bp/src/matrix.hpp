#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <initializer_list>
#include <ostream>
#include <memory>
#include <functional>

/* 实现一个自定义的matrix类 */
struct MatrixData{
    std::vector<float> d;
    int rows = 0;
    int cols = 0;
};

enum class optype_t : int{
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3
};

class Matrix{
public:
    Matrix();
    Matrix(const std::vector<float>& values);
    Matrix(int rows, int cols, const std::vector<float>& values={});

    void                    resize(int rows, int cols);
    bool                     empty() const                {return data_->d.empty();}
    int                       rows() const                {return data_->rows;}
    int                       cols() const                {return data_->cols;}
    std::vector<float>&       value()                     {return data_->d;}
    const std::vector<float>& value() const               {return data_->d;}
    float*                    ptr()                       {return data_->d.data();}
    const float*              ptr() const                 {return data_->d.data();}
    float*                    ptr(int irow)               {return data_->d.data() + irow * data_->cols;}
    const float*              ptr(int irow) const         {return data_->d.data() + irow * data_->cols;}
    const float&              operator[](int index) const {return data_->d[index];}
    float&                    operator[](int index)       {return data_->d[index];}
    const float&              operator()(int index) const {return data_->d[index];}
    float&                    operator()(int index)       {return data_->d[index];}
    const float&              operator()(int ir, int ic) const {return data_->d[ir * cols() + ic];}
    float&                    operator()(int ir, int ic)       {return data_->d[ir * cols() + ic];}
    size_t                    numel() const               {return data_->d.size();}
    float                     item() const                {return data_->d[0];}

    Matrix copy() const;
    Matrix operator+(float other) const;
    Matrix operator-(float other) const;
    Matrix operator*(float other) const;
    Matrix operator/(float other) const;
    Matrix& operator+=(float other);
    Matrix& operator-=(float other);
    Matrix& operator*=(float other);
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
    Matrix operator*(const Matrix& other);
    Matrix operator/(const Matrix& other);
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix operator-()const;
    Matrix power(float y) const;
    Matrix sqrt() const;
    Matrix T() const;
    Matrix sigmoid() const;
    Matrix relu() const;
    Matrix gemm(const Matrix& other, bool ta = false, bool tb = false, float alpha = 1.0f, float beta = 0.0f) const;
    int argmax(int irow) const;
    void fill_(float scalar);
    Matrix reduce_sum_by_row() const;
    Matrix reduce_sum_by_col() const;
    Matrix reduce_sum_all() const;
    Matrix slice(const std::vector<int>& indexs, int begin=0, int size=-1);

    static std::tuple<Matrix*, Matrix*, int> check_broadcast(Matrix* a, Matrix* b);

private:
    void compute(Matrix* a, Matrix* b, int broadcast, optype_t op);
    void compute_scalar(float value, Matrix* pmatrix, optype_t op) const;

private:
    std::shared_ptr<MatrixData> data_;
};

/* 全局操作符重载，使得能够被cout << m; */
std::ostream& operator << (std::ostream& out, const Matrix& m);
Matrix operator * (float value, const Matrix& m);
Matrix operator + (float value, const Matrix& m);

/* 对gemm的封装 */
Matrix gemm(const Matrix& a, bool ta, const Matrix& b, bool tb, float alpha, float beta);

#endif // MATRIX_HPP