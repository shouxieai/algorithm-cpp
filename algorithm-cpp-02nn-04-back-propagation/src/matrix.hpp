#ifndef GEMM_HPP
#define GEMM_HPP

#include <vector>
#include <initializer_list>
#include <ostream>

/* 实现一个自定义的matrix类 */
class Matrix{
public:
    Matrix();
    Matrix(int rows, int cols, const std::initializer_list<float>& pdata={});

    /* 操作符重载，使得支持()操作 */
    const float& operator()(int irow, int icol)const {return data_[irow * cols_ + icol];}
    float& operator()(int irow, int icol){return data_[irow * cols_ + icol];}
    Matrix operator*(float value) const;
    Matrix operator+(float value) const;
    Matrix operator+(const Matrix& value) const;
    Matrix operator-(const Matrix& value) const;
    int rows() const{return rows_;}
    int cols() const{return cols_;}
    int numel() const{return data_.size();}
    void resize(int rows, int cols);

    float* ptr(int irow = 0) const{return (float*)data_.data() + irow * cols_;}
    Matrix gemm(const Matrix& other, bool at=false, bool bt=false, float alpha=1.0f, float beta=0.0f);

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<float> data_;
};

/* 全局操作符重载，使得能够被cout << m; */
std::ostream& operator << (std::ostream& out, const Matrix& m);
Matrix operator * (float value, const Matrix& m);
Matrix operator + (float value, const Matrix& m);

/* 对gemm的封装 */
Matrix gemm(const Matrix& a, bool ta, const Matrix& b, bool tb, float alpha, float beta);

#endif // GEMM_HPP