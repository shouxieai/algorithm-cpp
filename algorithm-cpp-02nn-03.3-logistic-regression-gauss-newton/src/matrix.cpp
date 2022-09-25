
#include <vector>
#include <iostream>
#include <iomanip>
#include "openblas/cblas.h"
#include "openblas/lapacke.h"
#include "matrix.hpp"

Matrix::Matrix(){}
Matrix::Matrix(int rows, int cols, const std::initializer_list<float>& pdata){
    this->rows_ = rows;
    this->cols_ = cols;
    this->data_ = pdata;

    if(this->data_.size() < rows * cols)
        this->data_.resize(rows * cols);
}

Matrix Matrix::gemm(const Matrix& other, bool at, bool bt, float alpha, float beta){
    return ::gemm(*this, at, other, bt, alpha, beta);
}

Matrix Matrix::view(int rows, int cols) const{
    if(rows * cols != this->rows_ * this->cols_){
        printf("Invalid view to %d x %d\n", rows, cols);
        return Matrix();
    }
    Matrix newmat = *this;
    newmat.rows_ = rows;
    newmat.cols_ = cols;
    return newmat;
}

Matrix Matrix::operator+(const Matrix& other) const{
    return element_wise(other, [](float a, float b){return a + b;});
}

Matrix Matrix::operator-(const Matrix& other) const{
    return element_wise(other, [](float a, float b){return a - b;});
}

Matrix Matrix::operator*(const Matrix& other) const{
    return element_wise(other, [](float a, float b){return a * b;});
}

Matrix Matrix::sigmoid() const{
    return element_wise([](float x)->float{
        float o = 1 / (1 + exp(-x));
        if(o < 1e-5) return 1e-5;
        else if(o > 1 - 1e-5) return 1-1e-5;
        return o;
    });
}

Matrix Matrix::log() const{
    return element_wise([](float x){return ::log(x);});
}

Matrix Matrix::log_1subx() const{
    return element_wise([](float x){return ::log(1-x);});
}

Matrix Matrix::_1subx() const{
    return element_wise([](float x){return 1 - x;});
}

Matrix Matrix::element_wise(const std::function<float(float)>& func) const{
    Matrix output = *this;
    auto p0 = output.ptr();
    for(int i = 0; i < output.data_.size(); ++i, ++p0)
        *p0 = func(*p0);
    return output;
}

Matrix Matrix::element_wise(const Matrix& other, const std::function<float(float,float)>& func) const{
    Matrix output = *this;
    auto p0 = output.ptr();
    auto p1 = other.ptr();
    for(int i = 0; i < output.data_.size(); ++i, ++p0, ++p1)
        *p0 = func(*p0, *p1);
    return output;
}

float Matrix::reduce_sum() const{
    auto p0 = this->ptr();
    float output = 0;
    for(int i = 0; i < this->data_.size(); ++i)
        output += *p0++;
    return output;
}

Matrix Matrix::inv(){
    return ::inverse(*this);
}

Matrix Matrix::operator*(float value){
    
    Matrix m = *this;
    for(int i = 0; i < data_.size(); ++i)
        m.data_[i] *= value;
    return m;
}

std::ostream& operator << (std::ostream& out, const Matrix& m){

    for(int i = 0; i < m.rows(); ++i){
        for(int j = 0; j < m.cols(); ++j){
            out << m(i, j) << "\t";
        }
        out << "\n";
    }
    return out;
}

Matrix gemm(const Matrix& a, bool ta, const Matrix& b, bool tb, float alpha, float beta){

    int a_elastic_rows = ta ? a.cols() : a.rows();   /* 如果转置，则维度转过来 */
    int a_elastic_cols = ta ? a.rows() : a.cols();   /* 如果转置，则维度转过来 */
    int b_elastic_rows = tb ? b.cols() : b.rows();   /* 如果转置，则维度转过来 */
    int b_elastic_cols = tb ? b.rows() : b.cols();   /* 如果转置，则维度转过来 */

    /* c是转置后维度的行和列 */
    Matrix c(a_elastic_rows, b_elastic_cols);

    int m = a_elastic_rows;
    int n = b_elastic_cols;
    int k = a_elastic_cols;
    int lda = a.cols();
    int ldb = b.cols();
    int ldc = c.cols();

    /* cblas的gemm调用风格，在以后也会存在 */
    cblas_sgemm(
        CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
        m, n, k, alpha, a.ptr(), lda, b.ptr(), ldb, beta, c.ptr(), ldc
    );
    return c;
}

Matrix inverse(const Matrix& a){

    if(a.rows() != a.cols()){
        printf("Invalid to compute inverse matrix by %d x %d\n", a.rows(), a.cols());
        return Matrix();
    }

    Matrix output = a;
    int n = a.rows();
    int *ipiv = new int[n];

    /* LU分解 */
    int code = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, output.ptr(), n, ipiv);
    if(code == 0){
        /* 使用LU分解求解通用逆矩阵 */
        code = LAPACKE_sgetri(LAPACK_COL_MAJOR, n, output.ptr(), n, ipiv);
    }
    delete[] ipiv;

    if(code != 0){
        printf("LAPACKE inverse matrix failed, code = %d\n", code);
        return Matrix();
    }
    return output;
}