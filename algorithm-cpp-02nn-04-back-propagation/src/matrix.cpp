
#include "matrix.hpp"
#include "openblas/cblas.h"
#include <vector>
#include <iostream>
#include <iomanip>

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

Matrix Matrix::operator*(float value) const{
    
    Matrix m = *this;
    for(int i = 0; i < data_.size(); ++i)
        m.data_[i] *= value;
    return m;
}

Matrix Matrix::operator+(float value) const{
    
    Matrix m = *this;
    for(int i = 0; i < data_.size(); ++i)
        m.data_[i] += value;
    return m;
}

Matrix Matrix::operator+(const Matrix& value) const{

    if(value.numel() != this->numel()){
        if(this->numel() % value.numel() != 0 || value.numel() > this->numel()){
            printf("Invalid operator + [%d x %d] + [%d x %d]\n", rows(), cols(), value.rows(), value.cols());
            return *this;
        }
    }

    Matrix m = *this;
    for(int i = 0; i < data_.size(); ++i)
        m.data_[i] += value.data_[i % value.numel()];
    return m;
}

Matrix Matrix::operator-(const Matrix& value) const{

    if(value.numel() != this->numel()){
        printf("Invalid operator -\n");
        return *this;
    }

    Matrix m = *this;
    for(int i = 0; i < data_.size(); ++i)
        m.data_[i] -= value.data_[i];
    return m;
}

void Matrix::resize(int rows, int cols){
    this->rows_ = rows;
    this->cols_ = cols;
    this->data_.resize(rows * cols);
}

Matrix operator*(float value, const Matrix& m){
    return m * value;
}

Matrix operator+(float value, const Matrix& m){
    return m + value;
}

std::ostream& operator << (std::ostream& out, const Matrix& m){

    for(int i = 0; i < m.rows(); ++i){
        for(int j = 0; j < m.cols(); ++j){
            if(j > 10){
                out << " ... ";
                break;
            }
            out << m(i, j) << "\t";
        }

        if(i > 10){
            out << "\n ... \n";
            break;
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