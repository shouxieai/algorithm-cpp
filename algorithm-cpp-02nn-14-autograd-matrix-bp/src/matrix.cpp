
#include "matrix.hpp"
#include "openblas/cblas.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <memory.h>

using namespace std;

Matrix::Matrix(){
    data_.reset(new MatrixData());
}

// Matrix::Matrix(float value){
//     data_.reset(new MatrixData());
//     data_->d.resize(1);
//     data_->d[0] = value;
//     data_->rows = 1; 
//     data_->cols=1;
// }

Matrix::Matrix(const std::vector<float>& values) {
    data_.reset(new MatrixData());
    data_->d = values;
    data_->rows = values.size();
    data_->cols = 1;
}

Matrix::Matrix(int rows, int cols, const std::vector<float>& values) {
    data_.reset(new MatrixData());
    data_->d = values;
    data_->rows = rows;
    data_->cols = cols;
    if(data_->d.size() != rows * cols)
        data_->d.resize(rows * cols);
}

Matrix Matrix::copy() const {
    Matrix output;
    *output.data_.get() = *this->data_.get();
    return output;
}

Matrix Matrix::operator*(float other) const{
    Matrix output = copy();
    compute_scalar(other, &output, optype_t::Mul);
    return output;
}

Matrix Matrix::operator-(float other) const{
    Matrix output = copy();
    compute_scalar(other, &output, optype_t::Sub);
    return output;
}

Matrix Matrix::operator+(float other) const{
    Matrix output = copy();
    compute_scalar(other, &output, optype_t::Add);
    return output;
}

Matrix Matrix::operator/(float other) const{
    Matrix output = copy();
    compute_scalar(other, &output, optype_t::Div);
    return output;
}

Matrix& Matrix::operator+=(float other){
    compute_scalar(other, this, optype_t::Add);
    return *this;
}

Matrix& Matrix::operator-=(float other){
    compute_scalar(other, this, optype_t::Sub);
    return *this;
}

Matrix& Matrix::operator*=(float other){
    compute_scalar(other, this, optype_t::Mul);
    return *this;
}

Matrix Matrix::operator*(const Matrix& other){
    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵
    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);

    Matrix output = a->copy();
    compute(&output, b, broadcast, optype_t::Mul);
    return output;
}

Matrix Matrix::operator/(const Matrix& other){
    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵
    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);

    Matrix output = a->copy();
    compute(&output, b, broadcast, optype_t::Div);
    return output;
}

Matrix Matrix::operator-(const Matrix& other){
    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵

    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);

    Matrix output = a->copy();
    compute(&output, b, broadcast, optype_t::Sub);
    return output;
}

Matrix Matrix::operator+(const Matrix& other){

    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵

    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);
        
    Matrix output = a->copy();
    compute(&output, b, broadcast, optype_t::Add);
    return output;
}

Matrix& Matrix::operator+=(const Matrix& other){

    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵

    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);
        
    Matrix output = *a;
    if(this != a)output = a->copy();

    compute(&output, b, broadcast, optype_t::Add);
    if(this != a)swap(output.data_, this->data_);
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other){

    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵

    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);
        
    Matrix output = *a;
    if(this != a)output = a->copy();

    compute(&output, b, broadcast, optype_t::Sub);
    if(this != a)swap(output.data_, this->data_);            
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other){

    Matrix* a = this;             // 大矩阵 
    Matrix* b = (Matrix*)&other;  // 小矩阵

    int broadcast = 0;
    tie(a, b, broadcast) = check_broadcast(a, b);

    Matrix output = *a;
    if(this != a)output = a->copy();

    compute(&output, b, broadcast, optype_t::Mul);
    if(this != a)swap(output.data_, this->data_);
    return *this;
}

Matrix Matrix::operator-()const{
    Matrix output = this->copy();
    for(int i = 0; i < output.numel(); ++i)
        output[i] = -output[i];
    return output;
}

Matrix Matrix::power(float y) const{
    Matrix output = this->copy();
    auto op = output.ptr();
    for(int i = 0; i < output.numel(); ++i, ++op)
        *op = std::pow(*op, y);
    return output;
}

Matrix Matrix::T() const{
    Matrix output = this->copy();
    swap(output.data_->rows, output.data_->cols);

    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            output(j, i) = (*this)(i, j);
    return output;
}

int Matrix::argmax(int irow) const{
    auto p = ptr(irow);
    return std::max_element(p, p + cols()) - p;
}

Matrix Matrix::relu() const{

    Matrix m = copy();
    auto p = m.ptr();
    for(int i = 0; i < m.numel(); ++i){
        float& x = *p++;
        x = std::max(0.0f, x);
    }
    return m;
}

Matrix Matrix::sigmoid() const{

    Matrix m = copy();
    auto p = m.ptr();
    for(int i = 0; i < m.numel(); ++i){
        float& x = *p++;
        // if(x < 0){
        //     x = exp(x) / (1.0f + exp(x));
        // }else{
        //     x = 1.0f / (1.0f + exp(-x));
        // }
        x = 1.0f / (1.0f + exp(-x));
    }
    return m;
}

Matrix Matrix::sqrt() const{

    Matrix m = copy();
    auto p = m.ptr();
    for(int i = 0; i < m.numel(); ++i){
        float& x = *p++;
        x = std::sqrt(x);
    }
    return m;
}

Matrix Matrix::slice(const std::vector<int>& indexs, int begin, int size){

    if(size == -1) size = indexs.size();
    Matrix out(size, this->cols());
    for(int i = 0; i < size; ++i){
        int mrow = indexs[i + begin];
        int orow = i;
        memcpy(out.ptr(orow), this->ptr(mrow), sizeof(float) * this->cols());
    }
    return out;
}

void Matrix::fill_(float scalar){

    auto p = ptr();
    for(int i = 0; i < numel(); ++i)
        p[i] = scalar;
}

Matrix Matrix::reduce_sum_by_row() const{
    Matrix out(1, cols());
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            optr[j] += *vptr++;
    return out;
}

Matrix Matrix::reduce_sum_by_col() const{
    Matrix out(rows(), 1);
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            optr[i] += *vptr++;
    return out;
}

Matrix Matrix::reduce_sum_all() const{
    Matrix out(1, 1);
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < numel(); ++i)
        *optr += *vptr++;
    return out;
}

Matrix Matrix::gemm(const Matrix& other, bool ta, bool tb, float alpha, float beta) const{

    int a_elastic_cols = ta ? this->rows() : this->cols();   /* 如果转置，则维度转过来 */
    int b_elastic_rows = tb ? other.cols() : other.rows();   /* 如果转置，则维度转过来 */

    if(a_elastic_cols != b_elastic_rows){
        printf("Invalid matrix multiply %dx%d%s -- %dx%d%s\n", 
            this->rows(), this->cols(), ta ? "^T" : "", 
            other.rows(), other.cols(), tb ? "^T" : ""
        );
        return Matrix();
    }
    return ::gemm(*this, ta, other, tb, alpha, beta);
}

tuple<Matrix*, Matrix*, int> Matrix::check_broadcast(Matrix* a, Matrix* b){
    if(a->empty() || b->empty()){
        printf("Compute operator+= for empty Matrix\n");
        return make_tuple(nullptr, nullptr, 0);
    }

    int broadcast = 0; // 0无广播，1右边列向量，2右边行向量, 3右边是标量
    if(a->numel() != b->numel()){
        if(a->numel() < b->numel()){
            // this是小矩阵，other是广播到的大矩阵
            std::swap(a, b);
        }

        if(a->numel() % b->numel() != 0){
            printf("invalid numel %dx%d * %dx%d\n", a->rows(), a->cols(), b->rows(), b->cols());
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
    // a一定是大矩阵，b一定是小矩阵
    return make_tuple(a, b, broadcast);
}

// op = 0+, =1-, =2*, =3/
void Matrix::compute(Matrix* a, Matrix* b, int broadcast, optype_t op){

    // 0无广播，1右边列向量，2右边行向量, 3右边是标量
    float* odata = a->ptr();
    float* bdata = b->ptr();
    if(broadcast == 0){
             if(op == optype_t::Add) for(int i = 0; i < a->numel(); ++i) *odata++ += *bdata++;
        else if(op == optype_t::Sub) for(int i = 0; i < a->numel(); ++i) *odata++ -= *bdata++;
        else if(op == optype_t::Mul) for(int i = 0; i < a->numel(); ++i) *odata++ *= *bdata++;
        else if(op == optype_t::Div) for(int i = 0; i < a->numel(); ++i) *odata++ /= *bdata++;
    }else if(broadcast == 1){
             if(op == optype_t::Add) for(int i = 0; i < a->rows(); ++i){float bvalue = bdata[i];for(int j = 0; j < a->cols(); ++j) *odata++ += bvalue;}
        else if(op == optype_t::Sub) for(int i = 0; i < a->rows(); ++i){float bvalue = bdata[i];for(int j = 0; j < a->cols(); ++j) *odata++ -= bvalue;}
        else if(op == optype_t::Mul) for(int i = 0; i < a->rows(); ++i){float bvalue = bdata[i];for(int j = 0; j < a->cols(); ++j) *odata++ *= bvalue;}
        else if(op == optype_t::Div) for(int i = 0; i < a->rows(); ++i){float bvalue = bdata[i];for(int j = 0; j < a->cols(); ++j) *odata++ /= bvalue;}
    }else if(broadcast == 2){
             if(op == optype_t::Add) for(int i = 0; i < a->rows(); ++i){float* bvalue = bdata;for(int j = 0; j < a->cols(); ++j) *odata++ += *bvalue++;}
        else if(op == optype_t::Sub) for(int i = 0; i < a->rows(); ++i){float* bvalue = bdata;for(int j = 0; j < a->cols(); ++j) *odata++ -= *bvalue++;}
        else if(op == optype_t::Mul) for(int i = 0; i < a->rows(); ++i){float* bvalue = bdata;for(int j = 0; j < a->cols(); ++j) *odata++ *= *bvalue++;}
        else if(op == optype_t::Div) for(int i = 0; i < a->rows(); ++i){float* bvalue = bdata;for(int j = 0; j < a->cols(); ++j) *odata++ /= *bvalue++;}
    }else if(broadcast == 3){
             if(op == optype_t::Add) for(int i = 0; i < a->numel(); ++i) *odata++ += *bdata;
        else if(op == optype_t::Sub) for(int i = 0; i < a->numel(); ++i) *odata++ -= *bdata;
        else if(op == optype_t::Mul) for(int i = 0; i < a->numel(); ++i) *odata++ *= *bdata;
        else if(op == optype_t::Div) for(int i = 0; i < a->numel(); ++i) *odata++ /= *bdata;
    }
}

void Matrix::compute_scalar(float value, Matrix* pmatrix, optype_t op) const{
    auto ptr = pmatrix->ptr();
         if(op == optype_t::Add) for(int i = 0; i < pmatrix->numel(); ++i) *ptr++ += value;
    else if(op == optype_t::Sub) for(int i = 0; i < pmatrix->numel(); ++i) *ptr++ -= value;
    else if(op == optype_t::Mul) for(int i = 0; i < pmatrix->numel(); ++i) *ptr++ *= value;
    else if(op == optype_t::Div) for(int i = 0; i < pmatrix->numel(); ++i) *ptr++ /= value;
}

Matrix operator*(float value, const Matrix& m){
    return m * value;
}

Matrix operator+(float value, const Matrix& m){
    return m + value;
}

void Matrix::resize(int rows, int cols){
    this->data_->d.resize(rows * cols);
    this->data_->rows = rows;
    this->data_->cols = cols;
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