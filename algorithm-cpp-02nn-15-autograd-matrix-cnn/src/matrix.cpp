
#include "matrix.hpp"
#include "openblas/cblas.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <bits/stdc++.h>
#include <memory.h>
#include <stdio.h>

using namespace std;

struct __attribute__((packed)) matrix_header_t{
    unsigned int flag;
    unsigned int ndim;
    unsigned int dtype;
};

size_t compute_numel(const vector<int>& shape){
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void MatrixData::resize(size_t nsize){

    if(nsize > capacity_size){
        release();
        pdata = new float[nsize];
        owner = true;
        capacity_size = nsize;

        memset(pdata, 0, nsize * sizeof(float));
    }
    this->numel = nsize;
}

void MatrixData::release(){
    if(pdata && owner)
        delete[] pdata;
        
    pdata = nullptr;
    capacity_size = 0;
    numel = 0;
}

MatrixData::~MatrixData(){
    release();
}

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

Matrix::Matrix(const std::vector<int>& shape, const std::vector<float>& values) {
    data_.reset(new MatrixData());
    data_->resize(compute_numel(shape));
    shape_ = shape;

    if(data_->numel == values.size())
        memcpy(data_->pdata, values.data(), data_->numel * sizeof(float));
    else if(values.size() != 0){
        printf("Invalid values.size[%d] , shape numel[%d]\n", values.size(), data_->numel);
    }
}

Matrix Matrix::copy() const {
    Matrix output;
    output.data_->resize(compute_numel(shape_));
    memcpy(output.data_->pdata, this->data_->pdata, sizeof(float) * this->data_->numel);
    output.shape_ = shape_;
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
    if(this != a)swap(output, *this);
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
    auto p = output.ptr();
    for(int i = 0; i < output.numel(); ++i, ++p)
        *p = -*p;
    return output;
}

Matrix Matrix::power(float y) const{
    Matrix output = this->copy();
    auto op = output.ptr();
    for(int i = 0; i < output.numel(); ++i, ++op)
        *op = std::pow(*op, y);
    return output;
}

int Matrix::argmax(int irow) const{
    auto p = ptr(irow);
    return std::max_element(p, p + size(1)) - p;
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

Matrix Matrix::reference_d0(int idx) const{

    vector<int> newshape(shape_.size() - 1);
    for(int i = 1; i < shape_.size(); ++i)
        newshape[i-1] = shape_[i];
    
    return make_reference(newshape, offset(idx));
}

Matrix Matrix::make_reference(const std::vector<int>& shape, size_t offset)const{

    Matrix out;
    out.shape_ = shape;
    out.data_->owner = false;
    out.data_->pdata = data_->pdata + offset;
    out.data_->numel = compute_numel(shape);
    out.data_->capacity_size = data_->capacity_size - offset;
    return out;
}

Matrix Matrix::slice(const std::vector<int>& indexs, int begin, int size){

    if(size == -1) size = indexs.size();

    auto newshape = this->shape_;
    newshape[0] = size;

    size_t n = this->count_of(1);
    Matrix out(newshape);
    for(int i = 0; i < size; ++i){
        int mrow = indexs[i + begin];
        int orow = i;
        memcpy(out.ptr(orow), this->ptr(mrow), sizeof(float) * n);
    }
    return out;
}

void Matrix::fill_zero_(){
    auto p = ptr();
    memset(p, 0, sizeof(float) * numel());
}

void Matrix::fill_(float scalar){

    auto p = ptr();
    for(int i = 0; i < numel(); ++i)
        p[i] = scalar;
}

Matrix Matrix::reduce_sum_by_row() const{

    int n = size(1);
    Matrix out({1, n});
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < size(0); ++i)
        for(int j = 0; j < n; ++j)
            optr[j] += *vptr++;
    return out;
}

Matrix Matrix::reduce_sum_by_col() const{
    Matrix out({size(0), 1});
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < size(0); ++i)
        for(int j = 0; j < size(1); ++j)
            optr[i] += *vptr++;
    return out;
}

Matrix Matrix::reduce_sum_all() const{
    Matrix out({1, 1});
    auto optr = out.ptr();
    auto vptr = ptr();
    for(int i = 0; i < numel(); ++i)
        *optr += *vptr++;
    return out;
}

Matrix Matrix::unsqueeze(int dim) const{

    auto newshape = this->shape_;
    if(dim >= newshape.size() || dim < 0)
        printf("Invalid unsqueeze dim[%d]\n", dim);

    newshape.insert(newshape.begin() + dim, 1);
    return view(newshape);
}

Matrix Matrix::view(const std::vector<int>& shape) const{

    Matrix output = *this;
    output.shape_ = shape;

    int unknow_dim = -1;
    int know_volumn = 1;
    for(int i = 0; i < shape.size(); ++i){
        if(shape[i] == -1){
            if(unknow_dim != -1)
                printf("Multi dim has -1.\n");

            unknow_dim = i;
        }else
            know_volumn *= shape[i];
    }

    if(unknow_dim != -1){
        if(this->numel() % know_volumn != 0)
            printf("Invalid view, dim -1.\n");
        
        output.shape_[unknow_dim] = numel() / know_volumn;
    }else if(compute_numel(shape) != this->data_->numel){
        printf("Invalid view. has change volumn\n");
    }
    return output;
}

bool Matrix::save(std::ostream& outfile) const{
    matrix_header_t header{0};
    header.flag = 0xFCCFE2E2;
    header.ndim = ndim();
    header.dtype = 0;

    outfile.write((char*)&header, sizeof(header));
    outfile.write((char*)shape_.data(), sizeof(int) * header.ndim);
    outfile.write((char*)this->ptr(), sizeof(float) * this->numel());
    return outfile.good();
}

bool Matrix::load(std::istream& infile){

    matrix_header_t header{0};
    infile.read((char*)&header, sizeof(header));
    if(header.flag != 0xFCCFE2E2){
        printf("Invalid format for 0x%08X\n", header.flag);
        return false;
    }
    
    vector<int> shape(header.ndim);
    infile.read((char*)shape.data(), sizeof(int) * header.ndim);

    this->resize(shape);
    infile.read((char*)this->ptr(), sizeof(float) * this->numel());
    return infile.good();
}

bool Matrix::save(const std::string& file) const{

    ofstream fout(file, ios::binary | ios::out);
    if(!fout.good()) return false;
    return save(fout);
}

bool Matrix::load(const std::string& file){

    ifstream fin(file, ios::binary | ios::in);
    if(!fin.good()) return false;
    return load(fin);
}

Matrix Matrix::gemm(const Matrix& other, bool ta, bool tb, float alpha, float beta) const{

    int a_elastic_cols = ta ? this->size(0) : this->size(1);   /* 如果转置，则维度转过来 */
    int b_elastic_rows = tb ? other.size(1) : other.size(0);   /* 如果转置，则维度转过来 */

    if(a_elastic_cols != b_elastic_rows){
        printf("Invalid matrix multiply %dx%d%s -- %dx%d%s\n", 
            this->size(0), this->size(1), ta ? "^T" : "", 
            other.size(0), other.size(1), tb ? "^T" : ""
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
            printf("invalid numel %dx%d * %dx%d\n", a->size(0), a->size(1), b->size(0), b->size(1));
            return make_tuple(nullptr, nullptr, 0);
        }

        if(b->numel() == 1){
            broadcast = 3;
        }else if(b->size(1) == 1){
            broadcast = 1;
        }else if(b->size(0) == 1){
            broadcast = 2;
        }else{
            printf("Invalid broadcast for %d x %d\n", b->size(0), b->size(1));
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
             if(op == optype_t::Add) for(int i = 0; i < a->size(0); ++i){float bvalue = bdata[i];for(int j = 0; j < a->size(1); ++j) *odata++ += bvalue;}
        else if(op == optype_t::Sub) for(int i = 0; i < a->size(0); ++i){float bvalue = bdata[i];for(int j = 0; j < a->size(1); ++j) *odata++ -= bvalue;}
        else if(op == optype_t::Mul) for(int i = 0; i < a->size(0); ++i){float bvalue = bdata[i];for(int j = 0; j < a->size(1); ++j) *odata++ *= bvalue;}
        else if(op == optype_t::Div) for(int i = 0; i < a->size(0); ++i){float bvalue = bdata[i];for(int j = 0; j < a->size(1); ++j) *odata++ /= bvalue;}
    }else if(broadcast == 2){
             if(op == optype_t::Add) for(int i = 0; i < a->size(0); ++i){float* bvalue = bdata;for(int j = 0; j < a->size(1); ++j) *odata++ += *bvalue++;}
        else if(op == optype_t::Sub) for(int i = 0; i < a->size(0); ++i){float* bvalue = bdata;for(int j = 0; j < a->size(1); ++j) *odata++ -= *bvalue++;}
        else if(op == optype_t::Mul) for(int i = 0; i < a->size(0); ++i){float* bvalue = bdata;for(int j = 0; j < a->size(1); ++j) *odata++ *= *bvalue++;}
        else if(op == optype_t::Div) for(int i = 0; i < a->size(0); ++i){float* bvalue = bdata;for(int j = 0; j < a->size(1); ++j) *odata++ /= *bvalue++;}
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

void Matrix::resize(const vector<int>& shape){
    this->data_->resize(compute_numel(shape));
    this->shape_ = shape;
}

std::ostream& operator << (std::ostream& out, const Matrix& m){

    cout << "Matrix[ ";
    for(int i = 0; i < m.ndim(); ++i){
        cout << m.size(i);
        if(i < m.ndim() - 1)
            cout << " x ";
    }
    cout << " ] : " << m.ptr() << "\n";

    for(int i = 0; i < m.size(0); ++i){
        auto p = m.ptr(i);
        for(int j = 0; j < (m.ndim() > 1 ? m.size(1) : 1); ++j){
            if(j > 10){
                out << " ... ";
                break;
            }
            printf("%.3f\t", *p++);
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

    int a_elastic_rows = ta ? a.size(1) : a.size(0);   /* 如果转置，则维度转过来 */
    int a_elastic_cols = ta ? a.size(0) : a.size(1);   /* 如果转置，则维度转过来 */
    int b_elastic_rows = tb ? b.size(1) : b.size(0);   /* 如果转置，则维度转过来 */
    int b_elastic_cols = tb ? b.size(0) : b.size(1);   /* 如果转置，则维度转过来 */

    /* c是转置后维度的行和列 */
    Matrix c({a_elastic_rows, b_elastic_cols});

    int m = a_elastic_rows;
    int n = b_elastic_cols;
    int k = a_elastic_cols;
    int lda = a.size(1);
    int ldb = b.size(1);
    int ldc = c.size(1);

    /* cblas的gemm调用风格，在以后也会存在 */
    cblas_sgemm(
        CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
        m, n, k, alpha, a.ptr(), lda, b.ptr(), ldb, beta, c.ptr(), ldc
    );
    return c;
}