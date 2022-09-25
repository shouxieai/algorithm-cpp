#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <initializer_list>
#include <ostream>
#include <istream>
#include <memory>
#include <functional>

/* 实现一个自定义的matrix类 */
struct MatrixData{
    float* pdata = nullptr;
    size_t capacity_size = 0;
    int numel = 0;
    bool owner = false;

    void resize(size_t nsize);
    void release();
    virtual ~MatrixData();
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
    Matrix(const std::vector<int>& shape, const std::vector<float>& values={});

    void                    resize(const std::vector<int>& shape);
    std::vector<int>         shape() const                {return shape_;}
    bool                     empty() const                {return shape_.empty();}
    int                       size(int i) const           {return shape_[i];}
    float*                    ptr()                       {return data_->pdata;}
    const float*              ptr() const                 {return data_->pdata;}
    float&                    operator()(int index)       {return data_->pdata[index];}
    const float&              operator()(int index) const {return data_->pdata[index];}
    size_t                    numel() const               {return data_->numel;}
    float                     item() const                {return data_->pdata[0];}
    size_t                    ndim() const                {return shape_.size();}

    Matrix make_reference(const std::vector<int>& shape, size_t offset) const;
    Matrix reference_d0(int idx) const;
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
    Matrix sigmoid() const;
    Matrix relu() const;
    Matrix gemm(const Matrix& other, bool ta = false, bool tb = false, float alpha = 1.0f, float beta = 0.0f) const;
    int argmax(int irow) const;
    void fill_(float scalar);
    void fill_zero_();
    Matrix reduce_sum_by_row() const;
    Matrix reduce_sum_by_col() const;
    Matrix reduce_sum_all() const;
    Matrix slice(const std::vector<int>& indexs, int begin=0, int size=-1);
    Matrix view(const std::vector<int>& shape) const;
    Matrix unsqueeze(int dim=0) const;
    bool save(const std::string& file) const;
    bool load(const std::string& file);
    bool save(std::ostream& outfile) const;
    bool load(std::istream& infile);

    static std::tuple<Matrix*, Matrix*, int> check_broadcast(Matrix* a, Matrix* b);

    template<typename ... _Args>
    float*                ptr(_Args... idxs) {return data_->pdata + offset(idxs...);}

    template<typename ... _Args>
    float*                ptr(_Args... idxs) const {return data_->pdata + offset(idxs...);}

    template<typename ... _Args>
    const float&              operator()(int i, _Args... idxs) const {return data_->pdata[offset(i, idxs...)];}
    
    template<typename ... _Args>
    float&                    operator()(int i, _Args... idxs)       {return data_->pdata[offset(i, idxs...)];}

    size_t count_of(int begin_dim) const{

        size_t volumn = 1;
        for(int i = begin_dim; i < ndim(); ++i)
            volumn *= size(i);
        return volumn;
    }

    template<typename ... _Args>
    size_t                offset(_Args... idxs)const {
        const int idxs_array[] = {idxs...};

        size_t volumn = 0;
        for(int i = 0; i < ndim(); ++i){
            volumn *= shape_[i];

            if(i < sizeof...(idxs))
                volumn += idxs_array[i];
        }
        return volumn;
    }

private:
    void compute(Matrix* a, Matrix* b, int broadcast, optype_t op);
    void compute_scalar(float value, Matrix* pmatrix, optype_t op) const;

private:
    std::shared_ptr<MatrixData> data_;
    std::vector<int> shape_;
};

/* 全局操作符重载，使得能够被cout << m; */
std::ostream& operator << (std::ostream& out, const Matrix& m);
Matrix operator * (float value, const Matrix& m);
Matrix operator + (float value, const Matrix& m);

/* 对gemm的封装 */
Matrix gemm(const Matrix& a, bool ta, const Matrix& b, bool tb, float alpha, float beta);

#endif // MATRIX_HPP