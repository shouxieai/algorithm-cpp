#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

class recorder{
public:
    recorder(const char* file, float x):file_(file){
        fh_ = fopen(file, "wb");
        push(x);
    }

    void push(float x){
        fprintf(fh_, "%f\n", x);
    }

    virtual ~recorder(){
        fclose(fh_);
    }

private:
    FILE* fh_ = nullptr;
    const char* file_ = nullptr;
};

/* 梯度下降法，函数式：f(t) = (t^2 - C)^2 */
float sqrt_gradient_descent(float x, bool print=false){

    float t = x / 2.0f;
    float L = std::pow(t * t - x, 2.0f);
    float eps = 1e-5f;
    float alpha = 0.0001f;
    int iter_count = 0;
    recorder r("sqrt_gradient_descent", t);

    while(L > eps){
        float derivative = 2 * (t * t - x) * 2 * t;
        t = t - derivative * alpha;
        L = std::pow(t * t - x, 2.0f);

        if(print){
            iter_count++;
            printf("L = %f, t = %f, dt = %f, iter_count = %d\n", L, t, derivative * alpha, iter_count);
            r.push(t);
        }
    }
    return t;
}

/* 牛顿法，求解f'(t) = 0，函数式：f(t) = (t^2 - C)^2 */
float sqrt_newton_method1(float x, bool print=false){

    float t = x / 2.0f;
    float L = std::pow(t * t - x, 2.0f);
    float eps = 1e-5f;
    int iter_count = 0;
    recorder r("sqrt_newton_method1", t);

    while(L > eps){
        float first_derivative = 2 * (t * t - x) * 2 * t;
        float second_derivative = 4 * t * 2 * t + 4 * (t * t - x);

        t = t - first_derivative / second_derivative;
        L = std::pow(t * t - x, 2.0f);

        if(print){
            iter_count++;
            printf("L = %f, t = %f, dt1 = %f, dt2 = %f, iter_count = %d\n", L, t, first_derivative, second_derivative, iter_count);
            r.push(t);
        }
    }
    return t;
}

/* 牛顿法，求解f(t) = 0，函数式：f(t) = (t^2 - C)^2 */
float sqrt_newton_method2(float x, bool print=false){

    float t = x / 2.0f;
    float L = std::pow(t * t - x, 2.0f);
    float eps = 1e-5f;
    int iter_count = 0;
    recorder r("sqrt_newton_method2", t);

    while(L > eps){
        float fx             = L;
        float first_derivative = 2 * (t * t - x) * 2 * t;

        t = t - fx / first_derivative;
        L = std::pow(t * t - x, 2.0f);

        if(print){
            iter_count++;
            printf("L = %f, t = %f, fx = %f, dt = %f, iter_count = %d\n", L, t, fx, first_derivative, iter_count);
            r.push(t);
        }
    }
    return t;
}

/* 牛顿法，求解f(t) = 0，函数式：f(t) = t^2 - C */
float sqrt_newton_method3(float x, bool print=false){

    float t = x / 2.0f;
    float L = t * t - x;
    float eps = 1e-5f;
    int iter_count = 0;
    recorder r("sqrt_newton_method3", t);

    while(abs(L) > eps){
        float fx             = L;
        float first_derivative = 2 * t;

        t = t - fx / first_derivative;
        L = t * t - x;

        if(print){
            iter_count++;
            printf("L = %f, t = %f, fx = %f, dt = %f, iter_count = %d\n", L, t, fx, first_derivative, iter_count);
            r.push(t);
        }
    }
    return t;
}


int main(){

    float x = 60;
    cout << "sqrt_gradient_descent(x) : " << sqrt_gradient_descent(x) << endl;
    cout << "sqrt_newton_method1(x) : " << sqrt_newton_method1(x) << endl;
    cout << "sqrt_newton_method2(x) : " << sqrt_newton_method2(x) << endl;
    cout << "sqrt_newton_method3(x) : " << sqrt_newton_method3(x) << endl;
    cout << "sqrt(x) : " << std::sqrt(x) << endl;

    sqrt_gradient_descent(x, true);
    sqrt_newton_method1(x, true);
    sqrt_newton_method2(x, true);
    sqrt_newton_method3(x, true);
    return 0;
}