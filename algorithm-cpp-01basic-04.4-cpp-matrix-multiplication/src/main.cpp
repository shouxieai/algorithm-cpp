
#include <stdarg.h>
#include <iostream>
#include <vector>
#include "matrix.hpp"

using namespace std;

namespace Application{

    namespace logger{

        #define INFO(...)  Application::logger::__printf(__FILE__, __LINE__, __VA_ARGS__)

        void __printf(const char* file, int line, const char* fmt, ...){

            va_list vl;
            va_start(vl, fmt);

            printf("\e[32m[%s:%d]:\e[0m ", file, line);
            vprintf(fmt, vl);
            printf("\n");
        }
    };

    struct Point{
        float x, y;

        Point(float x, float y):x(x), y(y){}
        Point() = default;
    };

    Matrix mygemm(const Matrix& a, const Matrix& b){

        Matrix c(a.rows(), b.cols());
        for(int i = 0; i < c.rows(); ++i){
            for(int j = 0; j < c.cols(); ++j){
                float summary = 0;
                for(int k = 0; k < a.cols(); ++k)
                    summary += a(i, k) * b(k, j);

                c(i, j) = summary;
            }
        }
        return c;
    }

    /* 求解仿射变换矩阵 */
    Matrix get_affine_transform(const vector<Point>& src, const vector<Point>& dst){

        //         P                M        Y
        // x1, y1, 1, 0, 0, 0      m00       x1
        // 0, 0, 0, x1, y1, 1      m01       y1
        // x2, y2, 1, 0, 0, 0      m02       x2
        // 0, 0, 0, x2, y2, 1      m10       y2
        // x3, y3, 1, 0, 0, 0      m11       x3
        // 0, 0, 0, x3, y3, 1      m12       y3
        // Y = PM
        // P.inv() @ Y = M
        
        if(src.size() != 3 || dst.size() != 3){
            printf("Invalid to compute affine transform, src.size = %d, dst.size = %d\n", src.size(), dst.size());
            return Matrix();
        }

        Matrix P(6, 6, {
            src[0].x, src[0].y, 1, 0, 0, 0,
            0, 0, 0, src[0].x, src[0].y, 1,
            src[1].x, src[1].y, 1, 0, 0, 0,
            0, 0, 0, src[1].x, src[1].y, 1,
            src[2].x, src[2].y, 1, 0, 0, 0,
            0, 0, 0, src[2].x, src[2].y, 1
        });

        Matrix Y(6, 1, {
            dst[0].x, dst[0].y, dst[1].x, dst[1].y, dst[2].x, dst[2].y
        });
        return P.inv().gemm(Y).view(2, 3);
    }

    void test_matrix(){

        Matrix a1(2, 3, {
            1, 2, 3,
            4, 5, 6
        });

        Matrix b1(3, 2,{
            3, 0,
            2, 1,
            0, 2
        });

        INFO("A1 @ B1 = ");
        std::cout << a1.gemm(b1) << std::endl;

        Matrix a2(3, 2, {
            1, 4,
            2, 5,
            3, 6
        });

        INFO("A2.T @ B1 =");
        std::cout << gemm(a2, true, b1, false, 1.0f, 0.0f) << std::endl;

        INFO("A1 @ B1 = ");
        std::cout << mygemm(a1, b1) << std::endl;

        INFO("a2 * 2 = ");
        std::cout << a2 * 2 << std::endl;

        Matrix c(2, 2, {
            1, 2, 
            3, 4
        });
        INFO("C.inv = ");
        std::cout << c.inv() << std::endl;
        std::cout << c.gemm(c.inv()) << std::endl;
    }

    void test_affine(){

        auto M = get_affine_transform({
            Point(0, 0), Point(10, 0), Point(10, 10)
        }, 
        {
            Point(20, 20), Point(100, 20), Point(100, 100)
        });

        INFO("Affine matrix = ");
        std::cout << M << std::endl;
    }

    /* 测试矩阵求导的过程 */
    void test_matrix_derivation(){

        /* loss = (X @ theta).sum() */

        Matrix X(
            3, 2, {
                1, 2, 
                2, 1, 
                0, 2
            }
        );

        Matrix theta(
            2, 3, {
                5, 1, 0,
                2, 3, 1
            }
        );

        auto loss = X.gemm(theta).reduce_sum();

        // G = dloss / d(X @ theta) = ones_like(X @ theta)
        // dloss/dX     = G @ theta.T
        // dloss/dtheta = X.T @ G
        INFO("Loss = %f", loss);

        Matrix G(3, 3, {1, 1, 1, 1, 1, 1, 1, 1, 1});

        INFO("dloss / dX = ");
        std::cout << G.gemm(theta, false, true);

        INFO("dloss / dtheta = ");
        std::cout << X.gemm(G, true);

    }

    int run(){
        test_matrix();
        test_affine();
        test_matrix_derivation();
        return 0;
    }
};

int main(){
    return Application::run();
}