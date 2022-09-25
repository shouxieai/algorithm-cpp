#include <iostream> // c++ 标准库的一部分
#include <stdio.h>  // c标准库的一部分
using namespace std;

int main() { 
    
    int myNum = 5;               // Integer (whole number)
    float myFloatNum = 5.99;     // Floating point number
    double myDoubleNum = 9.98;   // Floating point number
    char myLetter = 'D';         // Character
    bool myBoolean = true;       // Boolean
    string myText = "Hello";     // String

    printf("myNum = %d\nmyFloatNum = %f\nmyDoubleNum = %f\n", myNum, myFloatNum, myDoubleNum);
    // 打印不同的变量可参考 https://www.runoob.com/cprogramming/c-function-printf.html

    return 0;
}