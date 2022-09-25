#include <iostream> // c++ 标准库的一部分
#include <stdio.h>  // c标准库的一部分
using namespace std;

int main(int argc, char* argv[]) { // 指向字符的指针数组
    printf("argc = %d\n", argc);
    printf("argv[0] = %s\n", argv[0]);
    printf("argv[1] = %s\n", argv[1]);
    printf("argv[2] = %s\n", argv[2]);
    
    return 0;
}