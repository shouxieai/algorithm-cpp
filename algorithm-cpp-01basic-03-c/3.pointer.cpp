#include <iostream> // c++ 标准库的一部分
#include <stdio.h>  // c标准库的一部分
using namespace std;

int main() {
    int yourNum = 6;
    int* yourNumPtr; 

    yourNumPtr = &yourNum;
    printf("youNum [直接取值] is %d\n", yourNum);
    printf("yourNumPtr %p\n", yourNumPtr);
    printf("yourNum [通过地址去取值] is %d\n", *yourNumPtr);
    printf("the addr of yourNumPtr is %p\n", &yourNumPtr);

    return 0;
}