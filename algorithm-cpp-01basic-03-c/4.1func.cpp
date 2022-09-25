#include<stdio.h>
using namespace std;


// 声明
void my_print(int&);

// 实现
void my_print(int& num){
    printf("the printed number is %d\n", num);
}

// 调用
int main(){
    int myNum = 666;
    my_print(myNum);
    return 0;
}