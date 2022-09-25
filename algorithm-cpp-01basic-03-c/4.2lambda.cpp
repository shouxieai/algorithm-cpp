#include<stdio.h>
using namespace std;

int main(){
    int c = 2;
    auto add = [=](int x, int y){return x + y + c;}; // [capture](parameters)->return-type{body}


    int sum = add(1, 8);
    printf("sum = %d\n", sum);
    return 0;
}