
#include <stdio.h>
#include <string.h>
#include <cstdint>

struct __attribute__((packed)) tensor_header_t{
    unsigned int flag;
    unsigned int ndim;
    unsigned int dtype;
};

int main(){

    FILE* f = fopen("test.tensor", "rb");
    tensor_header_t header{0};
    fread(&header, 1, sizeof(header), f);
    if(header.flag != 0xFCCFE2E2){
        printf("Invalid format for 0x%08X\n", header.flag);
        fclose(f);
        return 1;
    }
    
    printf("flag = 0x%08X, ndim = %d, dtype = %d\n", header.flag, header.ndim, header.dtype);

    unsigned int shape[16];
    size_t volumn_size = 1;

    fread(shape, 1, sizeof(int) * header.ndim, f);
    for(int i = 0; i < header.ndim; ++i){
        volumn_size *= shape[i];
        printf("shape[%d] = %d\n", i, shape[i]);
    }
    
    int64_t* pdata = new int64_t[volumn_size];
    fread(pdata, 1, sizeof(int64_t) * volumn_size, f);
    for(int i = 0; i < volumn_size; ++i)
        printf("%ld ", pdata[i]);
    
    printf("\n");

    delete []pdata;
    fclose(f);
    return 0;
}