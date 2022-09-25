
#include <stdio.h>
#include <string.h>

struct zip_header_nopacked_t{
    unsigned int flag;
    unsigned short version;
    unsigned short type;
    unsigned short mode;
    unsigned short last_modify_time;
    unsigned short last_modify_date;
    unsigned int crc32;
    unsigned int compress_size;
    unsigned int raw_size;
    unsigned short name_size;
    unsigned short extra_size;
};

struct __attribute__((packed)) zip_header_t{
    unsigned int flag;
    unsigned short version;
    unsigned short type;
    unsigned short mode;
    unsigned short last_modify_time;
    unsigned short last_modify_date;
    unsigned int crc32;
    unsigned int compress_size;
    unsigned int raw_size;
    unsigned short name_size;
    unsigned short extra_size;
};

int main(){

    printf("sizeof(zip_header_nopacked_t) = %d\n", sizeof(zip_header_nopacked_t));
    printf("sizeof(zip_header_t) = %d\n", sizeof(zip_header_t));

    FILE* f = fopen("vscode-plugin-demo-master.zip", "rb");
    if(f == nullptr){
        printf("Open failed.\n");
        return -1;
    }

    while(!feof(f)){

        zip_header_t header{0};
        fread(&header, 1, sizeof(header), f);
        if(header.flag != 0x04034b50){
            printf("Flag is %X\n", header.flag);
            break;
        }
        
        char name[512] = {0};
        fread(name, 1, header.name_size, f);
        fseek(f, header.compress_size + header.extra_size, SEEK_CUR);
        printf("File: %s, Size: %d byte / %d byte,  Ratio: %.2f %%.\n", 
            name, header.compress_size, header.raw_size, 
            header.raw_size != 0 ? header.compress_size / (float)header.raw_size * 100 : 0
        );
    }
    fclose(f);
    return 0;
}