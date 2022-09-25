
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>

struct __attribute__((packed)) mnist_labels_header_t{
    unsigned int magic_number;
    unsigned int number_of_items;
};

struct __attribute__((packed)) mnist_images_header_t{
    unsigned int magic_number;
    unsigned int number_of_images;
    unsigned int number_of_rows;
    unsigned int number_of_columns;
};

unsigned int inverse_byte(unsigned int v){
    unsigned char* p = (unsigned char*)&v;
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
    return v;
}

int main(){

    FILE* f = fopen("mnist/train-labels.idx1-ubyte", "rb");
    mnist_labels_header_t labels_header{0};
    fread(&labels_header, 1, sizeof(labels_header), f);
    printf("labels_header.magic_number = %X, number_of_items = %d\n", 
        inverse_byte(labels_header.magic_number), inverse_byte(labels_header.number_of_items));
    
    unsigned char label = 0;
    fread(&label, 1, sizeof(label), f);
    printf("First label is: %d\n", label);
    fclose(f);

    f = fopen("mnist/train-images.idx3-ubyte", "rb");
    mnist_images_header_t images_header{0};
    fread(&images_header, 1, sizeof(images_header), f);
    printf("images_header.magic_number = %X, number_of_images = %d, number_of_rows = %d, number_of_columns = %d\n", 
        inverse_byte(images_header.magic_number), 
        inverse_byte(images_header.number_of_images),
        inverse_byte(images_header.number_of_rows),
        inverse_byte(images_header.number_of_columns)
    );
    
    std::vector<unsigned char> image(inverse_byte(images_header.number_of_rows) * inverse_byte(images_header.number_of_columns));
    fread(image.data(), 1, image.size(), f);
    for(int i = 0;i < image.size(); ++i){
        if(image[i] == 0)
            printf("--- ");
        else
            printf("%03d ", image[i]);

        if((i + 1) % inverse_byte(images_header.number_of_columns) == 0)
            printf("\n");
    }
    fclose(f);
    return 0;
}