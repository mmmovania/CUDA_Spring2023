#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>

const int WIDTH               = 512;
const int HEIGHT              = 512;
const int SIZE                = WIDTH * HEIGHT;
const int IMAGE_SIZE_IN_BYTES = SIZE * sizeof(uint8_t);

void load_raw_image(const char* imageName, uint8_t* pData) {
    FILE* fp = fopen(imageName, "rb");
    if (fp) {
        fread(pData, 1, SIZE, fp);
        fclose(fp);
    } else
        puts("Cannot open raw image.");
}

void save_raw_image(const char* imageName, uint8_t* pData) {
    FILE* fp = fopen(imageName, "wb");
    if (fp) {
        fwrite(pData, 1, SIZE, fp);
        fclose(fp);
    } else
        puts("Cannot write raw image.");
}

int main(int argc, char** argv) { return 0; }