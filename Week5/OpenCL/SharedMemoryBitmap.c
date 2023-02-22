#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const uint32_t WIDTH               = 512;
const uint32_t HEIGHT              = 512;
const uint32_t SIZE                = WIDTH * HEIGHT;
const uint32_t IMAGE_SIZE_IN_BYTES = SIZE * sizeof(uint8_t) * 4;

typedef struct {
        uint32_t x, y, z;
} dim;

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
                fwrite(pData, 4 * sizeof(uint8_t), SIZE, fp);
                fclose(fp);
        } else
                puts("Cannot write raw image.");
}

int main(void) {
        cl_int err;

        cl_device_id device_id;

        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not get device.", err);
                return -1;
        }

        cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create context.", err);
                return -1;
        }

        cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create command queue.", err);
                return -1;
        }

        FILE* fp = fopen("kernel.cl", "r");
        fseek(fp, 0, SEEK_END);
        size_t size = ftell(fp);

        if (size == 0) {
                printf("Error: %d. kernel file has no function.", err);
                return -1;
        }

        fseek(fp, 0, SEEK_SET);
        char* source = (char*)malloc(size);
        fread(source, 1, size, fp);
        fclose(fp);

        cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create program.", err);
                return -1;
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not build program.", err);
                return -1;
        }

        cl_mem bitmap_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, IMAGE_SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_kernel kernel = clCreateKernel(program, "MakeImageKernel", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        err = clSetKernelArg(kernel, 0, sizeof(bitmap_mem_obj), (void*)&bitmap_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        const size_t global_item_size[] = {WIDTH, HEIGHT};
        const size_t local_item_size[]  = {16, 16};

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, (const size_t*)&global_item_size,
                                     (const size_t*)&local_item_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel.", err);
                return -1;
        }

        uint8_t* bitmap = (uint8_t*)malloc(IMAGE_SIZE_IN_BYTES);

        err = clEnqueueReadBuffer(queue, bitmap_mem_obj, CL_TRUE, 0, IMAGE_SIZE_IN_BYTES, bitmap, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        save_raw_image("OutputImage.raw", bitmap);
        free(bitmap);

        err = clReleaseKernel(kernel);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release kernel.", err);
                return -1;
        }

        err = clReleaseProgram(program);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release program.", err);
                return -1;
        }

        err = clReleaseCommandQueue(queue);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release command queue.", err);
                return -1;
        }

        err = clReleaseContext(context);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release context.", err);
                return -1;
        }

        err = clReleaseMemObject(bitmap_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release buffer.", err);
                return -1;
        }

        return 0;
}