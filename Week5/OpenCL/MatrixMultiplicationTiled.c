#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
        uint32_t x, y, z;
} dim3;

void MatrixMultHost(const float* A, const float* B, float* C, const int N) {
        for (int Row = 0; Row < N; ++Row) {
                for (int Col = 0; Col < N; ++Col) {
                        float Pvalue = 0.0f;
                        for (int k = 0; k < N; ++k) Pvalue += A[Row * N + k] * B[k * N + Col];
                        C[Col + Row * N] = Pvalue;
                }
        }
}

int main(void) {
        const uint32_t N             = 500;
        const uint32_t SIZE_IN_BYTES = N * N * sizeof(float);

        dim3 threadsBlock = {16, 16, 1}, blocksGrid = {4, 4, 1};

        do blocksGrid.x <<= 1;
        while (N > blocksGrid.x);
        do blocksGrid.y <<= 1;
        while (N > blocksGrid.y);

        const size_t global_item_size[2] = {blocksGrid.x, blocksGrid.y};
        const size_t local_item_size[2]  = {threadsBlock.x, threadsBlock.y};

        float* A           = (float*)malloc(SIZE_IN_BYTES);
        float* B           = (float*)malloc(SIZE_IN_BYTES);
        float* C           = (float*)malloc(SIZE_IN_BYTES);
        float* C_non_tiled = (float*)malloc(SIZE_IN_BYTES);  // device calc res
        float* C_tiled     = (float*)malloc(SIZE_IN_BYTES);  // device calc res

        // Initialize matrices on the host
        uint32_t i = 0;
        for (; i < N * N; i++) {
                A[i] = (float)(rand() % 10);
                B[i] = (float)(rand() % 10);
        }

        MatrixMultHost(A, B, C, N);

        cl_int       err;
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

        cl_program program =
                clCreateProgramWithSource(context, 1, (const char**)&source, &size, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create program.", err);
                return -1;
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not build program.", err);
                return -1;
        }

        cl_mem A_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem B_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem C_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE_IN_BYTES, NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        cl_mem Width_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create buffer.", err);
                return -1;
        }

        err = clEnqueueWriteBuffer(queue, A_mem_obj, CL_TRUE, 0, SIZE_IN_BYTES, A, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, B_mem_obj, CL_TRUE, 0, SIZE_IN_BYTES, B, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, Width_mem_obj, CL_TRUE, 0, sizeof(int), &N, 0, NULL,
                                    NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not write buffer.", err);
                return -1;
        }

        cl_kernel kernel_non_tiled = clCreateKernel(program, "MatrixMulKernel", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        err = clSetKernelArg(kernel_non_tiled, 0, sizeof(cl_mem), (void*)&A_mem_obj);
        err |= clSetKernelArg(kernel_non_tiled, 1, sizeof(cl_mem), (void*)&B_mem_obj);
        err |= clSetKernelArg(kernel_non_tiled, 2, sizeof(cl_mem), (void*)&C_mem_obj);
        err |= clSetKernelArg(kernel_non_tiled, 3, sizeof(cl_mem), (void*)&Width_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel_non_tiled, 2, NULL,
                                     (const size_t*)&global_item_size,
                                     (const size_t*)&local_item_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel_non_tiled.", err);
                return -1;
        }

        err = clEnqueueReadBuffer(queue, C_mem_obj, CL_TRUE, 0, SIZE_IN_BYTES, C_non_tiled, 0, NULL,
                                  NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        cl_kernel kernel_tiled = clCreateKernel(program, "MatrixMulKernelTiled", &err);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not create kernel.", err);
                return -1;
        }

        err = clSetKernelArg(kernel_tiled, 0, sizeof(cl_mem), (void*)&A_mem_obj);
        err |= clSetKernelArg(kernel_tiled, 1, sizeof(cl_mem), (void*)&B_mem_obj);
        err |= clSetKernelArg(kernel_tiled, 2, sizeof(cl_mem), (void*)&C_mem_obj);
        err |= clSetKernelArg(kernel_tiled, 3, sizeof(cl_mem), (void*)&Width_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not set kernel arguments.", err);
                return -1;
        }

        err = clEnqueueNDRangeKernel(queue, kernel_tiled, 2, NULL, (const size_t*)&global_item_size,
                                     (const size_t*)&local_item_size, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not enqueue kernel_tiled.", err);
                return -1;
        }

        err = clEnqueueReadBuffer(queue, C_mem_obj, CL_TRUE, 0, SIZE_IN_BYTES, C_tiled, 0, NULL,
                                  NULL);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not read buffer.", err);
                return -1;
        }

        // validate results
        // validate results computed by GPU
        int all_ok = 1;

        for (i = 0; i < N * N; ++i)
                if (C[i] != C_non_tiled[i]) {
                        all_ok = 0;
                        break;
                }

        // roughly compute speedup
        if (all_ok)
                printf("All results are correct!!! (non-Tiled)\n");
        else
                printf("incorrect results (non-Tiled)\n");

        // validate results
        // validate results computed by GPU Tiled
        all_ok = 1;
        for (i = 0; i < N * N; ++i)
                if (C[i] != C_tiled[i]) {
                        all_ok = 0;
                        break;
                };

        // roughly compute speedup
        if (all_ok)
                printf("All results are correct!!! (Tiled)\n");
        else
                printf("incorrect results (Tiled)\n");

        err = clReleaseKernel(kernel_non_tiled);
        err |= clReleaseKernel(kernel_tiled);

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

        err = clReleaseMemObject(A_mem_obj);
        err |= clReleaseMemObject(B_mem_obj);
        err |= clReleaseMemObject(C_mem_obj);
        err |= clReleaseMemObject(Width_mem_obj);

        if (err != CL_SUCCESS) {
                printf("Error: %d. OpenCL could not release buffer.", err);
                return -1;
        }

        free(A);
        free(B);
        free(C);
        free(C_non_tiled);
        free(C_tiled);

        return 0;
}