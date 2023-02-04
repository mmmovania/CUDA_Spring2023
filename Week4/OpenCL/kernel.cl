/*
    get_group_id(uint dimindx)    =   blockIdx.[xyz]
    get_local_size(uint dimindx)  =   blockDim.[xyz]
    get_local_id(uint dimindx)    =   threadIdx.[xyz]
    get_num_groups(uint dimindx)  =   gridDim.[xyz]
*/

__kernel void matrix_multiplication(__global const float *A,
                                    __global const float *B, __global float *C,
                                    __global const int *M,
                                    __global const int *P,
                                    __global const int *N) {
    float sum = 0.0f;

    int tx = get_global_id(0), i = tx / *N, j = tx % *N;
    if (i < *M && j < *N)
        for (int k = 0; k < *P; k++) sum += A[i * *P + k] * B[k * *N + j];

    C[i * *N + j] = sum;
}

__kernel void PictureKernel(__global uint8_t *d_Pin, __global uint8_t *d_Pout,
                            __global int *n, __global int *m,
                            __global float *brightness = 1) {
    // Calculate the row #
    // int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Row = get_group_id(1) * get_local_size(1) + get_local_id(1);

    // Calculate the column #
    // int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = get_global_id(0) * get_local_size(0) + get_local_id(0);

    if ((Row < m) && (Col < n)) {
        int offset = (Row * n) + Col;
        // this is to flip the output image
        int offset2 = (((n - 1) - Row) * n) + Col;

        d_Pout[offset2] = d_Pin[offset] * brightness;
    }
}
