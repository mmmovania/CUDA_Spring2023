__kernel void vector_add(__global const int *A, __global const int *B,
                         __global int *C) {
    int i = get_global_id(0);

    C[i] = A[i] + B[i];
}

__kernel void add(__global const int *A, __global const int *B,
                  __global int *C) {
    *C = *A + *B;
}

__kernel void InitData(__global int *A, __global int *B) {}