#include "cuda_util.hcu"
#include <stdio.h>
void catchCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

