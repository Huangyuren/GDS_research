#include <stdio.h>
#include <assert.h>

#define cudaCheckError() {                  \
    cudaError_t e = cudaGetLastError();     \
    if (e != cudaSuccess) {                 \
        printf("CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                 \
    }                                       \
}

inline cudaError_t cudaCheckErrorInline(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void initWith(float num, float *a, int N) {
    for(int i = 0; i < N; ++i) {
        a[i] = num;
    }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    for(int i = col; i < N; i += gridStride) {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N) {
    for(int i = 0; i < N; i++) {
        if(array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
  printf("SUCCESS! All values added correctly.\n");
}

int main() {
    const int N = 2<<20;
    size_t size = N * sizeof(float);

    float *h_a;
    float *h_b;
    float *h_c;

    // cudaCheckErrorInline(cudaMallocManaged(&h_a, size));
    // cudaCheckErrorInline(cudaMallocManaged(&h_b, size));
    // cudaCheckErrorInline(cudaMallocManaged(&h_c, size));

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    initWith(3, h_a, N);
    initWith(4, h_b, N);
    initWith(0, h_c, N);

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **) &dev_a, sizeof(float)*N);
    cudaMalloc((void **) &dev_b, sizeof(float)*N);
    cudaMalloc((void **) &dev_c, sizeof(float)*N);

    cudaMemcpy(dev_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, h_c, sizeof(float)*N, cudaMemcpyHostToDevice);

    size_t thread_per_block = 1024;
    size_t number_of_blocks = (N+thread_per_block - 1) / thread_per_block;
    addVectorsInto<<<number_of_blocks, thread_per_block>>>(dev_c, dev_a, dev_b, N);
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(h_c, dev_c, sizeof(float)*N, cudaMemcpyDeviceToHost);

    checkElementsAre(7, h_c, N);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaCheckErrorInline(cudaFree(dev_a));
    cudaCheckErrorInline(cudaFree(dev_b));
    cudaCheckErrorInline(cudaFree(dev_c));
}
