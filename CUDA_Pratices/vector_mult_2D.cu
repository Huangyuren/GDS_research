#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 1024
inline cudaError_t cudaCheckError(cudaError_t result) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

__global__ void matrixMultiplication() {}


int main(int argc, char* argv[]) {
	int row_a = atoi(argv[1]);
	int col_a = atoi(argv[2]);
	int col_b = atoi(argv[3]);
	int* h_a, *h_b, *h_c;
	cudaCheckError(cudaMallocHost(&h_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError(cudaMallocHost(&h_b, sizeof(int)*(col_a*col_b)));
	cudaCheckError(cudaMallocHost(&h_c, sizeof(int)*(row_a*col_b)));
	
	int* dev_a, *dev_b, *dev_c;
	cudaCheckError(cudaMalloc(&dev_a, sizeof(int)*(row_a*col_a)));
	cudaCheckError(cudaMalloc(&dev_b, sizeof(int)*(col-a*col_b)));
	cudaCheckError(cudaMalloc(&dev_c, sizeof(int)*(row_a*col_b)));

	cudaCheckError(cudaMemcpy(dev_a, h_a, sizeof(int)*(row_a*col_a), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(dev_b, h_b, sizeof(int)*(col_a*col_b), cudaMemcpyHostToDevice));

    int grid_row = (row_a + BLOCK_SIZE +1) / BLOCK_SIZE;
    int grid_col = (col_b + BLOCK_SIZE +1) / BLOCK_SIZE;
	dim3 dimGrid(grid_col, grid_row);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	matrixMultiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
}
