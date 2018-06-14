#include "cuda.hpp"
#include "cublas_v2.h"
#include "cuda_runtime.h"

using namespace std;

int *r1, *r2, *r3, *c1, *c2, *c3;
float *denom, *var1, *var2;
cublasHandle_t handle;

//Kernels
__global__
void kernel_matrix_add(float *m1, float *m2, float *m3, int *r1, int *c1, int *r2, int *c2, int *r3, int *c3) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	m1[x * (*c1) + y] = m2[x * (*c2) + y] + m3[x * (*c3) + y];
}

__global__
void kernel_ReLU(float *m1, float *m2, int *r1, int *c1, int *r2, int *c2) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	m1[x * (*c1) + y] = max( (float)0, m2[x * (*c2) + y] );
}

__global__
void kernel_softmax(float *m1, float *m2, int *r1, int *c1, int *r2, int *c2, float *denom) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	m1[x * (*c1) + y] = exp(m2[x * (*c2) + y]) / *denom;
}

__global__
void kernel_cross_entropy_prime(float *da, float *a, float *o) {
	int x = threadIdx.x;
	da[x] = -((o[x] / a[x]) + ((1-o[x]) / (1 - a[x])));
}

__global__
void kernel_softmax_prime(float *da0, float *a1, float *da1, int *r1) {
	int x = threadIdx.x;
	da0[x] = 0;
	for (int i = 0; i < *r1; i++) {
		if (i == x) {
			da0[x] += da1[i] * (a1[i] - pow(a1[i], 2));
		} else {
			da0[x] += da1[i] * (-(a1[i] * a1[x]));
		}
	}
}

__global__
void kernel_ReLU_prime_biases(float *db, float *a, float *da) {
	int x = threadIdx.x;
	db[x] += da[x] * (a[x] != 0);
}

__global__
void kernel_ReLU_prime_others(float *da0, float *dw, float *w, float *a0, float *a1, float *da1, int *r1, int *r2) {
	int idx = (blockIdx.x * 1024 + threadIdx.x);
	int idxW;
	if (idx < *r1) {
		da0[idx] = 0;
		for (int i = 0; i < *r2; i++) {
			idxW = (i * (*r1)) + idx;
			if (a1[i] != 0) {
				da0[idx] += (da1[i] * w[idxW]);
				dw[idxW] += (da1[i] * a0[idx]);
			}
		}
	}
}

__global__
void kernel_gradient_descent_step(float *x, float *dx, float *var1, float *var2, int *r1) {
	int idx = (blockIdx.x * 1024 + threadIdx.x);
	if (idx < *r1) {
		x[idx] = x[idx] - ((*var1) * (dx[idx] / (*var2)));
		dx[idx] = 0;
	}
}

//Wrappers and... stuff


void cuda_matrix_mul(float **m1, float **m2, float **m3, int R2, int C2, int C3) {
	/**
	 * Alright so, here all the matrices are actually stored in row major. But cuBLAS uses matrices in column major. Now 
	 * I didn't want to go and change the entire code to store all the matrices in column major. SO, here I am just 
	 * passing the 2nd matrix as 1st and vice versa to the multiplication operation with inverted dimensions of the 
	 * matrices and voila, even though cuBLAS will store the result in column major, for us, our resultant matrix is 
	 * stored in row major form. 
	 * And I couldn't use CUBLAS_OP_T's because that way it would've still stored the resultant in column major. 
	 * (Basically, I would've gotten the transpose of the actual result that I wanted).
	 */
	int lda=C3,ldb=C2,ldc=C3;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    const float *A = *m3;
    const float *B = *m2;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, C3, R2, C2, alpha, A, lda, B, ldb, beta, *m1, ldc);
    cudaDeviceSynchronize();
}

void cuda_matrix_add(float **m1, float **m2, float **m3, int R1, int C1) {
	*r1 = R1;
	*r2 = R1;
	*r3 = R1;
	*c1 = C1;
	*c2 = C1;
	*c3 = C1;
	dim3 grid(1, 1, 1);
	dim3 block(*r1, *c1, 1);
	kernel_matrix_add <<<grid, block>>> (*m1, *m2, *m3, r1, c1, r2, c2, r3, c3);
	cudaDeviceSynchronize();
}

void cuda_ReLU(float **m1, float **m2, int R1) {
	*r1 = R1;
	*r2 = R1;
	*c1 = 1;
	*c2 = 1;
	dim3 grid(1, 1, 1);
	dim3 block(*r1, *c1, 1);
	kernel_ReLU <<<grid, block>>> (*m1, *m2, r1, c1, r2, c2);
	cudaDeviceSynchronize();
}

void cuda_softmax(float **m1, float **m2, int R1) {
	*r1 = R1;
	*r2 = R1;
	*c1 = 1;
	*c2 = 1;
	dim3 grid(1, 1, 1);
	dim3 block(*r1, *c1, 1);
	*denom = 0;
	for (int i = 0; i < *r1; i++) {
		*denom += exp((*m2)[i]);
	}
	kernel_softmax <<<grid, block>>> (*m1, *m2, r1, c1, r2, c2, denom);
	cudaDeviceSynchronize();
}

void cuda_cross_entropy_prime(float **da, float **a, float **o, int R) {
	dim3 grid(1, 1, 1);
	dim3 block(R, 1, 1);
	kernel_cross_entropy_prime <<<grid, block>>> (*da, *a, *o);
	cudaDeviceSynchronize();
}

void cuda_softmax_prime(float **da0, float **a1, float **da1, int R) {
	*r1 = R;
	dim3 grid(1, 1, 1);
	dim3 block(R, 1, 1);
	kernel_softmax_prime <<<grid, block>>> (*da0, *a1, *da1, r1);
	cudaDeviceSynchronize();
}

void cuda_ReLU_prime_biases(float **db, float **a, float **da, int R) {
	dim3 grid(1, 1, 1);
	dim3 block(R, 1, 1);
	kernel_ReLU_prime_biases <<<grid, block>>> (*db, *a, *da);
	cudaDeviceSynchronize();
}

void cuda_ReLU_prime_others(float **da0, float **dw, float **w, float **a0, float **a1, float **da1, int R1, int R2) {
	*r1 = R1;
	*r2 = R2;
	dim3 grid((((int)(R1 / 1024)) + 1), 1, 1);
	dim3 block(1024, 1, 1);
	kernel_ReLU_prime_others <<<grid, block>>> (*da0, *dw, *w, *a0, *a1, *da1, r1, r2);
	cudaDeviceSynchronize();
}

void cuda_gradient_descent_step(float **x, float **dx, float lr, int bs, int R1) {
	*r1 = R1;
	*var1 = lr;
	*var2 = bs;
	dim3 grid((((int)(R1 / 1024)) + 1), 1, 1);
	dim3 block(1024, 1, 1);
	kernel_gradient_descent_step <<<grid, block>>> (*x, *dx, var1, var2, r1);
	cudaDeviceSynchronize();
}

template <typename T>
void cuda_allocate(T **ptr, long long size) {
	cudaMallocManaged(ptr, size * sizeof(T));
	cudaMemset(*ptr, 0, size * sizeof(T));
}

template <typename T>
void cuda_delete(T **ptr) {
	cudaFree(*ptr);
}

void cuda_copy_to_device(float **arr1, float **arr2, int size) {
	cudaMemcpy(*arr1, *arr2, size * sizeof(float), cudaMemcpyHostToDevice);
}

bool cuda_init() {
	cuda_allocate(&r1, 1);
	cuda_allocate(&r2, 1);
	cuda_allocate(&r3, 1);
	cuda_allocate(&c1, 1);
	cuda_allocate(&c2, 1);
	cuda_allocate(&c3, 1);
	cuda_allocate(&denom, 1);
	cuda_allocate(&var1, 1);
	cuda_allocate(&var2, 1);
	cublasCreate(&handle);

	int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) {
        deviceCount = 0;
    }
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) {
            ++gpuDeviceCount;
        }
    }
    return gpuDeviceCount;
}

void cuda_destroy() {
	cuda_delete(&r1);
	cuda_delete(&r2);
	cuda_delete(&r3);
	cuda_delete(&c1);
	cuda_delete(&c2);
	cuda_delete(&c3);
	cuda_delete(&denom);
	cuda_delete(&var1);
	cuda_delete(&var2);
	cublasDestroy(handle);
}


template void cuda_allocate <char> (char**, long long);
template void cuda_allocate <int> (int**, long long);
template void cuda_allocate <float> (float**, long long);
template void cuda_allocate <double> (double**, long long);
template void cuda_allocate <long long> (long long**, long long);

template void cuda_delete <char> (char**);
template void cuda_delete <int> (int**);
template void cuda_delete <float> (float**);
template void cuda_delete <double> (double**);
template void cuda_delete <long long> (long long**);
