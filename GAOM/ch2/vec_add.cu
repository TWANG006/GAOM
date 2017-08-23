#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include "vec_add.cuh"
#include <iostream>

using namespace std;

__global__
void vecAdd_kernel(float *d_A, float *d_B, float *d_C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n)
		d_C[i] = d_A[i] + d_B[i];
}

void vecAdd(float* h_A, float *h_B, float *h_C, int n)
{
	int size = n * sizeof(float);
	
	float *d_A, *d_B, *d_C;	// device pointers
	cudaEvent_t start, end, ma, hd, ke, dh;	// cuda events used for timing
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventCreate(&ma);
	cudaEventCreate(&hd);
	cudaEventCreate(&ke);	
	cudaEventCreate(&dh);

	cudaEventRecord(start);

	// Device memory allocation
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaEventRecord(ma);

	// Host->device memory transfer
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	cudaEventRecord(hd);

	// Kernel launch
	vecAdd_kernel<<<(n + 256 - 1) / 256, 256>>>(d_A, d_B, d_C, n);

	cudaEventRecord(ke);

	// Device->Host memory transfer
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(dh);

	// Device memory deallocation
	cudaFree(d_A);	cudaFree(d_B);	cudaFree(d_C);

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float ma_time = 0, hd_time = 0, ke_time = 0, dh_time = 0, md_time = 0, total_time = 0;
	cudaEventElapsedTime(&ma_time, start, ma);
	cudaEventElapsedTime(&hd_time, ma, hd);
	cudaEventElapsedTime(&ke_time, hd, ke);
	cudaEventElapsedTime(&dh_time, ke, dh);
	cudaEventElapsedTime(&md_time, dh, end);
	cudaEventElapsedTime(&total_time, start, end);

	cout << "CUDA Vector Addition overall running time: " << total_time << " ms" << endl;
	cout << "Memory allocation time: " << ma_time << " ms" << endl;
	cout << "Memory transferring time: " << hd_time + dh_time << " ms" << endl;
	cout << "Kernel execution time: " << ke_time << " ms" << endl;
	cout << "Memory deallocation time: " << md_time << " ms" << endl;
}