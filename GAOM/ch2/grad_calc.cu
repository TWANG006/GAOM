#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "grad_calc.cuh"
#include "grad_calc.h"
#include "common.h"

#include <iostream>

__global__ void gradient_kernel(float *d_Img, int iHeight, int iWidth,
	int iImgHeight, int iImgWidth,
	float *d_Gx, float *d_Gy, float *d_Gxy)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	// Recover the 2D indices
	int i = tid / iWidth;
	int j = tid % iWidth;

	if (tid < iWidth*iHeight){
		d_Gx[tid] = 0.5*(d_Img[iImgWidth*(i + 1) + j + 2] - d_Img[iImgWidth*(i + 1) + j]);
		d_Gy[tid] = 0.5*(d_Img[iImgWidth*(i + 2) + j + 1] - d_Img[iImgWidth*(i)+j + 1]);
		d_Gxy[tid] = 0.25*(d_Img[iImgWidth*(i + 2) + j + 2] - d_Img[iImgWidth*(i)+j + 2] -
			d_Img[iImgWidth*(i + 2) + j] + d_Img[iImgWidth*(i)+j]);
	}
}

__global__ void gradient_kernel_optimized(float *d_Img, int iHeight, int iWidth,
	int iImgHeight, int iImgWidth,
	float *d_Gx, float *d_Gy, float *d_Gxy)
{
	//Block Index
	const int by = blockIdx.y;	const int bx = blockIdx.x;
	//Thread Index
	const int ty = threadIdx.y;	const int tx = threadIdx.x;

	//Global Memory offset: every block actually begin with 2 overlapped pixels
	const int y = ty + (BLOCK_SIZE_Y - 2) * by;
	const int x = tx + (BLOCK_SIZE_X - 2) * bx;

	//Declare the shared memory for storing the tiled subset
	__shared__ float img_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	//Load the image from global memory to shared memory
	if (y < iImgHeight && x < iImgWidth) {
		img_sh[ty][tx] = d_Img[y*iImgWidth + x];
	}

	//Guarntee all the threads complete their work in a block
	__syncthreads();

	//Compute the gradients within the whole image, with 1-pixel shrunk on each boundary
	if (y >= 1 && y < iImgHeight - 1 && x >= 1 && x < iImgWidth - 1 &&
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1) {
		d_Gx[(y - 1)*iWidth + x - 1] = 0.5f * (img_sh[ty][tx + 1] - img_sh[ty][tx - 1]);
		d_Gy[(y - 1)*iWidth + x - 1] = 0.5f * (img_sh[ty + 1][tx] - img_sh[ty - 1][tx]);
		d_Gxy[(y - 1)*iWidth + x - 1] = 0.25 * (img_sh[ty + 1][tx + 1] - img_sh[ty - 1][tx + 1] - img_sh[ty + 1][tx - 1] + img_sh[ty - 1][tx - 1]);
	}
}

void gradientTest()
{
	int iImgWidth = 512, iImgHeight = 512;
	int iWidth = 510, iHeight = 510;
	int iImgSize = iImgWidth * iImgHeight * sizeof(float);
	int iSize = iWidth * iHeight * sizeof(float);
	dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks((int)ceil((float)iImgWidth / (BLOCK_SIZE_X - 2)), (int)ceil((float)iImgHeight / (BLOCK_SIZE_Y - 2)));

	float *h_Img, *d_Img, *h_Gx, *h_Gy, *h_Gxy, *d_Gx, *d_Gy, *d_Gxy;
	
	h_Img = (float*)malloc(iImgSize);
	h_Gx = (float*)malloc(iSize);
	h_Gy = (float*)malloc(iSize);
	h_Gxy = (float*)malloc(iSize);

	// Accuracy verification
	float *h_tGx, *h_tGy, *h_tGxy;
	h_tGx = (float*)malloc(iSize);
	h_tGy = (float*)malloc(iSize);
	h_tGxy = (float*)malloc(iSize);

	for (int i = 0; i < iImgWidth * iImgHeight; i++)
	{
		h_Img[i] = float(rand() % 256);
	}
	
	cudaMalloc((void**)&d_Img, iImgSize);
	cudaMalloc((void**)&d_Gx, iSize);
	cudaMalloc((void**)&d_Gy, iSize);
	cudaMalloc((void**)&d_Gxy, iSize);

	// Host time
	auto time1 = Time::now();
	gradient(h_Img, iImgWidth, iImgHeight, h_Gx, h_Gy, h_Gxy);
	auto time2 = Time::now();
	fsec hostTime = time2 - time1;
	std::cout << "Host Image gradient calculation of " << iImgHeight << " x "
		<< iImgWidth << " time: " << hostTime.count()*1000.0 << std::endl;

	// Device time
	cudaMemcpy(d_Img, h_Img, iImgSize, cudaMemcpyHostToDevice);

	cudaEvent_t start, k1, k2, end;
	cudaEventCreate(&start);
	cudaEventCreate(&k1);
	cudaEventCreate(&k2);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	gradient_kernel<<<(iImgWidth*iImgHeight + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_Img, 
		iHeight, iWidth, iImgHeight, iImgWidth, 
		d_Gx, d_Gy, d_Gxy);
	cudaEventRecord(k1);
	cudaEventSynchronize(k1);
	
	cudaMemcpy(h_tGx, d_Gx, iSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tGy, d_Gy, iSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tGxy, d_Gxy, iSize, cudaMemcpyDeviceToHost);

	bool passed = true;
	for (int i = 0; i < 10; i++)
	{
		if (!(h_tGx[i] - h_Gx[i] <= 1e-6 &&
			h_tGy[i] - h_Gy[i] <= 1e-6 &&
			h_tGxy[i] - h_Gxy[i] <= 1e-6))
		{
			passed = false;
		}
	}

	if (passed) {
		float t1;
		cudaEventElapsedTime(&t1, start, k1);

		std::cout << "Image gradient calculation of " << iImgHeight << " x "
			<< iImgWidth << " time: " << t1 << std::endl;
	}
	else
	{
		std::cout << "Incorrect Results: kernel 1." << std::endl;
	}

	cudaEventRecord(k2);
	gradient_kernel_optimized<<<blocks, threads>>>(d_Img,
		iHeight, iWidth, iImgHeight, iImgWidth,
		d_Gx, d_Gy, d_Gxy);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaMemcpy(h_tGx, d_Gx, iSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tGy, d_Gy, iSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tGxy, d_Gxy, iSize, cudaMemcpyDeviceToHost);

	passed = true;
	for (int i = 0; i < 10; i++)
	{
		if (!(h_tGx[i] - h_Gx[i] <= 1e-6 &&
			h_tGy[i] - h_Gy[i] <= 1e-6 &&
			h_tGxy[i] - h_Gxy[i] <= 1e-6))
		{
			passed = false;
		}
	}
	if (passed) {

		float t2;
		cudaEventElapsedTime(&t2, k2, end);
		std::cout << "Optimized image gradient calculation of " << iImgHeight << " x "
			<< iImgWidth << " time: " << t2 << std::endl;
	}
	

	// Memory deallocation
	cudaEventDestroy(start);
	cudaEventDestroy(k1);
	cudaEventDestroy(k2);
	cudaEventDestroy(end);

	cudaFree(d_Img);
	cudaFree(d_Gx);
	cudaFree(d_Gy);
	cudaFree(d_Gxy);

	free(h_Img); h_Img = nullptr;
	free(h_Gx); h_Gx = nullptr;
	free(h_Gy); h_Gy = nullptr;
	free(h_Gxy); h_Gxy = nullptr;
	free(h_tGx); h_tGx = nullptr;
	free(h_tGy); h_tGy = nullptr;
	free(h_tGxy); h_tGxy = nullptr;
}