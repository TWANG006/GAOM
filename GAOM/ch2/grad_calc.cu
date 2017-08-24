#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "grad_calc.cuh"

#include <iostream>

__global__ void gradient_kernel(float *d_Img, int iHeight, int iWidth,
	int iImgHeight, int iImgWidth,
	float *d_Gx, float *d_Gy, float *d_Gxy)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = tid / iWidth;
	int j = tid % iWidth;
	if (tid < iWidth*iHeight)
	{
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

	//Load the images into shared memory
	if (y < iImgHeight && x < iImgWidth) {
		img_sh[ty][tx] = d_Img[y*iImgWidth + x];
	}

	//guarntee all the threads complete their work in a block
	__syncthreads();

	//Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
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

	float *h_Img, *d_Img, *d_Gx, *d_Gy, *d_Gxy;
	h_Img = (float*)malloc(iImgSize);

	for (int i = 0; i < iImgWidth * iImgHeight; i++)
	{
		h_Img[i] = float(rand() % 256);
	}
	
	cudaMalloc((void**)&d_Img, iImgSize);
	cudaMalloc((void**)&d_Gx, iSize);
	cudaMalloc((void**)&d_Gy, iSize);
	cudaMalloc((void**)&d_Gxy, iSize);

	cudaMemcpy(d_Img, h_Img, iImgSize, cudaMemcpyHostToDevice);

	cudaEvent_t start, k1, end;
	cudaEventCreate(&start);
	cudaEventCreate(&k1);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	gradient_kernel<<<(iImgWidth*iImgHeight + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_Img, 
		iHeight, iWidth, iImgHeight, iImgWidth, 
		d_Gx, d_Gy, d_Gxy);
	cudaEventRecord(k1);
	gradient_kernel_optimized<<<blocks, threads>>>(d_Img,
		iHeight, iWidth, iImgHeight, iImgWidth,
		d_Gx, d_Gy, d_Gxy);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float t1, t2;
	cudaEventElapsedTime(&t1, start, k1);
	cudaEventElapsedTime(&t2, k1, end);

	std::cout << "Image gradient calculation of " << iImgHeight << " x " << iImgWidth << " time: " << t1 << std::endl;
	std::cout << "Optimized image gradient calculation of " << iImgHeight << " x " << iImgWidth << " time: " << t2 << std::endl;
}