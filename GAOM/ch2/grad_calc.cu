#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "grad_calc.cuh"

__global__ void gradient_Kernel(float *in_Img, int iImgHeight, int iImgWidth, float *out_xx, float *out_yy, float *out_xy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = idy*(iImgWidth - 2) + idx;

	if (idx < iImgWidth && idy < iImgHeight)
	{
	}
}

__global__ void gradient_kernel(
	float *m_dImg1, float *m_dImg2,
	int m_iHeight, int m_iWidth,
	int m_iImgHeight, int m_iImgWidth,
	float *m_dR,
	float *m_dRx, float *m_dRy,
	float *m_dT,
	float *m_dTx, float *m_dTy, float *m_dTxy)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = tid / m_iWidth;
	int j = tid%m_iWidth;
	if (tid < m_iWidth*m_iHeight)
	{
		m_dR[tid] = m_dImg1[m_iImgWidth*(i + 1) + j + 1];
		m_dRx[tid] = 0.5*(m_dImg1[m_iImgWidth*(i + 1) + j + 2] - m_dImg1[m_iImgWidth*(i + 1) + j]);
		m_dRy[tid] = 0.5*(m_dImg1[m_iImgWidth*(i + 2) + j + 1] - m_dImg1[m_iImgWidth*(i)+j + 1]);

		m_dT[tid] = m_dImg2[m_iImgWidth*(i + 1) + j + 1];
		m_dTx[tid] = 0.5*(m_dImg2[m_iImgWidth*(i + 1) + j + 2] - m_dImg2[m_iImgWidth*(i + 1) + j]);
		m_dTy[tid] = 0.5*(m_dImg2[m_iImgWidth*(i + 2) + j + 1] - m_dImg2[m_iImgWidth*(i)+j + 1]);
		m_dTxy[tid] = 0.25*(m_dImg2[m_iImgWidth*(i + 2) + j + 2] - m_dImg2[m_iImgWidth*(i)+j + 2] -
			m_dImg2[m_iImgWidth*(i + 2) + j] + m_dImg2[m_iImgWidth*(i)+j]);
	}
}

__global__ void gradient_kernel_optimized(
	float *m_dImg1, float *m_dImg2,
	int m_iHeight, int m_iWidth,
	int m_iImgHeight, int m_iImgWidth,
	float *m_dR,
	float *m_dRx, float *m_dRy,
	float *m_dT,
	float *m_dTx, float *m_dTy, float *m_dTxy)
{
	//Block Index
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	//Thread Index
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	//Global Memory offset: every block actually begin with 2 overlapped pixels
	const int y = ty + (BLOCK_SIZE_Y - 2) * by;
	const int x = tx + (BLOCK_SIZE_X - 2) * bx;

	//Declare the shared memory for storing the tiled subset
	__shared__ float imgR_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ float imgT_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	//Load the images into shared memory
	if (y<m_iImgHeight && x<m_iImgWidth) {
		imgR_sh[ty][tx] = m_dImg1[y*m_iImgWidth + x];
		imgT_sh[ty][tx] = m_dImg2[y*m_iImgWidth + x];
	}
	__syncthreads();	//guarntee all the threads complete their work in a block

						//Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
	if (y >= 1 && y<m_iImgHeight - 1 && x >= 1 && x<m_iImgWidth - 1 &&
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1) {
		m_dR[(y - 1)*m_iWidth + x - 1] = imgR_sh[ty][tx];
		m_dRx[(y - 1)*m_iWidth + x - 1] = 0.5f * (imgR_sh[ty][tx + 1] - imgR_sh[ty][tx - 1]);
		m_dRy[(y - 1)*m_iWidth + x - 1] = 0.5f * (imgR_sh[ty + 1][tx] - imgR_sh[ty - 1][tx]);

		m_dT[(y - 1)*m_iWidth + x - 1] = imgT_sh[ty][tx];
		m_dTx[(y - 1)*m_iWidth + x - 1] = 0.5f * (imgT_sh[ty][tx + 1] - imgT_sh[ty][tx - 1]);
		m_dTy[(y - 1)*m_iWidth + x - 1] = 0.5f * (imgT_sh[ty + 1][tx] - imgT_sh[ty - 1][tx]);
		m_dTxy[(y - 1)*m_iWidth + x - 1] = 0.25 * (imgT_sh[ty + 1][tx + 1] - imgT_sh[ty - 1][tx + 1] - imgT_sh[ty + 1][tx - 1] + imgT_sh[ty - 1][tx - 1]);
	}
}

void runTest()
{
	
}