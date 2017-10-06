/*
MIT License

Copyright (c) 2017 Tianyi Wang (twang006@e.ntu.edu.sg)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "precomputation.cuh"
#include "common.h"
#include "compute.h"

__constant__ int c_BicubicCoeffs[16][16] = {
	{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 },
	{ -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0 },
	{ 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 },
	{ -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1 },
	{ 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
	{ -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1 },
	{ 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 } };

/*************************************************/
/*@brief: kernel to calculate the gradients of both the reference and the target images*/
__global__ void gradient_kernel(float *dImg1, float *dImg2,	int iHeight, int iWidth, int iImgHeight, int iImgWidth,
	float *dR, float *dRx, float *dRy, float *dT, float *dTx, float *dTy, float *dTxy)
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
	if (y<iImgHeight && x<iImgWidth) {
		imgR_sh[ty][tx] = dImg1[y*iImgWidth + x];
		imgT_sh[ty][tx] = dImg2[y*iImgWidth + x];
	}
	__syncthreads();	//guarntee all the threads complete their work in a block

						//Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
	if (y >= 1 && y<iImgHeight - 1 && x >= 1 && x<iImgWidth - 1 &&
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1) {
		dR[(y - 1)*iWidth + x - 1] = imgR_sh[ty][tx];
		dRx[(y - 1)*iWidth + x - 1] = 0.5f * (imgR_sh[ty][tx + 1] - imgR_sh[ty][tx - 1]);
		dRy[(y - 1)*iWidth + x - 1] = 0.5f * (imgR_sh[ty + 1][tx] - imgR_sh[ty - 1][tx]);

		dT[(y - 1)*iWidth + x - 1] = imgT_sh[ty][tx];
		dTx[(y - 1)*iWidth + x - 1] = 0.5f * (imgT_sh[ty][tx + 1] - imgT_sh[ty][tx - 1]);
		dTy[(y - 1)*iWidth + x - 1] = 0.5f * (imgT_sh[ty + 1][tx] - imgT_sh[ty - 1][tx]);
		dTxy[(y - 1)*iWidth + x - 1] = 0.25 * (imgT_sh[ty + 1][tx + 1] - imgT_sh[ty - 1][tx + 1] - imgT_sh[ty + 1][tx - 1] + imgT_sh[ty - 1][tx - 1]);
	}
}

/*@brief: kernel to compute the bicubic LUT using the tiling pattern*/
__global__ void precompute_bicubic_kernel_pointwise(const float* din_fimgt, const float* din_ftx, const float* din_fty, const float* din_ftxy, const int iheight, const int iwidth,
	float4* dout_fbicubicinterpolants)
	/* input: din_fimgt, din_ftx, din_fty, din_ftxy computed by the precompute_gradient_kernel
	iheight, iwidth
	output: dout_fbicubicinterpolants
	perform: precompute the bicubic interpolants for the sub-pixel level interpolation
	*/
{
	
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	float falphat[16], ftaot[16];		//these two temporary arrays may consult 32 registers, half of the size for each thread's

	if (y<iheight - 1 && x<iwidth - 1) {

		//load the 16 parameters from the global memory
		ftaot[0] = din_fimgt[y*iwidth + x];
		ftaot[1] = din_fimgt[y*iwidth + x + 1];
		ftaot[2] = din_fimgt[(y + 1)*iwidth + x];
		ftaot[3] = din_fimgt[(y + 1)*iwidth + x + 1];
		ftaot[4] = din_ftx[y*iwidth + x];
		ftaot[5] = din_ftx[y*iwidth + x + 1];
		ftaot[6] = din_ftx[(y + 1)*iwidth + x];
		ftaot[7] = din_ftx[(y + 1)*iwidth + x + 1];
		ftaot[8] = din_fty[y*iwidth + x];
		ftaot[9] = din_fty[y*iwidth + x + 1];
		ftaot[10] = din_fty[(y + 1)*iwidth + x];
		ftaot[11] = din_fty[(y + 1)*iwidth + x + 1];
		ftaot[12] = din_ftxy[y*iwidth + x];
		ftaot[13] = din_ftxy[y*iwidth + x + 1];
		ftaot[14] = din_ftxy[(y + 1)*iwidth + x];
		ftaot[15] = din_ftxy[(y + 1)*iwidth + x + 1];

		//reduction to calculate falphat (unroll the "for" loop)
		falphat[0] = c_BicubicCoeffs[0][0] * ftaot[0] + c_BicubicCoeffs[0][1] * ftaot[1] + c_BicubicCoeffs[0][2] * ftaot[2] + c_BicubicCoeffs[0][3] * ftaot[3] +
			c_BicubicCoeffs[0][4] * ftaot[4] + c_BicubicCoeffs[0][5] * ftaot[5] + c_BicubicCoeffs[0][6] * ftaot[6] + c_BicubicCoeffs[0][7] * ftaot[7] +
			c_BicubicCoeffs[0][8] * ftaot[8] + c_BicubicCoeffs[0][9] * ftaot[9] + c_BicubicCoeffs[0][10] * ftaot[10] + c_BicubicCoeffs[0][11] * ftaot[11] +
			c_BicubicCoeffs[0][12] * ftaot[12] + c_BicubicCoeffs[0][13] * ftaot[13] + c_BicubicCoeffs[0][14] * ftaot[14] + c_BicubicCoeffs[0][15] * ftaot[15];
		falphat[1] = c_BicubicCoeffs[1][0] * ftaot[0] + c_BicubicCoeffs[1][1] * ftaot[1] + c_BicubicCoeffs[1][2] * ftaot[2] + c_BicubicCoeffs[1][3] * ftaot[3] +
			c_BicubicCoeffs[1][4] * ftaot[4] + c_BicubicCoeffs[1][5] * ftaot[5] + c_BicubicCoeffs[1][6] * ftaot[6] + c_BicubicCoeffs[1][7] * ftaot[7] +
			c_BicubicCoeffs[1][8] * ftaot[8] + c_BicubicCoeffs[1][9] * ftaot[9] + c_BicubicCoeffs[1][10] * ftaot[10] + c_BicubicCoeffs[1][11] * ftaot[11] +
			c_BicubicCoeffs[1][12] * ftaot[12] + c_BicubicCoeffs[1][13] * ftaot[13] + c_BicubicCoeffs[1][14] * ftaot[14] + c_BicubicCoeffs[1][15] * ftaot[15];
		falphat[2] = c_BicubicCoeffs[2][0] * ftaot[0] + c_BicubicCoeffs[2][1] * ftaot[1] + c_BicubicCoeffs[2][2] * ftaot[2] + c_BicubicCoeffs[2][3] * ftaot[3] +
			c_BicubicCoeffs[2][4] * ftaot[4] + c_BicubicCoeffs[2][5] * ftaot[5] + c_BicubicCoeffs[2][6] * ftaot[6] + c_BicubicCoeffs[2][7] * ftaot[7] +
			c_BicubicCoeffs[2][8] * ftaot[8] + c_BicubicCoeffs[2][9] * ftaot[9] + c_BicubicCoeffs[2][10] * ftaot[10] + c_BicubicCoeffs[2][11] * ftaot[11] +
			c_BicubicCoeffs[2][12] * ftaot[12] + c_BicubicCoeffs[2][13] * ftaot[13] + c_BicubicCoeffs[2][14] * ftaot[14] + c_BicubicCoeffs[2][15] * ftaot[15];
		falphat[3] = c_BicubicCoeffs[3][0] * ftaot[0] + c_BicubicCoeffs[3][1] * ftaot[1] + c_BicubicCoeffs[3][2] * ftaot[2] + c_BicubicCoeffs[3][3] * ftaot[3] +
			c_BicubicCoeffs[3][4] * ftaot[4] + c_BicubicCoeffs[3][5] * ftaot[5] + c_BicubicCoeffs[3][6] * ftaot[6] + c_BicubicCoeffs[3][7] * ftaot[7] +
			c_BicubicCoeffs[3][8] * ftaot[8] + c_BicubicCoeffs[3][9] * ftaot[9] + c_BicubicCoeffs[3][10] * ftaot[10] + c_BicubicCoeffs[3][11] * ftaot[11] +
			c_BicubicCoeffs[3][12] * ftaot[12] + c_BicubicCoeffs[3][13] * ftaot[13] + c_BicubicCoeffs[3][14] * ftaot[14] + c_BicubicCoeffs[3][15] * ftaot[15];
		falphat[4] = c_BicubicCoeffs[4][0] * ftaot[0] + c_BicubicCoeffs[4][1] * ftaot[1] + c_BicubicCoeffs[4][2] * ftaot[2] + c_BicubicCoeffs[4][3] * ftaot[3] +
			c_BicubicCoeffs[4][4] * ftaot[4] + c_BicubicCoeffs[4][5] * ftaot[5] + c_BicubicCoeffs[4][6] * ftaot[6] + c_BicubicCoeffs[4][7] * ftaot[7] +
			c_BicubicCoeffs[4][8] * ftaot[8] + c_BicubicCoeffs[4][9] * ftaot[9] + c_BicubicCoeffs[4][10] * ftaot[10] + c_BicubicCoeffs[4][11] * ftaot[11] +
			c_BicubicCoeffs[4][12] * ftaot[12] + c_BicubicCoeffs[4][13] * ftaot[13] + c_BicubicCoeffs[4][14] * ftaot[14] + c_BicubicCoeffs[4][15] * ftaot[15];
		falphat[5] = c_BicubicCoeffs[5][0] * ftaot[0] + c_BicubicCoeffs[5][1] * ftaot[1] + c_BicubicCoeffs[5][2] * ftaot[2] + c_BicubicCoeffs[5][3] * ftaot[3] +
			c_BicubicCoeffs[5][4] * ftaot[4] + c_BicubicCoeffs[5][5] * ftaot[5] + c_BicubicCoeffs[5][6] * ftaot[6] + c_BicubicCoeffs[5][7] * ftaot[7] +
			c_BicubicCoeffs[5][8] * ftaot[8] + c_BicubicCoeffs[5][9] * ftaot[9] + c_BicubicCoeffs[5][10] * ftaot[10] + c_BicubicCoeffs[5][11] * ftaot[11] +
			c_BicubicCoeffs[5][12] * ftaot[12] + c_BicubicCoeffs[5][13] * ftaot[13] + c_BicubicCoeffs[5][14] * ftaot[14] + c_BicubicCoeffs[5][15] * ftaot[15];
		falphat[6] = c_BicubicCoeffs[6][0] * ftaot[0] + c_BicubicCoeffs[6][1] * ftaot[1] + c_BicubicCoeffs[6][2] * ftaot[2] + c_BicubicCoeffs[6][3] * ftaot[3] +
			c_BicubicCoeffs[6][4] * ftaot[4] + c_BicubicCoeffs[6][5] * ftaot[5] + c_BicubicCoeffs[6][6] * ftaot[6] + c_BicubicCoeffs[6][7] * ftaot[7] +
			c_BicubicCoeffs[6][8] * ftaot[8] + c_BicubicCoeffs[6][9] * ftaot[9] + c_BicubicCoeffs[6][10] * ftaot[10] + c_BicubicCoeffs[6][11] * ftaot[11] +
			c_BicubicCoeffs[6][12] * ftaot[12] + c_BicubicCoeffs[6][13] * ftaot[13] + c_BicubicCoeffs[6][14] * ftaot[14] + c_BicubicCoeffs[6][15] * ftaot[15];
		falphat[7] = c_BicubicCoeffs[7][0] * ftaot[0] + c_BicubicCoeffs[7][1] * ftaot[1] + c_BicubicCoeffs[7][2] * ftaot[2] + c_BicubicCoeffs[7][3] * ftaot[3] +
			c_BicubicCoeffs[7][4] * ftaot[4] + c_BicubicCoeffs[7][5] * ftaot[5] + c_BicubicCoeffs[7][6] * ftaot[6] + c_BicubicCoeffs[7][7] * ftaot[7] +
			c_BicubicCoeffs[7][8] * ftaot[8] + c_BicubicCoeffs[7][9] * ftaot[9] + c_BicubicCoeffs[7][10] * ftaot[10] + c_BicubicCoeffs[7][11] * ftaot[11] +
			c_BicubicCoeffs[7][12] * ftaot[12] + c_BicubicCoeffs[7][13] * ftaot[13] + c_BicubicCoeffs[7][14] * ftaot[14] + c_BicubicCoeffs[7][15] * ftaot[15];
		falphat[8] = c_BicubicCoeffs[8][0] * ftaot[0] + c_BicubicCoeffs[8][1] * ftaot[1] + c_BicubicCoeffs[8][2] * ftaot[2] + c_BicubicCoeffs[8][3] * ftaot[3] +
			c_BicubicCoeffs[8][4] * ftaot[4] + c_BicubicCoeffs[8][5] * ftaot[5] + c_BicubicCoeffs[8][6] * ftaot[6] + c_BicubicCoeffs[8][7] * ftaot[7] +
			c_BicubicCoeffs[8][8] * ftaot[8] + c_BicubicCoeffs[8][9] * ftaot[9] + c_BicubicCoeffs[8][10] * ftaot[10] + c_BicubicCoeffs[8][11] * ftaot[11] +
			c_BicubicCoeffs[8][12] * ftaot[12] + c_BicubicCoeffs[8][13] * ftaot[13] + c_BicubicCoeffs[8][14] * ftaot[14] + c_BicubicCoeffs[8][15] * ftaot[15];
		falphat[9] = c_BicubicCoeffs[9][0] * ftaot[0] + c_BicubicCoeffs[9][1] * ftaot[1] + c_BicubicCoeffs[9][2] * ftaot[2] + c_BicubicCoeffs[9][3] * ftaot[3] +
			c_BicubicCoeffs[9][4] * ftaot[4] + c_BicubicCoeffs[9][5] * ftaot[5] + c_BicubicCoeffs[9][6] * ftaot[6] + c_BicubicCoeffs[9][7] * ftaot[7] +
			c_BicubicCoeffs[9][8] * ftaot[8] + c_BicubicCoeffs[9][9] * ftaot[9] + c_BicubicCoeffs[9][10] * ftaot[10] + c_BicubicCoeffs[9][11] * ftaot[11] +
			c_BicubicCoeffs[9][12] * ftaot[12] + c_BicubicCoeffs[9][13] * ftaot[13] + c_BicubicCoeffs[9][14] * ftaot[14] + c_BicubicCoeffs[9][15] * ftaot[15];
		falphat[10] = c_BicubicCoeffs[10][0] * ftaot[0] + c_BicubicCoeffs[10][1] * ftaot[1] + c_BicubicCoeffs[10][2] * ftaot[2] + c_BicubicCoeffs[10][3] * ftaot[3] +
			c_BicubicCoeffs[10][4] * ftaot[4] + c_BicubicCoeffs[10][5] * ftaot[5] + c_BicubicCoeffs[10][6] * ftaot[6] + c_BicubicCoeffs[10][7] * ftaot[7] +
			c_BicubicCoeffs[10][8] * ftaot[8] + c_BicubicCoeffs[10][9] * ftaot[9] + c_BicubicCoeffs[10][10] * ftaot[10] + c_BicubicCoeffs[10][11] * ftaot[11] +
			c_BicubicCoeffs[10][12] * ftaot[12] + c_BicubicCoeffs[10][13] * ftaot[13] + c_BicubicCoeffs[10][14] * ftaot[14] + c_BicubicCoeffs[10][15] * ftaot[15];
		falphat[11] = c_BicubicCoeffs[11][0] * ftaot[0] + c_BicubicCoeffs[11][1] * ftaot[1] + c_BicubicCoeffs[11][2] * ftaot[2] + c_BicubicCoeffs[11][3] * ftaot[3] +
			c_BicubicCoeffs[11][4] * ftaot[4] + c_BicubicCoeffs[11][5] * ftaot[5] + c_BicubicCoeffs[11][6] * ftaot[6] + c_BicubicCoeffs[11][7] * ftaot[7] +
			c_BicubicCoeffs[11][8] * ftaot[8] + c_BicubicCoeffs[11][9] * ftaot[9] + c_BicubicCoeffs[11][10] * ftaot[10] + c_BicubicCoeffs[11][11] * ftaot[11] +
			c_BicubicCoeffs[11][12] * ftaot[12] + c_BicubicCoeffs[11][13] * ftaot[13] + c_BicubicCoeffs[11][14] * ftaot[14] + c_BicubicCoeffs[11][15] * ftaot[15];
		falphat[12] = c_BicubicCoeffs[12][0] * ftaot[0] + c_BicubicCoeffs[12][1] * ftaot[1] + c_BicubicCoeffs[12][2] * ftaot[2] + c_BicubicCoeffs[12][3] * ftaot[3] +
			c_BicubicCoeffs[12][4] * ftaot[4] + c_BicubicCoeffs[12][5] * ftaot[5] + c_BicubicCoeffs[12][6] * ftaot[6] + c_BicubicCoeffs[12][7] * ftaot[7] +
			c_BicubicCoeffs[12][8] * ftaot[8] + c_BicubicCoeffs[12][9] * ftaot[9] + c_BicubicCoeffs[12][10] * ftaot[10] + c_BicubicCoeffs[12][11] * ftaot[11] +
			c_BicubicCoeffs[12][12] * ftaot[12] + c_BicubicCoeffs[12][13] * ftaot[13] + c_BicubicCoeffs[12][14] * ftaot[14] + c_BicubicCoeffs[12][15] * ftaot[15];
		falphat[13] = c_BicubicCoeffs[13][0] * ftaot[0] + c_BicubicCoeffs[13][1] * ftaot[1] + c_BicubicCoeffs[13][2] * ftaot[2] + c_BicubicCoeffs[13][3] * ftaot[3] +
			c_BicubicCoeffs[13][4] * ftaot[4] + c_BicubicCoeffs[13][5] * ftaot[5] + c_BicubicCoeffs[13][6] * ftaot[6] + c_BicubicCoeffs[13][7] * ftaot[7] +
			c_BicubicCoeffs[13][8] * ftaot[8] + c_BicubicCoeffs[13][9] * ftaot[9] + c_BicubicCoeffs[13][10] * ftaot[10] + c_BicubicCoeffs[13][11] * ftaot[11] +
			c_BicubicCoeffs[13][12] * ftaot[12] + c_BicubicCoeffs[13][13] * ftaot[13] + c_BicubicCoeffs[13][14] * ftaot[14] + c_BicubicCoeffs[13][15] * ftaot[15];
		falphat[14] = c_BicubicCoeffs[14][0] * ftaot[0] + c_BicubicCoeffs[14][1] * ftaot[1] + c_BicubicCoeffs[14][2] * ftaot[2] + c_BicubicCoeffs[14][3] * ftaot[3] +
			c_BicubicCoeffs[14][4] * ftaot[4] + c_BicubicCoeffs[14][5] * ftaot[5] + c_BicubicCoeffs[14][6] * ftaot[6] + c_BicubicCoeffs[14][7] * ftaot[7] +
			c_BicubicCoeffs[14][8] * ftaot[8] + c_BicubicCoeffs[14][9] * ftaot[9] + c_BicubicCoeffs[14][10] * ftaot[10] + c_BicubicCoeffs[14][11] * ftaot[11] +
			c_BicubicCoeffs[14][12] * ftaot[12] + c_BicubicCoeffs[14][13] * ftaot[13] + c_BicubicCoeffs[14][14] * ftaot[14] + c_BicubicCoeffs[14][15] * ftaot[15];
		falphat[15] = c_BicubicCoeffs[15][0] * ftaot[0] + c_BicubicCoeffs[15][1] * ftaot[1] + c_BicubicCoeffs[15][2] * ftaot[2] + c_BicubicCoeffs[15][3] * ftaot[3] +
			c_BicubicCoeffs[15][4] * ftaot[4] + c_BicubicCoeffs[15][5] * ftaot[5] + c_BicubicCoeffs[15][6] * ftaot[6] + c_BicubicCoeffs[15][7] * ftaot[7] +
			c_BicubicCoeffs[15][8] * ftaot[8] + c_BicubicCoeffs[15][9] * ftaot[9] + c_BicubicCoeffs[15][10] * ftaot[10] + c_BicubicCoeffs[15][11] * ftaot[11] +
			c_BicubicCoeffs[15][12] * ftaot[12] + c_BicubicCoeffs[15][13] * ftaot[13] + c_BicubicCoeffs[15][14] * ftaot[14] + c_BicubicCoeffs[15][15] * ftaot[15];

		//write the results back to the fbicubicinterpolants array
		dout_fbicubicinterpolants[0 * iwidth*iheight + (y*iwidth + x)].w = falphat[0];
		dout_fbicubicinterpolants[0 * iwidth*iheight + (y*iwidth + x)].x = falphat[1];
		dout_fbicubicinterpolants[0 * iwidth*iheight + (y*iwidth + x)].y = falphat[2];
		dout_fbicubicinterpolants[0 * iwidth*iheight + (y*iwidth + x)].z = falphat[3];
		dout_fbicubicinterpolants[1 * iwidth*iheight + (y*iwidth + x)].w = falphat[4];
		dout_fbicubicinterpolants[1 * iwidth*iheight + (y*iwidth + x)].x = falphat[5];
		dout_fbicubicinterpolants[1 * iwidth*iheight + (y*iwidth + x)].y = falphat[6];
		dout_fbicubicinterpolants[1 * iwidth*iheight + (y*iwidth + x)].z = falphat[7];
		dout_fbicubicinterpolants[2 * iwidth*iheight + (y*iwidth + x)].w = falphat[8];
		dout_fbicubicinterpolants[2 * iwidth*iheight + (y*iwidth + x)].x = falphat[9];
		dout_fbicubicinterpolants[2 * iwidth*iheight + (y*iwidth + x)].y = falphat[10];
		dout_fbicubicinterpolants[2 * iwidth*iheight + (y*iwidth + x)].z = falphat[11];
		dout_fbicubicinterpolants[3 * iwidth*iheight + (y*iwidth + x)].w = falphat[12];
		dout_fbicubicinterpolants[3 * iwidth*iheight + (y*iwidth + x)].x = falphat[13];
		dout_fbicubicinterpolants[3 * iwidth*iheight + (y*iwidth + x)].y = falphat[14];
		dout_fbicubicinterpolants[3 * iwidth*iheight + (y*iwidth + x)].z = falphat[15];
	}
}

__global__ void precompute_Bicubic_kernel(const float* dIn_fImgT, const float* dIn_fTx, const float* dIn_fTy, const float* dIn_fTxy, const int iHeight, const int iWidth,
	float4* dOut_fBicubicInterpolants)
{
	__shared__ float sh_T[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ float sh_Tx[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ float sh_Ty[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ float sh_Txy[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	const int bx = blockIdx.x;			const int tx = threadIdx.x;
	const int by = blockIdx.y;			const int ty = threadIdx.y;

	//Global Memory offset: every block actually begin with 2 overlapped pixels
	const int x = tx + (BLOCK_SIZE_X - 1) * bx;
	const int y = ty + (BLOCK_SIZE_Y - 1) * by;

	// Load the global mem to shared memory
	if (y < iHeight && x < iWidth)
	{
		sh_T[ty][tx] = dIn_fImgT[y*iWidth + x];
		sh_Tx[ty][tx] = dIn_fTx[y*iWidth + x];
		sh_Ty[ty][tx] = dIn_fTy[y*iWidth + x];
		sh_Txy[ty][tx] = dIn_fTxy[y*iWidth + x];
	}
	__syncthreads();

	float fAlphaT[16], fTaoT[16];		//These two temporary arrays may consult 32 registers, half of the size for each thread's

	if (y < iHeight - 1 && x < iWidth - 1 && ty != BLOCK_SIZE_Y - 1 && tx != BLOCK_SIZE_X - 1) {

		//Load the 16 parameters from the global memory
		fTaoT[0] = sh_T[ty][tx];
		fTaoT[1] = sh_T[ty][tx + 1];
		fTaoT[2] = sh_T[ty + 1][tx];
		fTaoT[3] = sh_T[ty + 1][tx + 1];
		fTaoT[4] = sh_Tx[ty][tx];
		fTaoT[5] = sh_Tx[ty][tx + 1];
		fTaoT[6] = sh_Tx[ty + 1][tx];
		fTaoT[7] = sh_Tx[ty + 1][tx + 1];
		fTaoT[8] = sh_Ty[ty][tx];
		fTaoT[9] = sh_Ty[ty][tx + 1];
		fTaoT[10] = sh_Ty[ty + 1][tx];
		fTaoT[11] = sh_Ty[ty + 1][tx + 1];
		fTaoT[12] = sh_Txy[ty][tx];
		fTaoT[13] = sh_Txy[ty][tx + 1];
		fTaoT[14] = sh_Txy[ty + 1][tx];
		fTaoT[15] = sh_Txy[ty + 1][tx + 1];

		//Reduction to calculate fAlphaT (unroll the "for" loop)
		fAlphaT[0] = c_BicubicCoeffs[0][0] * fTaoT[0] + c_BicubicCoeffs[0][1] * fTaoT[1] + c_BicubicCoeffs[0][2] * fTaoT[2] + c_BicubicCoeffs[0][3] * fTaoT[3] +
			c_BicubicCoeffs[0][4] * fTaoT[4] + c_BicubicCoeffs[0][5] * fTaoT[5] + c_BicubicCoeffs[0][6] * fTaoT[6] + c_BicubicCoeffs[0][7] * fTaoT[7] +
			c_BicubicCoeffs[0][8] * fTaoT[8] + c_BicubicCoeffs[0][9] * fTaoT[9] + c_BicubicCoeffs[0][10] * fTaoT[10] + c_BicubicCoeffs[0][11] * fTaoT[11] +
			c_BicubicCoeffs[0][12] * fTaoT[12] + c_BicubicCoeffs[0][13] * fTaoT[13] + c_BicubicCoeffs[0][14] * fTaoT[14] + c_BicubicCoeffs[0][15] * fTaoT[15];
		fAlphaT[1] = c_BicubicCoeffs[1][0] * fTaoT[0] + c_BicubicCoeffs[1][1] * fTaoT[1] + c_BicubicCoeffs[1][2] * fTaoT[2] + c_BicubicCoeffs[1][3] * fTaoT[3] +
			c_BicubicCoeffs[1][4] * fTaoT[4] + c_BicubicCoeffs[1][5] * fTaoT[5] + c_BicubicCoeffs[1][6] * fTaoT[6] + c_BicubicCoeffs[1][7] * fTaoT[7] +
			c_BicubicCoeffs[1][8] * fTaoT[8] + c_BicubicCoeffs[1][9] * fTaoT[9] + c_BicubicCoeffs[1][10] * fTaoT[10] + c_BicubicCoeffs[1][11] * fTaoT[11] +
			c_BicubicCoeffs[1][12] * fTaoT[12] + c_BicubicCoeffs[1][13] * fTaoT[13] + c_BicubicCoeffs[1][14] * fTaoT[14] + c_BicubicCoeffs[1][15] * fTaoT[15];
		fAlphaT[2] = c_BicubicCoeffs[2][0] * fTaoT[0] + c_BicubicCoeffs[2][1] * fTaoT[1] + c_BicubicCoeffs[2][2] * fTaoT[2] + c_BicubicCoeffs[2][3] * fTaoT[3] +
			c_BicubicCoeffs[2][4] * fTaoT[4] + c_BicubicCoeffs[2][5] * fTaoT[5] + c_BicubicCoeffs[2][6] * fTaoT[6] + c_BicubicCoeffs[2][7] * fTaoT[7] +
			c_BicubicCoeffs[2][8] * fTaoT[8] + c_BicubicCoeffs[2][9] * fTaoT[9] + c_BicubicCoeffs[2][10] * fTaoT[10] + c_BicubicCoeffs[2][11] * fTaoT[11] +
			c_BicubicCoeffs[2][12] * fTaoT[12] + c_BicubicCoeffs[2][13] * fTaoT[13] + c_BicubicCoeffs[2][14] * fTaoT[14] + c_BicubicCoeffs[2][15] * fTaoT[15];
		fAlphaT[3] = c_BicubicCoeffs[3][0] * fTaoT[0] + c_BicubicCoeffs[3][1] * fTaoT[1] + c_BicubicCoeffs[3][2] * fTaoT[2] + c_BicubicCoeffs[3][3] * fTaoT[3] +
			c_BicubicCoeffs[3][4] * fTaoT[4] + c_BicubicCoeffs[3][5] * fTaoT[5] + c_BicubicCoeffs[3][6] * fTaoT[6] + c_BicubicCoeffs[3][7] * fTaoT[7] +
			c_BicubicCoeffs[3][8] * fTaoT[8] + c_BicubicCoeffs[3][9] * fTaoT[9] + c_BicubicCoeffs[3][10] * fTaoT[10] + c_BicubicCoeffs[3][11] * fTaoT[11] +
			c_BicubicCoeffs[3][12] * fTaoT[12] + c_BicubicCoeffs[3][13] * fTaoT[13] + c_BicubicCoeffs[3][14] * fTaoT[14] + c_BicubicCoeffs[3][15] * fTaoT[15];
		fAlphaT[4] = c_BicubicCoeffs[4][0] * fTaoT[0] + c_BicubicCoeffs[4][1] * fTaoT[1] + c_BicubicCoeffs[4][2] * fTaoT[2] + c_BicubicCoeffs[4][3] * fTaoT[3] +
			c_BicubicCoeffs[4][4] * fTaoT[4] + c_BicubicCoeffs[4][5] * fTaoT[5] + c_BicubicCoeffs[4][6] * fTaoT[6] + c_BicubicCoeffs[4][7] * fTaoT[7] +
			c_BicubicCoeffs[4][8] * fTaoT[8] + c_BicubicCoeffs[4][9] * fTaoT[9] + c_BicubicCoeffs[4][10] * fTaoT[10] + c_BicubicCoeffs[4][11] * fTaoT[11] +
			c_BicubicCoeffs[4][12] * fTaoT[12] + c_BicubicCoeffs[4][13] * fTaoT[13] + c_BicubicCoeffs[4][14] * fTaoT[14] + c_BicubicCoeffs[4][15] * fTaoT[15];
		fAlphaT[5] = c_BicubicCoeffs[5][0] * fTaoT[0] + c_BicubicCoeffs[5][1] * fTaoT[1] + c_BicubicCoeffs[5][2] * fTaoT[2] + c_BicubicCoeffs[5][3] * fTaoT[3] +
			c_BicubicCoeffs[5][4] * fTaoT[4] + c_BicubicCoeffs[5][5] * fTaoT[5] + c_BicubicCoeffs[5][6] * fTaoT[6] + c_BicubicCoeffs[5][7] * fTaoT[7] +
			c_BicubicCoeffs[5][8] * fTaoT[8] + c_BicubicCoeffs[5][9] * fTaoT[9] + c_BicubicCoeffs[5][10] * fTaoT[10] + c_BicubicCoeffs[5][11] * fTaoT[11] +
			c_BicubicCoeffs[5][12] * fTaoT[12] + c_BicubicCoeffs[5][13] * fTaoT[13] + c_BicubicCoeffs[5][14] * fTaoT[14] + c_BicubicCoeffs[5][15] * fTaoT[15];
		fAlphaT[6] = c_BicubicCoeffs[6][0] * fTaoT[0] + c_BicubicCoeffs[6][1] * fTaoT[1] + c_BicubicCoeffs[6][2] * fTaoT[2] + c_BicubicCoeffs[6][3] * fTaoT[3] +
			c_BicubicCoeffs[6][4] * fTaoT[4] + c_BicubicCoeffs[6][5] * fTaoT[5] + c_BicubicCoeffs[6][6] * fTaoT[6] + c_BicubicCoeffs[6][7] * fTaoT[7] +
			c_BicubicCoeffs[6][8] * fTaoT[8] + c_BicubicCoeffs[6][9] * fTaoT[9] + c_BicubicCoeffs[6][10] * fTaoT[10] + c_BicubicCoeffs[6][11] * fTaoT[11] +
			c_BicubicCoeffs[6][12] * fTaoT[12] + c_BicubicCoeffs[6][13] * fTaoT[13] + c_BicubicCoeffs[6][14] * fTaoT[14] + c_BicubicCoeffs[6][15] * fTaoT[15];
		fAlphaT[7] = c_BicubicCoeffs[7][0] * fTaoT[0] + c_BicubicCoeffs[7][1] * fTaoT[1] + c_BicubicCoeffs[7][2] * fTaoT[2] + c_BicubicCoeffs[7][3] * fTaoT[3] +
			c_BicubicCoeffs[7][4] * fTaoT[4] + c_BicubicCoeffs[7][5] * fTaoT[5] + c_BicubicCoeffs[7][6] * fTaoT[6] + c_BicubicCoeffs[7][7] * fTaoT[7] +
			c_BicubicCoeffs[7][8] * fTaoT[8] + c_BicubicCoeffs[7][9] * fTaoT[9] + c_BicubicCoeffs[7][10] * fTaoT[10] + c_BicubicCoeffs[7][11] * fTaoT[11] +
			c_BicubicCoeffs[7][12] * fTaoT[12] + c_BicubicCoeffs[7][13] * fTaoT[13] + c_BicubicCoeffs[7][14] * fTaoT[14] + c_BicubicCoeffs[7][15] * fTaoT[15];
		fAlphaT[8] = c_BicubicCoeffs[8][0] * fTaoT[0] + c_BicubicCoeffs[8][1] * fTaoT[1] + c_BicubicCoeffs[8][2] * fTaoT[2] + c_BicubicCoeffs[8][3] * fTaoT[3] +
			c_BicubicCoeffs[8][4] * fTaoT[4] + c_BicubicCoeffs[8][5] * fTaoT[5] + c_BicubicCoeffs[8][6] * fTaoT[6] + c_BicubicCoeffs[8][7] * fTaoT[7] +
			c_BicubicCoeffs[8][8] * fTaoT[8] + c_BicubicCoeffs[8][9] * fTaoT[9] + c_BicubicCoeffs[8][10] * fTaoT[10] + c_BicubicCoeffs[8][11] * fTaoT[11] +
			c_BicubicCoeffs[8][12] * fTaoT[12] + c_BicubicCoeffs[8][13] * fTaoT[13] + c_BicubicCoeffs[8][14] * fTaoT[14] + c_BicubicCoeffs[8][15] * fTaoT[15];
		fAlphaT[9] = c_BicubicCoeffs[9][0] * fTaoT[0] + c_BicubicCoeffs[9][1] * fTaoT[1] + c_BicubicCoeffs[9][2] * fTaoT[2] + c_BicubicCoeffs[9][3] * fTaoT[3] +
			c_BicubicCoeffs[9][4] * fTaoT[4] + c_BicubicCoeffs[9][5] * fTaoT[5] + c_BicubicCoeffs[9][6] * fTaoT[6] + c_BicubicCoeffs[9][7] * fTaoT[7] +
			c_BicubicCoeffs[9][8] * fTaoT[8] + c_BicubicCoeffs[9][9] * fTaoT[9] + c_BicubicCoeffs[9][10] * fTaoT[10] + c_BicubicCoeffs[9][11] * fTaoT[11] +
			c_BicubicCoeffs[9][12] * fTaoT[12] + c_BicubicCoeffs[9][13] * fTaoT[13] + c_BicubicCoeffs[9][14] * fTaoT[14] + c_BicubicCoeffs[9][15] * fTaoT[15];
		fAlphaT[10] = c_BicubicCoeffs[10][0] * fTaoT[0] + c_BicubicCoeffs[10][1] * fTaoT[1] + c_BicubicCoeffs[10][2] * fTaoT[2] + c_BicubicCoeffs[10][3] * fTaoT[3] +
			c_BicubicCoeffs[10][4] * fTaoT[4] + c_BicubicCoeffs[10][5] * fTaoT[5] + c_BicubicCoeffs[10][6] * fTaoT[6] + c_BicubicCoeffs[10][7] * fTaoT[7] +
			c_BicubicCoeffs[10][8] * fTaoT[8] + c_BicubicCoeffs[10][9] * fTaoT[9] + c_BicubicCoeffs[10][10] * fTaoT[10] + c_BicubicCoeffs[10][11] * fTaoT[11] +
			c_BicubicCoeffs[10][12] * fTaoT[12] + c_BicubicCoeffs[10][13] * fTaoT[13] + c_BicubicCoeffs[10][14] * fTaoT[14] + c_BicubicCoeffs[10][15] * fTaoT[15];
		fAlphaT[11] = c_BicubicCoeffs[11][0] * fTaoT[0] + c_BicubicCoeffs[11][1] * fTaoT[1] + c_BicubicCoeffs[11][2] * fTaoT[2] + c_BicubicCoeffs[11][3] * fTaoT[3] +
			c_BicubicCoeffs[11][4] * fTaoT[4] + c_BicubicCoeffs[11][5] * fTaoT[5] + c_BicubicCoeffs[11][6] * fTaoT[6] + c_BicubicCoeffs[11][7] * fTaoT[7] +
			c_BicubicCoeffs[11][8] * fTaoT[8] + c_BicubicCoeffs[11][9] * fTaoT[9] + c_BicubicCoeffs[11][10] * fTaoT[10] + c_BicubicCoeffs[11][11] * fTaoT[11] +
			c_BicubicCoeffs[11][12] * fTaoT[12] + c_BicubicCoeffs[11][13] * fTaoT[13] + c_BicubicCoeffs[11][14] * fTaoT[14] + c_BicubicCoeffs[11][15] * fTaoT[15];
		fAlphaT[12] = c_BicubicCoeffs[12][0] * fTaoT[0] + c_BicubicCoeffs[12][1] * fTaoT[1] + c_BicubicCoeffs[12][2] * fTaoT[2] + c_BicubicCoeffs[12][3] * fTaoT[3] +
			c_BicubicCoeffs[12][4] * fTaoT[4] + c_BicubicCoeffs[12][5] * fTaoT[5] + c_BicubicCoeffs[12][6] * fTaoT[6] + c_BicubicCoeffs[12][7] * fTaoT[7] +
			c_BicubicCoeffs[12][8] * fTaoT[8] + c_BicubicCoeffs[12][9] * fTaoT[9] + c_BicubicCoeffs[12][10] * fTaoT[10] + c_BicubicCoeffs[12][11] * fTaoT[11] +
			c_BicubicCoeffs[12][12] * fTaoT[12] + c_BicubicCoeffs[12][13] * fTaoT[13] + c_BicubicCoeffs[12][14] * fTaoT[14] + c_BicubicCoeffs[12][15] * fTaoT[15];
		fAlphaT[13] = c_BicubicCoeffs[13][0] * fTaoT[0] + c_BicubicCoeffs[13][1] * fTaoT[1] + c_BicubicCoeffs[13][2] * fTaoT[2] + c_BicubicCoeffs[13][3] * fTaoT[3] +
			c_BicubicCoeffs[13][4] * fTaoT[4] + c_BicubicCoeffs[13][5] * fTaoT[5] + c_BicubicCoeffs[13][6] * fTaoT[6] + c_BicubicCoeffs[13][7] * fTaoT[7] +
			c_BicubicCoeffs[13][8] * fTaoT[8] + c_BicubicCoeffs[13][9] * fTaoT[9] + c_BicubicCoeffs[13][10] * fTaoT[10] + c_BicubicCoeffs[13][11] * fTaoT[11] +
			c_BicubicCoeffs[13][12] * fTaoT[12] + c_BicubicCoeffs[13][13] * fTaoT[13] + c_BicubicCoeffs[13][14] * fTaoT[14] + c_BicubicCoeffs[13][15] * fTaoT[15];
		fAlphaT[14] = c_BicubicCoeffs[14][0] * fTaoT[0] + c_BicubicCoeffs[14][1] * fTaoT[1] + c_BicubicCoeffs[14][2] * fTaoT[2] + c_BicubicCoeffs[14][3] * fTaoT[3] +
			c_BicubicCoeffs[14][4] * fTaoT[4] + c_BicubicCoeffs[14][5] * fTaoT[5] + c_BicubicCoeffs[14][6] * fTaoT[6] + c_BicubicCoeffs[14][7] * fTaoT[7] +
			c_BicubicCoeffs[14][8] * fTaoT[8] + c_BicubicCoeffs[14][9] * fTaoT[9] + c_BicubicCoeffs[14][10] * fTaoT[10] + c_BicubicCoeffs[14][11] * fTaoT[11] +
			c_BicubicCoeffs[14][12] * fTaoT[12] + c_BicubicCoeffs[14][13] * fTaoT[13] + c_BicubicCoeffs[14][14] * fTaoT[14] + c_BicubicCoeffs[14][15] * fTaoT[15];
		fAlphaT[15] = c_BicubicCoeffs[15][0] * fTaoT[0] + c_BicubicCoeffs[15][1] * fTaoT[1] + c_BicubicCoeffs[15][2] * fTaoT[2] + c_BicubicCoeffs[15][3] * fTaoT[3] +
			c_BicubicCoeffs[15][4] * fTaoT[4] + c_BicubicCoeffs[15][5] * fTaoT[5] + c_BicubicCoeffs[15][6] * fTaoT[6] + c_BicubicCoeffs[15][7] * fTaoT[7] +
			c_BicubicCoeffs[15][8] * fTaoT[8] + c_BicubicCoeffs[15][9] * fTaoT[9] + c_BicubicCoeffs[15][10] * fTaoT[10] + c_BicubicCoeffs[15][11] * fTaoT[11] +
			c_BicubicCoeffs[15][12] * fTaoT[12] + c_BicubicCoeffs[15][13] * fTaoT[13] + c_BicubicCoeffs[15][14] * fTaoT[14] + c_BicubicCoeffs[15][15] * fTaoT[15];

		//Write the results back to the fBicubicInterpolants array
		dOut_fBicubicInterpolants[0 * iWidth*iHeight + (y*iWidth + x)].w = fAlphaT[0];
		dOut_fBicubicInterpolants[0 * iWidth*iHeight + (y*iWidth + x)].x = fAlphaT[1];
		dOut_fBicubicInterpolants[0 * iWidth*iHeight + (y*iWidth + x)].y = fAlphaT[2];
		dOut_fBicubicInterpolants[0 * iWidth*iHeight + (y*iWidth + x)].z = fAlphaT[3];
		dOut_fBicubicInterpolants[1 * iWidth*iHeight + (y*iWidth + x)].w = fAlphaT[4];
		dOut_fBicubicInterpolants[1 * iWidth*iHeight + (y*iWidth + x)].x = fAlphaT[5];
		dOut_fBicubicInterpolants[1 * iWidth*iHeight + (y*iWidth + x)].y = fAlphaT[6];
		dOut_fBicubicInterpolants[1 * iWidth*iHeight + (y*iWidth + x)].z = fAlphaT[7];
		dOut_fBicubicInterpolants[2 * iWidth*iHeight + (y*iWidth + x)].w = fAlphaT[8];
		dOut_fBicubicInterpolants[2 * iWidth*iHeight + (y*iWidth + x)].x = fAlphaT[9];
		dOut_fBicubicInterpolants[2 * iWidth*iHeight + (y*iWidth + x)].y = fAlphaT[10];
		dOut_fBicubicInterpolants[2 * iWidth*iHeight + (y*iWidth + x)].z = fAlphaT[11];
		dOut_fBicubicInterpolants[3 * iWidth*iHeight + (y*iWidth + x)].w = fAlphaT[12];
		dOut_fBicubicInterpolants[3 * iWidth*iHeight + (y*iWidth + x)].x = fAlphaT[13];
		dOut_fBicubicInterpolants[3 * iWidth*iHeight + (y*iWidth + x)].y = fAlphaT[14];
		dOut_fBicubicInterpolants[3 * iWidth*iHeight + (y*iWidth + x)].z = fAlphaT[15];
	}
}

/*@brief: kernel to precompute the positions of POIs
    iNumberX: number of POIs in x direction
	iNumberY: number of POIs in y direction
	iMarginX: extra safe margin in x direction
	iMarginY: extra safe margin in y direction
	iSubsetX: subset Size in x = 2 * iSubsetX + 1
	iSubsetY: subset Size in y = 2 * iSubsetY + 1
	iGridSpaceX,Y: space between each two POIs
*/
__global__  void prepare_pxy_kernel(int iNumberX, int iNumberY,	int iMarginX, int iMarginY, int iSubsetX, int iSubsetY, int iGridSpaceX, int iGridSpaceY,
	int *dPXY)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int offset = y * iNumberX + x;

	int iTempY, iTempX;		//Save the computed value into registers first

	if (y<iNumberY && x<iNumberX) {
		iTempX = iMarginX + iSubsetY + y * iGridSpaceY;		//Index starts from 0
		iTempY = iMarginY + iSubsetX + x * iGridSpaceX;		//Index starts from 0

		dPXY[offset * 2 + 0] = iTempX;
		dPXY[offset * 2 + 1] = iTempY;
	}
}

void cu_precomputation(float *d_ImgR, float *d_ImgT,
	ICGN_d_Handle &Handle)
{
	int m_iImgHeight = Handle.m_iImgHeight;
	int m_iImgWidth = Handle.m_iImgWidth;
	int m_iHeight = Handle.m_iHeight;
	int m_iWidth = Handle.m_iWidth;
	int m_iNumberX = Handle.m_iNumberX;
	int m_iNumberY = Handle.m_iNumberY;
	int m_iMarginX = Handle.m_iMarginX;
	int m_iMarginY = Handle.m_iMarginY;
	int m_iSubsetX = Handle.m_iSubsetX;
	int m_iSubsetY = Handle.m_iSubsetY;
	int m_iGridSpaceX = Handle.m_iGridSpaceX;
	int m_iGridSpaceY = Handle.m_iGridSpaceY;
	float *m_d_dR = Handle.m_d_dR;
	float *m_d_dRx = Handle.m_d_dRx;
	float *m_d_dRy = Handle.m_d_dRy;
	float *m_d_dT = Handle.m_d_dT;
	float *m_d_dTx = Handle.m_d_dTx;
	float *m_d_dTy = Handle.m_d_dTy;
	float *m_d_dTxy = Handle.m_d_dTxy;
	float4 *m_d_dTBicubic = Handle.m_d_dTBicubic;
	int *m_d_dPXY = Handle.m_d_dPXY;

	dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks((int)ceil((float)m_iImgWidth / (BLOCK_SIZE_X - 2)), (int)ceil((float)m_iImgHeight / (BLOCK_SIZE_Y - 2)));
	gradient_kernel<<<blocks, threads>>>
		(d_ImgR, d_ImgT, m_iHeight, m_iWidth, m_iImgHeight, m_iImgWidth,
		m_d_dR, m_d_dRx, m_d_dRy, m_d_dT, m_d_dTx, m_d_dTy, m_d_dTxy);

	//dim3 threads2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	//dim3 blocks2((m_iWidth - 1) / BLOCK_SIZE_X + 1, (m_iHeight - 1) / BLOCK_SIZE_Y + 1);
	////cudaFuncSetCacheConfig(precompute_Bicubic_kernel, cudaFuncCachePreferL1);
	//precompute_Bicubic_kernel_pointwise<< <blocks2, threads2 >> >(m_d_dT, m_d_dTx, m_d_dTy, m_d_dTxy, m_iHeight, m_iWidth,
	//	m_d_dTBicubic);

	dim3 threads2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks2((int)ceil((float)m_iWidth / (BLOCK_SIZE_X - 1)),
		(int)ceil((float)m_iHeight / (BLOCK_SIZE_Y - 1)));
	//cudaFuncSetCacheConfig(precompute_Bicubic_kernel, cudaFuncCachePreferL1);
	precompute_Bicubic_kernel << <blocks2, threads2 >> >(m_d_dT, m_d_dTx, m_d_dTy, m_d_dTxy, m_iHeight, m_iWidth,
		m_d_dTBicubic);
	
	dim3 blocks1((m_iNumberX - 1) / BLOCK_SIZE_X + 1, (m_iNumberY - 1) / BLOCK_SIZE_Y + 1);
	prepare_pxy_kernel<<<blocks1, threads>>>
		(m_iNumberX, m_iNumberY, m_iMarginX, m_iMarginY, m_iSubsetX, m_iSubsetY, m_iGridSpaceX, m_iGridSpaceY, m_d_dPXY);
}