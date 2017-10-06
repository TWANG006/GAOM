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

#include "icgn.cuh"
#include "common.h"
#include "reducer.cuh"

__global__ void get_AveR_kernel_all_iteration(float*dR, int *dPXY,
	int iSubsetH, int iSubsetW,	int iSubsetX, int iSubsetY,	int iHeight, int iWidth,
	float *whole_dSubSet, float *whole_dSubsetAve)
{

	//默认smsize和dim大小相等
	//dim取64
	__shared__ float sm[BLOCK_SIZE_128];
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int size = iSubsetH*iSubsetW;
	float avg;// = 0;
	float mySum = 0;
	float tempt;
	float *dSubSet = whole_dSubSet + size*bid;
	float *dSubsetAve = whole_dSubsetAve + (size + 1)*bid;

	int t_dpxy0 = dPXY[bid * 2];
	int t_dpxy1 = dPXY[bid * 2 + 1];
	int t_iidx = iWidth*(int)(t_dpxy0 - iSubsetY) + (int)t_dpxy1 - iSubsetX;//少个l*iWidth
	float* p_dR = dR + t_iidx;

	for (int id = tid; id<size; id += dim)
	{
		int	l = id / iSubsetW;
		int m = id%iSubsetW;
		int t_sidx = m + l*iWidth;
		tempt = *(p_dR + t_sidx);;// dR[int(dPXY[bid * 2] - iSubsetY + l)*iWidth + int(dPXY[bid * 2 + 1] - iSubsetX + m)];
		dSubSet[id] = tempt;
		mySum += tempt / size;
	}
	__syncthreads();
	sumReduceBlock<BLOCK_SIZE_128, float>(sm, mySum, tid);
	__syncthreads();
	avg = sm[0];
	mySum = 0;
	for (int id = tid; id<size; id += dim)
	{
		tempt = dSubSet[id] - avg;
		mySum += tempt*tempt;
		dSubsetAve[id + 1] = tempt;
	}
	__syncthreads();
	sumReduceBlock<BLOCK_SIZE_128, float>(sm, mySum, tid);
	__syncthreads();
	if (tid == 0)
	{
		dSubsetAve[0] = sqrt(sm[tid]);
	}
}

__global__ void get_hessian_kernel_all_iteration(float* dRx, float *dRy, int *dPXY,
	int iSubsetX, int iSubsetY,	int iSubsetH, int iSubsetW,	int iHeight, int iWidth,
	float2 *whole_dRDescent, float* whole_dInvHessian
)
{
	__shared__ float dHessian[96];
	__shared__ float sm[BLOCK_SIZE_128];
	__shared__ int iIndOfRowTempt[8];
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int iSubWindowSize = iSubsetH*iSubsetW;
	int l;
	int m;

	float tempt;
	float t_dD0;
	float t_dD1;
	float t_dD2;
	float t_dD3;
	float t_dD4;
	float t_dD5;

	float2* dRDescent = whole_dRDescent + bid*iSubWindowSize * 3;
	float *r_dInvHessian = whole_dInvHessian + bid * 36;
	for (int id = tid; id<96; id += dim)
	{
		dHessian[id] = 0;
	}

	int t_dpxy0 = dPXY[bid * 2];
	int t_dpxy1 = dPXY[bid * 2 + 1];
	int t_iidx = iWidth*(int)(t_dpxy0 - iSubsetY) + (int)t_dpxy1 - iSubsetX;
	float* p_dRx = dRx + t_iidx;
	float* p_dRy = dRy + t_iidx;


	for (int id = tid; id<iSubWindowSize; id += dim)
	{
		l = id / iSubsetW;
		m = id%iSubsetW;

		int t_sidx = m + l*iWidth;
		float tx = *(p_dRx + t_sidx);// dRx[t_iidx + t_sidx];
		dRDescent[l*iSubsetW + m].x = t_dD0 = tx;
		dRDescent[l*iSubsetW + m].y = t_dD1 = tx*(m - iSubsetX);
		dRDescent[iSubWindowSize + l*iSubsetW + m].x = t_dD2 = tx*(l - iSubsetY);

		float ty = *(p_dRy + t_sidx);// dRy[t_iidx + t_sidx];
		dRDescent[iSubWindowSize + l*iSubsetW + m].y = t_dD3 = ty;
		dRDescent[iSubWindowSize * 2 + l*iSubsetW + m].x = t_dD4 = ty*(m - iSubsetX);
		dRDescent[iSubWindowSize * 2 + l*iSubsetW + m].y = t_dD5 = ty*(l - iSubsetY);
		//00		
		tempt = t_dD0 * t_dD0;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 0] += sm[0];
		}
		//11
		tempt = t_dD1 * t_dD1;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[1 * 16 + 1] += sm[0];
		}
		//22		
		tempt = t_dD2 * t_dD2;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[2 * 16 + 2] += sm[0];
		}
		//33
		tempt = t_dD3 * t_dD3;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[3 * 16 + 3] += sm[0];
		}
		//44		
		tempt = t_dD4 * t_dD4;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[4 * 16 + 4] += sm[0];
		}
		//55
		tempt = t_dD5 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[5 * 16 + 5] += sm[0];
		}

		//01		
		tempt = t_dD0 * t_dD1;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 1] += sm[0];
		}
		//02
		tempt = t_dD0 * t_dD2;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 2] += sm[0];
		}
		//03		
		tempt = t_dD0 * t_dD3;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 3] += sm[0];
		}
		//04
		tempt = t_dD0 * t_dD4;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 4] += sm[0];
		}
		//05		
		tempt = t_dD0 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[0 * 16 + 5] += sm[0];
		}

		//12
		tempt = t_dD1 * t_dD2;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[1 * 16 + 2] += sm[0];
		}
		//13		
		tempt = t_dD1 * t_dD3;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[1 * 16 + 3] += sm[0];
		}
		//14
		tempt = t_dD1 * t_dD4;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[1 * 16 + 4] += sm[0];
		}
		//15		
		tempt = t_dD1 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[1 * 16 + 5] += sm[0];
		}



		//23		
		tempt = t_dD2 * t_dD3;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[2 * 16 + 3] += sm[0];
		}
		//24
		tempt = t_dD2 * t_dD4;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[2 * 16 + 4] += sm[0];
		}
		//25		
		tempt = t_dD2 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[2 * 16 + 5] += sm[0];
		}


		//34
		tempt = t_dD3 * t_dD4;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[3 * 16 + 4] += sm[0];
		}
		//35		
		tempt = t_dD3 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[3 * 16 + 5] += sm[0];
		}

		//45		
		tempt = t_dD4 * t_dD5;
		sumReduceBlock<BLOCK_SIZE_128, float>(sm, tempt, tid);
		if (tid == 0)
		{
			dHessian[4 * 16 + 5] += sm[0];
		}
		/*if((tid+dim<iSubWindowSize))
		{	*/
		__syncthreads();
		if (tid<BLOCK_SIZE_128)
			sm[tid] = 0;
		//}
	}
	if (tid<5)
	{
		dHessian[(tid + 1) * 16 + 0] = dHessian[0 * 16 + (tid + 1)];
	}
	if (tid<4)
	{
		dHessian[(tid + 2) * 16 + 1] = dHessian[1 * 16 + (tid + 2)];
	}
	if (tid<3)
	{
		dHessian[(tid + 3) * 16 + 2] = dHessian[2 * 16 + (tid + 3)];
	}
	if (tid<2)
	{
		dHessian[(tid + 4) * 16 + 3] = dHessian[3 * 16 + (tid + 4)];
	}
	if (tid == 0)
	{
		dHessian[5 * 16 + 4] = dHessian[4 * 16 + 5];
	}

	if (tid<6)
	{
		dHessian[tid * 16 + tid + 8] = 1;
	}

	//6
	//find the max
	if (tid<16)
	{
		for (int l = 0; l < 6; l++)
		{
			//Find pivot (maximum lth column element) in the rest (6-l) rows
			if (tid<8)
			{
				iIndOfRowTempt[tid] = l;
			}
			if (tid<6 - l)
			{
				iIndOfRowTempt[tid] = tid + l;
			}
			if (tid<4)
			{
				if (dHessian[iIndOfRowTempt[tid] * 16 + l]<dHessian[iIndOfRowTempt[tid + 4] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 4];
			}
			if (tid<2)
			{
				if (dHessian[iIndOfRowTempt[tid] * 16 + l]<dHessian[iIndOfRowTempt[tid + 2] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 2];
			}
			if (tid == 0)
			{
				if (dHessian[iIndOfRowTempt[tid] * 16 + l]<dHessian[iIndOfRowTempt[tid + 1] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 1];
				if (dHessian[iIndOfRowTempt[tid] * 16 + l] == 0)
				{
					dHessian[iIndOfRowTempt[tid] * 16 + l] = 0.0000001;
				}
			}
			if (tid<12)
			{
				int iIndexOfCol = tid / 6 * 8 + tid % 6;
				float dTempt;
				if (iIndOfRowTempt[0] != l)
				{
					dTempt = dHessian[iIndOfRowTempt[0] * 16 + iIndexOfCol];
					dHessian[iIndOfRowTempt[0] * 16 + iIndexOfCol] = dHessian[l * 16 + iIndexOfCol];
					dHessian[l * 16 + iIndexOfCol] = dTempt;
				}

				// Perform row operation to form required identity matrix out of the Hessian matrix
				dHessian[l * 16 + iIndexOfCol] /= dHessian[l * 16 + l];
				for (int next_row = 0; next_row<6; next_row++)
				{
					if (next_row != l)
					{
						dHessian[next_row * 16 + iIndexOfCol] -= dHessian[l * 16 + iIndexOfCol] * dHessian[next_row * 16 + l];
					}
				}
			}
		}
	}

	//inv Hessian
	if (tid<32)
		r_dInvHessian[tid] = dHessian[tid / 6 * 16 + tid % 6 + 8];
	if (tid<4)
	{
		r_dInvHessian[tid + 32] = dHessian[(tid + 32) / 6 * 16 + (tid + 32) % 6 + 8];
	}
}

__global__ void icgn_kernel_all_iteration(int* d_iU, int *d_iV, int* d_dPXY,
	int iWidth, int iHeight, int iSubsetX, int iSubsetY, int iSubsetH, int iSubsetW, int iMaxIteration,
	float dNormDeltaP,
	float *dT, float*whole_d_dInvHessian,float4 *dTBicubic,	float2 *whole_d_2dRDescent, float *whole_d_dSubsetAveR,
	//tempt val
	float*whole_d_dSubsetT,	float *whole_d_dSubsetAveT,
	//return val
	int *whole_d_iIteration, float *whole_d_dP)
{
	__shared__ float sm[BLOCK_SIZE_64];
	__shared__ float dDP[6];
	__shared__ float dWarp[6];
	__shared__ float dP[6];
	__shared__ int break_sig[1];

	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	float dWarpX, dWarpY;
	int iTempX, iTempY;
	float dTempX, dTempY;
	float dtemptVal;
	int size = iSubsetH*iSubsetW;
	float *dSubsetT = whole_d_dSubsetT + iSubsetH*iSubsetW*bid;
	float *dSubsetAveT = whole_d_dSubsetAveT + (iSubsetH*iSubsetW + 1)*bid;
	float *dInvHessian = whole_d_dInvHessian + bid * 36;
	float2 *dRDescent = whole_d_2dRDescent + bid*iSubsetH*iSubsetW * 3;
	float *dSubsetAveR = whole_d_dSubsetAveR + bid*(iSubsetH*iSubsetW + 1);
	
	if (tid == 0)
	{
		// Transfer the initial guess to IC-GN algorithm
		dP[0] = d_iU[bid];
		dP[1] = 0;
		dP[2] = 0;
		dP[3] = d_iV[bid];
		dP[4] = 0;
		dP[5] = 0;

		// Initialize the warp matrix
		dWarp[0] = 1 + dP[1];
		dWarp[1] = dP[2];
		dWarp[2] = dP[0];
		dWarp[3] = dP[4];
		dWarp[4] = 1 + dP[5];
		dWarp[5] = dP[3];
	}

	if (tid == 32)
	{
		break_sig[0] = 0;
	}
	__syncthreads();

	int iIteration;
	for (iIteration = 0; iIteration < iMaxIteration; iIteration++)
	{
		float mySum = 0;
		for (int id = tid; id<size; id += dim)
		{
			int l = id / iSubsetW;
			int m = id%iSubsetW;
			if (l<iSubsetH&&m<iSubsetW)
			{
				dWarpX = d_dPXY[2 * bid + 1] + dWarp[0] * (m - iSubsetX) + dWarp[1] * (l - iSubsetY) + dWarp[2];
				dWarpY = d_dPXY[2 * bid] + dWarp[3] * (m - iSubsetX) + dWarp[4] * (l - iSubsetY) + dWarp[5];
				if (dWarpX<0)dWarpX = 0;
				if (dWarpY<0)dWarpY = 0;
				if (dWarpX >= iWidth)dWarpX = iWidth - 1;
				if (dWarpY >= iHeight)dWarpY = iHeight - 1;
				iTempX = int(dWarpX);
				iTempY = int(dWarpY);

				dTempX = dWarpX - iTempX;
				dTempY = dWarpY - iTempY;
				if ((dTempX == 0) && (dTempY == 0))
				{
					dtemptVal = dT[iTempY*iWidth + iTempX];
				}
				else
				{
					//unroll for loop
					float4 a1, a2, a3, a4;
					a1 = dTBicubic[0 * iWidth*iHeight + iTempY*iWidth + iTempX];//need
					a2 = dTBicubic[1 * iWidth*iHeight + iTempY*iWidth + iTempX];
					a3 = dTBicubic[2 * iWidth*iHeight + iTempY*iWidth + iTempX];
					a4 = dTBicubic[3 * iWidth*iHeight + iTempY*iWidth + iTempX];

					dtemptVal =
						a1.w * pow(dTempY, 0) * pow(dTempX, 0) +
						a1.x * pow(dTempY, 0) * pow(dTempX, 1) +
						a1.y * pow(dTempY, 0) * pow(dTempX, 2) +
						a1.z * pow(dTempY, 0) * pow(dTempX, 3) +

						a2.w * pow(dTempY, 1) * pow(dTempX, 0) +
						a2.x * pow(dTempY, 1) * pow(dTempX, 1) +
						a2.y * pow(dTempY, 1) * pow(dTempX, 2) +
						a2.z * pow(dTempY, 1) * pow(dTempX, 3) +

						a3.w * pow(dTempY, 2) * pow(dTempX, 0) +
						a3.x * pow(dTempY, 2) * pow(dTempX, 1) +
						a3.y * pow(dTempY, 2) * pow(dTempX, 2) +
						a3.z * pow(dTempY, 2) * pow(dTempX, 3) +

						a4.w * pow(dTempY, 3) * pow(dTempX, 0) +
						a4.x * pow(dTempY, 3) * pow(dTempX, 1) +
						a4.y * pow(dTempY, 3) * pow(dTempX, 2) +
						a4.z * pow(dTempY, 3) * pow(dTempX, 3);

				}
				dSubsetT[l*iSubsetW + m] = dtemptVal;
				mySum += dtemptVal / size;
			}
		}

		__syncthreads();
		
		float avg;
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
		__syncthreads();
		avg = sm[0];
		mySum = 0;
		for (int id = tid; id<size; id += dim)
		{
			dtemptVal = dSubsetT[id] - avg;
			mySum += dtemptVal*dtemptVal;
			dSubsetAveT[id + 1] = dtemptVal;
		}
		__syncthreads();
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
		__syncthreads();

		if (tid == 0)
		{

			dSubsetAveT[0] = dtemptVal = sqrt(sm[tid]);
			sm[tid] = dSubsetAveR[0] / dtemptVal;
		}

		float n0, n1, n2, n3, n4, n5;
		n0 = 0; n1 = 0; n2 = 0; n3 = 0; n4 = 0; n5 = 0;
		float2 rd;
		__syncthreads();
	
		float Nor = sm[0];
		for (int id = tid; id<size; id += dim)
		{
			dtemptVal = (Nor)* dSubsetAveT[id + 1] - dSubsetAveR[id + 1];
			rd = dRDescent[id];
			n0 += (rd.x * dtemptVal);
			n1 += (rd.y * dtemptVal);
			rd = dRDescent[size + id];
			n2 += (rd.x * dtemptVal);
			n3 += (rd.y * dtemptVal);
			rd = dRDescent[size * 2 + id];
			n4 += (rd.x * dtemptVal);
			n5 += (rd.y * dtemptVal);
		}


		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n0, tid);
		if (tid<6)
			n0 = sm[0];
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n1, tid);
		if (tid<6)
			n1 = sm[0];
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n2, tid);
		if (tid<6)
			n2 = sm[0];
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n3, tid);
		if (tid<6)
			n3 = sm[0];
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n4, tid);
		if (tid<6)
			n4 = sm[0];
		sumReduceBlock<BLOCK_SIZE_64, float>(sm, n5, tid);
		if (tid<6)
			n5 = sm[0];
		if (tid<6)
		{
			dDP[tid] =
				dInvHessian[tid * 6 + 0] * n0 +
				dInvHessian[tid * 6 + 1] * n1 +
				dInvHessian[tid * 6 + 2] * n2 +
				dInvHessian[tid * 6 + 3] * n3 +
				dInvHessian[tid * 6 + 4] * n4 +
				dInvHessian[tid * 6 + 5] * n5;
		}
		if (tid == 0)
		{
			dtemptVal = (1 + dDP[1]) * (1 + dDP[5]) - dDP[2] * dDP[4];
			dWarp[0] = ((1 + dP[1]) * (1 + dDP[5]) - dP[2] * dDP[4]) / dtemptVal;
			dWarp[1] = (dP[2] * (1 + dDP[1]) - (1 + dP[1]) * dDP[2]) / dtemptVal;
			dWarp[2] = dP[0] + (dP[2] * (dDP[0] * dDP[4] - dDP[3] - dDP[3] * dDP[1]) - (1 + dP[1]) * (dDP[0] * dDP[5] + dDP[0] - dDP[2] * dDP[3])) / dtemptVal;
			dWarp[3] = (dP[4] * (1 + dDP[5]) - (1 + dP[5]) * dDP[4]) / dtemptVal;
			dWarp[4] = ((1 + dP[5]) * (1 + dDP[1]) - dP[4] * dDP[2]) / dtemptVal;
			dWarp[5] = dP[3] + ((1 + dP[5]) * (dDP[0] * dDP[4] - dDP[3] - dDP[3] * dDP[1]) - dP[4] * (dDP[0] * dDP[5] + dDP[0] - dDP[2] * dDP[3])) / dtemptVal;

			// Update DeltaP
			dP[0] = dWarp[2];
			dP[1] = dWarp[0] - 1;
			dP[2] = dWarp[1];
			dP[3] = dWarp[5];
			dP[4] = dWarp[3];
			dP[5] = dWarp[4] - 1;

			if (sqrt(
				pow(dDP[0], 2) +
				pow(dDP[1] * iSubsetX, 2) +
				pow(dDP[2] * iSubsetY, 2) +
				pow(dDP[3], 2) +
				pow(dDP[4] * iSubsetX, 2) +
				pow(dDP[5] * iSubsetY, 2))
				< dNormDeltaP)
			{
				break_sig[0] = 1;
			}
		}
		__syncthreads();
		if (break_sig[0] == 1)
			break;
	}
	if (tid == 0)
	{
		whole_d_iIteration[bid] = iIteration;
	}
	if (tid<6)
	{
		whole_d_dP[bid * 6 + tid] = dP[tid];
	}
}


void all_gpu_main_icgn_process_all_iteration(ICGN_d_Handle &m_Handle,
	const int &m_iMaxIteration,
	const float &m_dNormDeltaP)
{
	icgn_kernel_all_iteration <<<m_Handle.m_iNumberX*m_Handle.m_iNumberY, BLOCK_SIZE_64 >> >(
		m_Handle.m_d_iU, m_Handle.m_d_iV,
		m_Handle.m_d_dPXY,
		m_Handle.m_iWidth, m_Handle.m_iHeight,
		m_Handle.m_iSubsetX, m_Handle.m_iSubsetY,
		m_Handle.m_iSubsetH, m_Handle.m_iSubsetW,
		m_iMaxIteration,
		m_dNormDeltaP,
		m_Handle.m_d_dT,
		m_Handle.m_d_dInvHessian,
		m_Handle.m_d_dTBicubic,
		m_Handle.m_d_2dRDescent,
		m_Handle.m_d_dSubsetAveR,
		//tempt val
		m_Handle.m_d_dSubsetT,
		m_Handle.m_d_dSubsetAveT,
		//return val
		m_Handle.m_d_iIterationNum,
		m_Handle.m_d_dP
		);
}

void all_gpu_cu_prepare_Hessian_all_iteration(ICGN_d_Handle &m_Handle)
{
	int m_iNumberX = m_Handle.m_iNumberX;
	int m_iNumberY = m_Handle.m_iNumberY;;
	int m_iNumSize = m_iNumberX*m_iNumberY;
	get_AveR_kernel_all_iteration << < m_iNumSize, BLOCK_SIZE_128 >> >
		(
			m_Handle.m_d_dR, m_Handle.m_d_dPXY,
			m_Handle.m_iSubsetH, m_Handle.m_iSubsetW,
			m_Handle.m_iSubsetX, m_Handle.m_iSubsetY,
			m_Handle.m_iHeight, m_Handle.m_iWidth,
			m_Handle.m_d_dSubsetR,
			m_Handle.m_d_dSubsetAveR
			);

	get_hessian_kernel_all_iteration << <m_iNumSize, BLOCK_SIZE_128 >> >(
		m_Handle.m_d_dRx, m_Handle.m_d_dRy,
		m_Handle.m_d_dPXY,
		m_Handle.m_iSubsetX, m_Handle.m_iSubsetY,
		m_Handle.m_iSubsetH, m_Handle.m_iSubsetW,
		m_Handle.m_iHeight, m_Handle.m_iWidth,
		m_Handle.m_d_2dRDescent,
		m_Handle.m_d_dInvHessian
		);

}

void cu_icgn(ICGN_d_Handle &m_Handle, const int m_iMaxIteration, const float m_dNormDeltaP)
{
	int *m_d_dPXY = m_Handle.m_d_dPXY;
	float *m_d_dR = m_Handle.m_d_dR;
	float *m_d_dT = m_Handle.m_d_dT;
	float *m_d_dRx = m_Handle.m_d_dRx;
	float *m_d_dRy = m_Handle.m_d_dRy;
	int *m_d_iU = m_Handle.m_d_iU;
	int *m_d_iV = m_Handle.m_d_iV;
	float4 *m_d_dTBicubic = m_Handle.m_d_dTBicubic;
	int m_iSubsetX = m_Handle.m_iSubsetX;
	int m_iSubsetY = m_Handle.m_iSubsetY;
	int m_iWidth = m_Handle.m_iWidth;
	int m_iHeight = m_Handle.m_iHeight;
	int m_iMarginX = m_Handle.m_iMarginX;
	int m_iMarginY = m_Handle.m_iMarginY;
	int m_iGridSpaceX = m_Handle.m_iGridSpaceX;
	int m_iGridSpaceY = m_Handle.m_iGridSpaceY;

	float *m_d_dP = m_Handle.m_d_dP;
	int *m_d_iIterationNum = m_Handle.m_d_iIterationNum;


	int m_iNumberX = m_Handle.m_iNumberX;
	int m_iNumberY = m_Handle.m_iNumberY;
	int m_iSubsetW = m_Handle.m_iSubsetW;
	int m_iSubsetH = m_Handle.m_iSubsetH;

	int m_iNumSize = m_iNumberX*m_iNumberY;
	int m_iSubwindowSize = m_iSubsetW*m_iSubsetH;

	float * w_d_dSubsetT = m_Handle.m_d_dSubsetT;
	float *w_d_dSubsetAveT = m_Handle.m_d_dSubsetAveT;
	float * w_d_dSubsetR = m_Handle.m_d_dSubsetR;
	float *w_d_dSubsetAveR = m_Handle.m_d_dSubsetAveR;
	float*w_d_dInvHessian = m_Handle.m_d_dInvHessian;
	float2* w_d_2dRDescent = m_Handle.m_d_2dRDescent;

	get_AveR_kernel_all_iteration <<<m_iNumSize, BLOCK_SIZE_128 >>>(m_d_dR, m_d_dPXY,
			m_iSubsetH, m_iSubsetW,
			m_iSubsetX, m_iSubsetY,
			m_iHeight, m_iWidth,
			w_d_dSubsetR,
			w_d_dSubsetAveR);
	get_hessian_kernel_all_iteration <<<m_iNumSize, BLOCK_SIZE_128 >>>(m_d_dRx, m_d_dRy,
		m_d_dPXY,
		m_iSubsetX, m_iSubsetY,
		m_iSubsetH, m_iSubsetW,
		m_iHeight, m_iWidth,
		w_d_2dRDescent,
		w_d_dInvHessian);
	icgn_kernel_all_iteration <<<m_iNumSize, BLOCK_SIZE_64 >>>(m_d_iU, m_d_iV, m_d_dPXY,
		m_iWidth, m_iHeight,
		m_iSubsetX, m_iSubsetY,
		m_iSubsetH, m_iSubsetW,
		m_iMaxIteration,
		m_dNormDeltaP,
		m_d_dT,
		w_d_dInvHessian,
		m_d_dTBicubic,
		w_d_2dRDescent,
		w_d_dSubsetAveR,
		//tempt val
		w_d_dSubsetT,
		w_d_dSubsetAveT,
		//return val
		m_d_iIterationNum,
		m_d_dP);
}