#include "common.h"
#include "precomputation.cuh"
#include "icgn.cuh"
#include "TW_MemManager.h"
#include <omp.h>


void icgn_algorithm(float ** m_dImg1, float ** m_dImg2,
	ICGN_d_Handle &m_Handle, const int &m_iMaxIteration, const float &m_dNormDeltaP,
	//return value
	int ***&m_dPXY,	float ***&m_dP,
	int **&m_iIterationNum, int &m_iIteration,
	Timer& m_Time)
{
	double t_whole = omp_get_wtime();

	int m_iImgWidth = m_Handle.m_iImgWidth;
	int m_iImgHeight = m_Handle.m_iImgHeight;
	int m_iSubsetX = m_Handle.m_iSubsetX;
	int m_iSubsetY = m_Handle.m_iSubsetY;
	int m_iMarginX = m_Handle.m_iMarginX;
	int m_iMarginY = m_Handle.m_iMarginY;
	int m_iGridSpaceX = m_Handle.m_iGridSpaceX;
	int m_iGridSpaceY = m_Handle.m_iGridSpaceY;

	int m_iWidth = m_Handle.m_iWidth;
	int m_iHeight = m_Handle.m_iHeight;

	//Estimate the number of points of interest(POIs)
	int m_iNumberX = m_Handle.m_iNumberX;
	int m_iNumberY = m_Handle.m_iNumberY;
	int m_iSubsetW = m_Handle.m_iSubsetW;
	int m_iSubsetH = m_Handle.m_iSubsetH;

	int m_iNumSize = m_iNumberX*m_iNumberY;
	int m_iSubwindowSize = m_iSubsetW*m_iSubsetH;
	//Calculate the coefficients of bicubic spline Interpolation (natural boundary condition) in the target image
	//Define the size of region of interest (ROI)

	float *m_d_dImg1 = m_Handle.m_d_dImg1;
	float *m_d_dImg2 = m_Handle.m_d_dImg2;
	float *m_d_dR = m_Handle.m_d_dR;
	float *m_d_dRx = m_Handle.m_d_dRx;
	float *m_d_dRy = m_Handle.m_d_dRy;
	float *m_d_dT = m_Handle.m_d_dT;
	float *m_d_dTx = m_Handle.m_d_dTx;
	float *m_d_dTy = m_Handle.m_d_dTy;
	float *m_d_dTxy = m_Handle.m_d_dTxy;
	float4 *m_d_dTBicubic = m_Handle.m_d_dTBicubic;
	int *m_d_dPXY = m_Handle.m_d_dPXY;

	int * m_d_iU = m_Handle.m_d_iU;
	int * m_d_iV = m_Handle.m_d_iV;

	// NOTE: initialize m_d_iU, m_d_iV to 0's. In real applciations, these can be 
	// substituted with more accurate initial values
	cudaMemset(m_d_iU, 0, sizeof(int)*m_iNumSize);
	cudaMemset(m_d_iV, 0, sizeof(int)*m_iNumSize);

	float* w_d_dP = m_Handle.m_d_dP;
	int *m_d_iIterationNum = m_Handle.m_d_iIterationNum;
	cudaEvent_t start, cp1, pre, icgn, cp2;
	cudaEventCreate(&start);
	cudaEventCreate(&cp1);
	cudaEventCreate(&pre);
	cudaEventCreate(&icgn);
	cudaEventCreate(&cp2);
	
	// Copy the images to device
	cudaEventRecord(start);
	cudaMemcpyAsync(m_d_dImg1, m_dImg1[0], sizeof(float)*m_iImgWidth*m_iImgHeight, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(m_d_dImg2, m_dImg2[0], sizeof(float)*m_iImgWidth*m_iImgHeight, cudaMemcpyHostToDevice);
	cudaEventRecord(cp1);

	// Precomputation starts
	cu_precomputation(m_d_dImg1, m_d_dImg2,
		m_Handle);
	cudaEventRecord(pre);

	// ICGN
	all_gpu_cu_prepare_Hessian_all_iteration(m_Handle);
	all_gpu_main_icgn_process_all_iteration(m_Handle,
		m_iMaxIteration,
		m_dNormDeltaP);
	cudaEventRecord(icgn);

	cudaMemcpyAsync(m_iIterationNum[0], m_d_iIterationNum, sizeof(int)*m_iNumSize, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(m_dP[0][0], w_d_dP, sizeof(float)*m_iNumSize * 6, cudaMemcpyDeviceToHost);
	m_iIteration = 0;
	cudaMemcpyAsync(m_dPXY[0][0], m_Handle.m_d_dPXY, sizeof(int)*m_iNumSize * 2, cudaMemcpyDeviceToHost);
	cudaEventRecord(cp2);
	cudaDeviceSynchronize();
	for (int i = 0; i < m_iNumberY; i++)
	{
		for (int j = 0; j < m_iNumberX; j++)
		{
			m_iIteration += m_iIterationNum[i][j];
		}
	}
	float t_cp1, t_pre, t_icgn, t_cp2;

	cudaEventElapsedTime(&t_cp1, start, cp1);
	cudaEventElapsedTime(&t_pre, cp1, pre);
	cudaEventElapsedTime(&t_icgn, pre, icgn);
	cudaEventElapsedTime(&t_cp2, icgn, cp2);

	t_whole = (omp_get_wtime() - t_whole) * 1000;
	m_Time.m_dConsumedTime = t_whole;
	m_Time.m_dMemCpy = t_cp1 + t_cp2;
	m_Time.m_dPrecomputeTime = t_pre;
	m_Time.m_dICGNTime = t_icgn;
}


void icgn_prepare(ICGN_d_Handle &m_Handle, int ***&m_dPXY, float ***&m_dP, int ** &m_iIterationNum)
{
	int m_iWidth = m_Handle.m_iWidth;
	int m_iHeight = m_Handle.m_iHeight;

	//Estimate the number of points of interest(POIs)
	int m_iNumberX = m_Handle.m_iNumberX;
	int m_iNumberY = m_Handle.m_iNumberY;
	int m_iSubsetW = m_Handle.m_iSubsetW;
	int m_iSubsetH = m_Handle.m_iSubsetH;

	int m_iNumSize = m_iNumberX*m_iNumberY;
	int m_iSubwindowSize = m_iSubsetW*m_iSubsetH;

	//Initialize the parameters for IC-GN algorithm
	cucreateptr(m_iIterationNum, m_iNumberY, m_iNumberX); // iteration step taken at each POI
	cucreateptr(m_dPXY, m_iNumberY, m_iNumberX, 2);//location of each POI in the global coordinate system
	cucreateptr(m_dP, m_iNumberY, m_iNumberX, 6); // parameter of deformation p
}

void icgn_finalize(int ***m_dPXY, float ***m_dP, int ** m_iIterationNum)
{
	cudestroyptr(m_dPXY);
	cudestroyptr(m_dP);
	cudestroyptr(m_iIterationNum);
}

void icgn_gpu_prepare(ICGN_d_Handle& m_Handle,
	const int &m_iImgWidth, const int &m_iImgHeight,
	const int &m_iSubsetX, const int &m_iSubsetY,
	const int &m_iMarginX, const int &m_iMarginY,
	const int &m_iGridSpaceX, const int &m_iGridSpaceY, 
	Timer &m_Time)
{
	m_Handle.m_iImgWidth = m_iImgWidth;
	m_Handle.m_iImgHeight = m_iImgHeight;
	m_Handle.m_iSubsetX = m_iSubsetX;
	m_Handle.m_iSubsetY = m_iSubsetY;
	m_Handle.m_iMarginX = m_iMarginX;
	m_Handle.m_iMarginY = m_iMarginY;
	m_Handle.m_iGridSpaceX = m_iGridSpaceX;
	m_Handle.m_iGridSpaceY = m_iGridSpaceY;


	m_Handle.m_iWidth = m_iImgWidth - 2; // set margin = 1 column
	m_Handle.m_iHeight = m_iImgHeight - 2; // set margin = 1 row

	//Estimate the number of points of interest(POIs)
	m_Handle.m_iNumberX = int(floor((m_Handle.m_iWidth - m_iSubsetX * 2 - m_iMarginX * 2) / float(m_iGridSpaceX))) + 1;
	m_Handle.m_iNumberY = int(floor((m_Handle.m_iHeight - m_iSubsetY * 2 - m_iMarginY * 2) / float(m_iGridSpaceY))) + 1;
	m_Handle.m_iSubsetW = m_iSubsetX * 2 + 1;
	m_Handle.m_iSubsetH = m_iSubsetY * 2 + 1;

	int m_iImgsize = m_Handle.m_iHeight*m_Handle.m_iWidth;
	int m_iNumSize = m_Handle.m_iNumberY*m_Handle.m_iNumberX;
	int m_iSubwindowSize = m_Handle.m_iSubsetW*m_Handle.m_iSubsetH;

	double t_start = omp_get_wtime();
	cudaMalloc((void**)&m_Handle.m_d_dImg1, sizeof(float)*m_iImgWidth*m_iImgHeight);
	cudaMalloc((void**)&m_Handle.m_d_dImg2, sizeof(float)*m_iImgWidth*m_iImgHeight);
	cudaMalloc((void**)&(m_Handle.m_d_dR), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dRx), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dRy), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dT), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dTx), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dTy), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dTxy), sizeof(float)*m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dTBicubic), sizeof(float4) * 4 * m_iImgsize);
	cudaMalloc((void**)&(m_Handle.m_d_dPXY), sizeof(int)*m_iImgsize * 2);

	cudaMalloc((void**)&m_Handle.m_d_dP, sizeof(float)*m_iNumSize * 6);
	cudaMalloc((void**)&m_Handle.m_d_iIterationNum, sizeof(int)*m_iNumSize);
	cudaMalloc((void**)&m_Handle.m_d_iU, sizeof(int)*m_iNumSize);
	cudaMalloc((void**)&m_Handle.m_d_iV, sizeof(int)*m_iNumSize);
	cudaMalloc((void**)&m_Handle.m_d_dZNCC, sizeof(float)*m_iNumSize);

	//icgn temptval
	cudaMalloc((void**)&m_Handle.m_d_dSubsetT, sizeof(float)*m_iNumSize*m_iSubwindowSize);
	cudaMalloc((void**)&m_Handle.m_d_dSubsetAveT, sizeof(float)*m_iNumSize*(m_iSubwindowSize + 1));
	cudaMalloc((void**)&m_Handle.m_d_dSubsetR, sizeof(float)*m_iNumSize*m_iSubwindowSize);
	cudaMalloc((void**)&m_Handle.m_d_dSubsetAveR, sizeof(float)*m_iNumSize*(m_iSubwindowSize + 1));
	cudaMalloc((void**)&m_Handle.m_d_dInvHessian, sizeof(float)*m_iNumSize * 36);
	cudaMalloc((void**)&m_Handle.m_d_2dRDescent, sizeof(float2) * 3 * m_iSubwindowSize*m_iNumSize);
	m_Time.m_dDevMemAlloc = (omp_get_wtime() - t_start) * 1000;
	t_start = omp_get_wtime();
}

void icgn_gpu_finalize(ICGN_d_Handle &m_Handle, Timer& m_Time)
{
	double t_start = omp_get_wtime();
	cudaFree(m_Handle.m_d_dImg1);
	cudaFree(m_Handle.m_d_dImg2);
	cudaFree(m_Handle.m_d_dR);
	cudaFree(m_Handle.m_d_dRx);
	cudaFree(m_Handle.m_d_dRy);
	cudaFree(m_Handle.m_d_dT);
	cudaFree(m_Handle.m_d_dTx);
	cudaFree(m_Handle.m_d_dTy);
	cudaFree(m_Handle.m_d_dTxy);
	cudaFree(m_Handle.m_d_dTBicubic);
	cudaFree(m_Handle.m_d_dPXY);

	cudaFree(m_Handle.m_d_iU);
	cudaFree(m_Handle.m_d_iV);
	cudaFree(m_Handle.m_d_dZNCC);
	cudaFree(m_Handle.m_d_dP);
	cudaFree(m_Handle.m_d_iIterationNum);
	cudaFree(m_Handle.m_d_dSubsetT);
	cudaFree(m_Handle.m_d_dSubsetAveT);
	cudaFree(m_Handle.m_d_dSubsetR);
	cudaFree(m_Handle.m_d_dSubsetAveR);
	cudaFree(m_Handle.m_d_dInvHessian);
	cudaFree(m_Handle.m_d_2dRDescent);
	m_Time.m_dDevMemFree = (omp_get_wtime() - t_start) * 1000;
	t_start = omp_get_wtime();
}