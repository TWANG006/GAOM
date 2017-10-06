#ifndef COMPUTE_H
#define COMPUTE_H

#include "common.h"

struct ICGN_d_Handle
{
	//original
	int m_iImgWidth;
	int m_iImgHeight;
	int m_iSubsetX;
	int m_iSubsetY;
	int m_iMarginX;
	int m_iMarginY;
	int m_iGridSpaceX;
	int m_iGridSpaceY;
	//some cchange
	int m_iWidth;
	int m_iHeight;
	int m_iNumberX;
	int m_iNumberY;
	int m_iSubsetW;
	int m_iSubsetH;

	//gpu space
	float *m_d_dImg1;
	float *m_d_dImg2;
	float *m_d_dR;
	float *m_d_dRx;
	float *m_d_dRy;
	float *m_d_dT;
	float *m_d_dTx;
	float *m_d_dTy;
	float *m_d_dTxy;
	float4 *m_d_dTBicubic;
	int *m_d_dPXY;

	int * m_d_iU;
	int * m_d_iV;
	float * m_d_dZNCC;
	float* m_d_dP;
	int *m_d_iIterationNum;

	float * m_d_dSubsetT;
	float * m_d_dSubsetAveT;
	float * m_d_dSubsetR;
	float *m_d_dSubsetAveR;
	float *m_d_dInvHessian;
	float2* m_d_2dRDescent;
};

struct Timer
{
	double m_dDevMemAlloc;
	double m_dHosMemAlloc;
	double m_dDevMemFree;
	double m_dHosMemFree;

	double m_dMemCpy;

	double m_dConsumedTime;
	double m_dPrecomputeTime;
	double m_dICGNTime;
};

void icgn_algorithm(float ** m_dImg1, float ** m_dImg2,
	ICGN_d_Handle &m_Handle, const int &m_iMaxIteration, const float &m_dNormDeltaP,
	//return value
	int ***&m_dPXY, float ***&m_dP,
	int **&m_iIterationNum, int &m_iIteration,
	Timer& m_Time);

void icgn_prepare(ICGN_d_Handle &m_Handle, int ***&m_dPXY, float ***&m_dP, int ** &m_iIterationNum);

void icgn_finalize(int ***m_dPXY, float ***m_dP, int ** m_iIterationNum);

void icgn_gpu_prepare(ICGN_d_Handle& m_Handle,
	const int &m_iImgWidth, const int &m_iImgHeight,
	const int &m_iSubsetX, const int &m_iSubsetY,
	const int &m_iMarginX, const int &m_iMarginY,
	const int &m_iGridSpaceX, const int &m_iGridSpaceY,
	Timer&m_Time);

void icgn_gpu_finalize(ICGN_d_Handle &m_Handle, Timer& m_Time);

#endif // !COMPUTE_H
