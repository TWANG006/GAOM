#ifndef ICGN_CUH
#define ICGN_CUH

#include "common.h"
#include "compute.h"

/*
@INPUTS:
  d_iU, d_iV: initial integer-displacement of each POI
  d_POIxy: POI positions
  d_R, d_Rx, d_Ry: reference ROI and its gradients
  d_T, d_Tx, d_Ty, d_Txy: target ROI and its gradients
  d_BicubicLUT: LUT of bicubic coefficients
  iMaxIterations: 20
  fNormDeltaP: 0.001
@OUTPUTS:
  d_w_iterationNum: number of iterations of each subset
  d_P: deformation vector P[6]
*/
void cu_icgn(ICGN_d_Handle &m_Handle, const int iMaxIterations, const float fNormDeltaP);
void all_gpu_main_icgn_process_all_iteration(ICGN_d_Handle &m_Handle,
	const int &m_iMaxIteration,
	const float &m_dNormDeltaP);
void all_gpu_cu_prepare_Hessian_all_iteration(ICGN_d_Handle &m_Handle);
#endif // ICGN_CUH
