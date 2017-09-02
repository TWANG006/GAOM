#ifndef CU_WFF_H
#define CU_WFF_H

#include "WFF.h"

class  WFF2_CUDAF
{
public:

	WFF2_CUDAF() = delete;
	WFF2_CUDAF(const WFF2_CUDAF&) = delete;
	WFF2_CUDAF &operator=(const WFF2_CUDAF&) = delete;


	// Default parameters are used based on the WFT_TYPE
	WFF2_CUDAF(int iWidth, int iHeight,
		WFT2_DeviceResultsF& z);

	// Parameters are set by the input parameters
	WFF2_CUDAF(int iWidth, int iHeight,
		float rSigmaX, float rWxl, float rWxh, float rWxi,
		float rSigmaY, float rWyl, float rWyh, float rWyi,
		float rThr,
		WFT2_DeviceResultsF &z);

	~WFF2_CUDAF();

	// Make this class a callable object (functor)
	void operator() (cufftComplex *d_f,
		WFT2_DeviceResultsF &d_z,
		double &time);

private:
	/* Initilaize the WFT2 algorithm
	1. Calculate the padding size.
	2. Allocate&initialize arrays with the padded size
	3. Make cufft plans								   */
	int Initialize(WFT2_DeviceResultsF &d_z);
	void cuWFF2_Init(WFT2_DeviceResultsF &d_z);

	/* Feed the f into its padded m_d_fPadded */
	void cuWFF2_feed_fPadded(cufftComplex *d_f);

	// Set the threashold of the WFF2 algorithm if the initial value of m_rThr = -1;
	void cuWFF2_SetThreashold(cufftComplex *d_f);

	/* CUDA WFF Algorithms */
	void cuWFF2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time);

private:
	/* Internal Arrays */
	cufftComplex	*m_d_fPadded;				// Padded f
	cufftReal		*m_d_xf;					// Explicit Freq in x for Gaussian Window
	cufftReal		*m_d_yf;					// Explicit Freq in y for Gaussian Window

	cufftHandle		m_planPadded;

	/* Intermediate Results */
	cufftComplex	*im_d_Fg;					// Explicitly computed Fg in Fourier Domain
	cufftComplex	*im_d_Sf;

	/* WFF Intermediate Results for each CUDA Stream */
	cufftComplex	*im_d_filtered;

	/* Internal Parameters */
	int				m_iWidth;					// width of the fringe pattern
	int				m_iHeight;					// height of the fringe pattern	

	/* Initially, size(A) + size(B) - 1, search the lookup table for *
	 * Optimized size for the FFT									 */
	int				m_iPaddedWidth;				// width after padding for optimized FFT
	int				m_iPaddedHeight;			// height after padding for optimized FFT

	int				m_iPaddedWidthCurvature;	// width after padding for optimized FFT for curvature computation
	int				m_iPaddedHeightCurvature;	// height after padding for optimized FFT for curvature computation

	int				m_iSx;						// half Windows size along x
	int				m_iSy;						// half Windows size along y
	int				m_iWinWidth;				// Gaussian Window width
	int				m_iWinHeight;				// Gaussian Window height
	float			m_rSigmaX;					// sigma of the window in x-axis
	float			m_rWxl;						// lower bound of frequency in x-axis
	float			m_rWxi;						// step size of frequency in x-axis
	float			m_rWxh;						// upper bound of frequency in x-axis
	float			m_rSigmaY;					// sigma of the window in x-axis
	float			m_rWyl;						// lower bound of frequency in y-axis
	float			m_rWyh;						// upper bound of frequency in y-axis
	float			m_rWyi;						// step size of frequency in y-axis	
	float			m_rGaussianNorm2;			// L2-norm normalization of Gaussian Window
	float			m_rSumxxg;					// 1 / x.*x.*g
	float			m_rSumyyg;					// 1 / y.*y.*g

	/* threshold for 'wff'						*
	 * NOTE: if m_rThr < 0, it is calculated as *
	 * m_rThr = 6 * sqrt(mean2(abs(f).^2)/3)    */
	float			m_rThr;
	float			*m_d_rThr;

	/* Parameters for Thread control */
	int m_iSMs;							// Number of SMs of the device 0
};

#endif // !CU_WFF_H
