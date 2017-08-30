#include "cu_wff.h"

#define _USE_MATH_DEFINES
#include <math.h>

/*--------------------------------------------------WFT2 Implementations-----------------------------------------------*/
WFF2_CUDAF::WFF2_CUDAF(int iWidth, int iHeight, 
	WFT2_DeviceResultsF& z)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_rThr(-1)
	, m_d_rThr(nullptr)
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
	, im_d_Fg(nullptr)
	, im_d_filtered(nullptr)
{
	// Get the number of SMs on GPU 
	cudaDeviceGetAttribute(&m_iSMs, cudaDevAttrMultiProcessorCount, 0);

	// WFF default parameters initialization
	m_rSigmaX = 10.0f;
	m_rWxl = -2.0f - 3.0f / m_rSigmaX;
	m_rWxi = 1.0f / m_rSigmaX;
	m_rWxh = 2.0f + 3.0f / m_rSigmaX;

	m_rSigmaY = 10.0f;
	m_rWyl = -2.0f - 3.0f / m_rSigmaY;
	m_rWyi = 1.0f / m_rSigmaY;
	m_rWyh = 2.0f + 3.0f / m_rSigmaY;

	// scale for window so that norm2 of the window is 1. 
	m_rGaussianNorm2 = sqrt(4 * float(M_PI)*m_rSigmaX*m_rSigmaY);
}

WFF2_CUDAF::WFF2_CUDAF(int iWidth, int iHeight,
	float rSigmaX, float rWxl, float rWxh, float rWxi,
	float rSigmaY, float rWyl, float rWyh, float rWyi,
	float rThr,
	WFT2_DeviceResultsF &z)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_rSigmaX(rSigmaX)
	, m_rSigmaY(rSigmaY)
	, m_rWxl(rWxl)
	, m_rWxi(rWxi)
	, m_rWxh(rWxh)
	, m_rWyl(rWxl)
	, m_rWyi(rWyi)
	, m_rWyh(rWyh)
	, m_rThr(rThr)
	, m_d_rThr(nullptr)
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
	, im_d_Fg(nullptr)
	, im_d_filtered(nullptr)
{
	// Get the number of SMs on GPU 
	cudaDeviceGetAttribute(&m_iSMs, cudaDevAttrMultiProcessorCount, 0);

	// scale for window so that norm2 of the window is 1. 
	m_rGaussianNorm2 = sqrt(4 * float(M_PI)*m_rSigmaX*m_rSigmaY);

	/* Do the Initialization */
	if (-1 == Initialize(z))
	{
		std::cout << "FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!" << std::endl;
		throw - 1;
	}
}

WFF2_CUDAF::~WFF2_CUDAF()
{
	cudaSafeFree(m_d_fPadded);
	cudaSafeFree(m_d_xf);
	cudaSafeFree(m_d_yf);

	cufftDestroy(m_planPadded);

	cudaSafeFree(im_d_filtered);
	cudaSafeFree(im_d_Fg);
	cudaSafeFree(im_d_Sf);
	cudaSafeFree(m_d_rThr);
}

void WFF2_CUDAF::operator()(cufftComplex *d_f,
	WFT2_DeviceResultsF &d_z,
	double &time)
{
	cuWFF2(d_f, d_z, time);
}

void WFF2_CUDAF::cuWFF2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{

}

/* Private functions */