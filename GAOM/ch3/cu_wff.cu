#include "cu_wff.h"

#define _USE_MATH_DEFINES
#include <math.h>

/*
  warp(32 threads) reduce is more efficient on post-kepler GPUs
  https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
*/
__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/* cufft Complex number scaling */
static	__device__ __host__
inline cufftComplex ComplexScale(cufftComplex a, float s)
{
	cufftComplex c;
	c.x = s*a.x;
	c.y = s*a.y;
	return c;
}

/* cufft Complex number multiplication */
static __device__ __host__
inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

/*---------------------------------------------CUDA Kernels-------------------------------------------------*/
/*
PURPOSE:
1. Generate the xf and yf for analytically computation of the Gaussian Window in Fourier Domain
[yf xf]=meshgrid(-fix(nn/2):nn-fix(nn/2)-1,-fix(mm/2):mm-fix(mm/2)-1); mm, nn are padded height&width
NOTE:
Currently only even size in each dimension is supported
INPUTS:
width, height: width and height of the xf and yf matrices
OUTPUTS:
d_out_xf, d_out_yf: Generated meshgrid
*/
__global__
void gen_xf_yf_Kernel(cufftReal *d_out_xf,
	cufftReal *d_out_yf,
	int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfy = iHeight / 2;
	int iHalfx = iWidth / 2;

	if (i < iHeight && j < iWidth)
	{
		d_out_xf[id] = j - iHalfx;
		d_out_yf[id] = i - iHalfy;
	}
}

/*
PURPOSE:
Point-wise multiplication of two matrices of complex numbers
INPUT:
d_in_a, d_in_b: Two matrices to be multiplied
iSize: size of the matrices
OUTPUT:
d_out_c: The result after multiplication
*/
__global__
void complex_pointwise_multiplication_kernel(cufftComplex *d_in_a,
	cufftComplex *d_in_b,
	int iSize,
	cufftComplex *d_out_c)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		i < iSize;
		i += blockDim.x*gridDim.x)
	{
		d_out_c[i] =
			ComplexScale(ComplexMul(d_in_a[i], d_in_b[i]), 1.0f / iSize);
	}
}

/*
PURPOSE:
Explicitly Compute the FFT of the Gaussian Window
Fg=exp(-(xf*2*pi/mm-wxt).^2/2*sigmax*sigmax - (yf*2*pi/nn-wyt).^2/2*sigmay*sigmay)*sn2;
INPUTS:
d_in_xf, d_in_yf: meshgrid in frequency domain
iPaddedWidth, iPaddedHeight: Padded Gaussian Window size
wxt, wyt: frequencies in integer intervals
wxi, wyi: steps
wxl, wyl: lower bound of the frequencies
sigmax, sigmay: sigma's in x&y directions
sn2: normalization params (norm2 = 1);
OUTPUTS:
d_out_Fg: Fg
*/
__global__
void compute_Fg_kernel(cufftReal *d_in_xf,
	cufftReal *d_in_yf,
	int iPaddedWidth, int iPaddedHeight,
	int wxt, int wyt, float wxi,
	float wyi, float wxl, float wyl,
	float sigmax, float sigmay,
	float sn2,
	cufftComplex *d_out_Fg)
{
	cufftReal rwxt = wxl + cufftReal(wxt) * wxi;
	cufftReal rwyt = wyl + cufftReal(wyt) * wyi;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < iPaddedHeight*iPaddedWidth;
		i += blockDim.x * gridDim.x)
	{
		cufftReal tempx = d_in_xf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedWidth) - rwxt;
		cufftReal tempy = d_in_yf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedHeight) - rwyt;
		tempx = -tempx * tempx * 0.5f * sigmax * sigmax;
		tempy = -tempy * tempy * 0.5f * sigmay * sigmay;

		d_out_Fg[i].x = exp(tempx + tempy) * sn2;
		d_out_Fg[i].y = 0;
	}
}

/*
PURPOSE:
2. Do the fftshift on xf and yf to be coincide with the CUFFT's results
NOTE:
Currently only even size in each dimension is supported
INPUTS:
width, height: width and height of the xf and yf matrices
OUTPUTS:
d_out_xf, d_out_yf: In-place fft-shifted xf, yf

*/
__global__
void fftshift_xf_yf_kernel(cufftReal *d_out_xf,
	cufftReal *d_out_yf,
	int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfx = iWidth / 2;
	int iHalfy = iHeight / 2;
	int iSlice = iWidth * iHeight;

	int idQ13 = iSlice / 2 + iHalfx;
	int idQ24 = iSlice / 2 - iHalfx;

	cufftReal Tempx, Tempy;

	if (j < iHalfx && i < iHalfy)
	{
		Tempx = d_out_xf[id];
		Tempy = d_out_yf[id];

		// First Quadrant
		d_out_xf[id] = d_out_xf[id + idQ13];
		d_out_yf[id] = d_out_yf[id + idQ13];

		// Third Quadrant
		d_out_xf[id + idQ13] = Tempx;
		d_out_yf[id + idQ13] = Tempy;
	}
	else if (j >= iHalfx && j < iWidth && i < iHalfy)
	{

		Tempx = d_out_xf[id];
		Tempy = d_out_yf[id];

		// Second Quadrant
		d_out_xf[id] = d_out_xf[id + idQ24];
		d_out_yf[id] = d_out_yf[id + idQ24];

		// Fourth Quadrant
		d_out_xf[id + idQ24] = Tempx;
		d_out_yf[id + idQ24] = Tempy;
	}
}

/*
PURPOSE:
Feed the input f into the Padded matrix m_d_fPadded
INPUTS:
d_in_f: The input fringe pattern
iWidth, iHeight: size of the d_f
iPaddedWidth, iPaddedHeight: FFT preferred size after padding
OUTPUTS:
d_out_fPadded: The padded d_f
*/
__global__
void feed_fPadded_kernel(cufftComplex *d_in_f,
	cufftComplex *d_out_fPadded,
	int iWidth, int iHeight,
	int iPaddedWidth, int iPaddedHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idImg = y * iWidth + x;
	int idPadded = y * iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iHeight && x < iWidth)
		{
			d_out_fPadded[idPadded].x = d_in_f[idImg].x;
			d_out_fPadded[idPadded].y = d_in_f[idImg].y;
		}
		else
		{
			d_out_fPadded[idPadded].x = 0;
			d_out_fPadded[idPadded].y = 0;
		}
	}
}

/*
PURPOSE:
Calculate the threshold value for the WFF if it's not specified using Parallel Reduction Algorithm
thr = 6*sqrt(mean2(abs(f).^2)/3);
INPUTS:
d_in:	 type of cufftComplex input array
size: size(width*height) of the in
OUTPUS:
d_out: 1-element device array
*/
__global__
void compute_WFF_threshold_kernel(cufftComplex *d_in, float *d_out, int size)
{
	float sum = float(0);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < size;
		i += blockDim.x*gridDim.x)
	{
		float abs = cuCabsf(d_in[i]);
		sum += abs*abs;
	}

	sum = warpReduceSum(sum);

	if (threadIdx.x % warpSize == 0)
		atomicAdd(d_out, sum);
}

/*
PURPOSE:
Initialize all WFF related matrices to 0's
INPUTS:
iWidth, iHeight: size of the final results
OUTPUTS:
d_out_filtered:
*/
__global__
void init_WFF_matrices_kernel(cufftComplex *d_out_filtered,
	int iWidth, int iHeight)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	int idImg = y * iWidth + x;

	if (y < iHeight && x < iWidth)
	{
		d_out_filtered[idImg].x = 0;
		d_out_filtered[idImg].y = 0;
	}
}

/*
PURPOSE:
Threshold the spectrum sf
INPUTS:
iWidth, iHeight: image size
iPaddedWidth, iPaddedHeight: Padded size
thr: the threshold
OUTPUTS:
d_out_sf: sf after threshold
*/
__global__
void threshold_sf_kernel(cufftComplex *d_out_sf,
	int iWidth, int iHeight,
	int iPaddedWidth, int iPaddedHeight,
	float thr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iHeight && x < iWidth)
		{
			if (cuCabsf(d_out_sf[idPadded]) < thr)
			{
				d_out_sf[idPadded].x = 0;
				d_out_sf[idPadded].y = 0;
			}
		}
		else
		{
			d_out_sf[idPadded].x = 0;
			d_out_sf[idPadded].y = 0;
		}
	}
}

/*
PURPOSE:
Update the partial results im_d_filtered of each stream
INPUTS:
iWidth, iHeight: image size
iPaddedWidth, iPaddedHeight: Padded size
d_in_im_sf: spectrum of each stream
OUTPUTS:
d_out_im_filtered: filtered image after of each stream
*/
__global__
void update_WFF_partial_filtered_kernel(cufftComplex *d_in_im_sf,
	int iWidth, int iHeight,
	int iPaddedWidth, int iPaddedHeight,
	cufftComplex *d_out_im_filtered)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;
	int idImg = y*iWidth + x;

	if (y < iHeight && x < iWidth)
	{
		d_out_im_filtered[idImg].x += d_in_im_sf[idPadded].x;
		d_out_im_filtered[idImg].y += d_in_im_sf[idPadded].y;
	}
}

/*
PURPOSE:
Update the final z.filtered
INPUTS:
d_in_im_filtered: the partial filtered results in each stream
imgSize: size of the fringe pattern
OUTPTS:
d_out_filtered: the final results
*/
__global__
void update_WFF_final_filtered_kernel(cufftComplex *d_in_im_filtered,
	int imgSize,
	cufftComplex *d_out_filtered)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		i < imgSize;
		i += blockDim.x * gridDim.x)
	{
		d_out_filtered[i].x += d_in_im_filtered[i].x;
		d_out_filtered[i].y += d_in_im_filtered[i].y;
	}
}

/*
PURPOSE:
Scale the final results
INPUTS:
d_out_filtered: the unscaled final results
imagSize: size of the fringe pattern
wxi,wyi: step size of the frequencies
OUTPUT:
d_out_filtered: scaled final results
*/
__global__
void scale_WFF_final_filtered_kernel(cufftComplex *d_out_filtered,
	int imgSize,
	float wxi, float wyi)
{
	float factor = 0.25f * (1.0f / float(M_PI*M_PI)) * wxi * wyi;

	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		i < imgSize;
		i += blockDim.x * gridDim.x)
	{
		d_out_filtered[i].x *= factor;
		d_out_filtered[i].y *= factor;
	}
}

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
	, m_fCoreTime(0)
	, m_fMemTime(0)
	, m_fPrecomputeTime(0)
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
	, m_fCoreTime(0)
	, m_fMemTime(0)
	, m_fPrecomputeTime(0)
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
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	cudaSafeFree(m_d_fPadded);
	cudaSafeFree(m_d_xf);
	cudaSafeFree(m_d_yf);

	cudaSafeFree(im_d_filtered);
	cudaSafeFree(im_d_Fg);
	cudaSafeFree(im_d_Sf);
	cudaSafeFree(m_d_rThr);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float t = 0;
	cudaEventElapsedTime(&t, start, end);
	m_fMemTime += t;

	std::cout << "paWFF Memory alloc/dealloc time is: " << m_fMemTime << " ms" << std::endl;

	cufftDestroy(m_planPadded);
}

void WFF2_CUDAF::operator()(cufftComplex *d_f,
	WFT2_DeviceResultsF &d_z,
	double &time)
{
	cuWFF2(d_f, d_z, time);
}

void WFF2_CUDAF::cuWFF2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{
	m_fCoreTime = 0;

	/* CUDA blocks & threads scheduling */
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocksPadded((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	dim3 blocksImg((m_iWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	int blocks1D = std::min((m_iPaddedWidth*m_iPaddedHeight + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);

	cudaEvent_t start, pre, end;
	cudaEventCreate(&start);
	cudaEventCreate(&pre);
	cudaEventCreate(&end);
	/* Set the threshold m_rThr if it's not specified by the client */

	cudaEventRecord(start);
	cuWFF2_SetThreashold(d_f);

	/* Feed the f to its padded version */
	cuWFF2_feed_fPadded(d_f);

	/* Pre-compute the FFT of m_d_fPadded */
	checkCudaErrors(cufftExecC2C(m_planPadded, m_d_fPadded, m_d_fPadded, CUFFT_FORWARD));

	/* Clear the results if they already contain last results */
	init_WFF_matrices_kernel << <blocksImg, threads >> >(d_z.m_d_filtered, m_iWidth, m_iHeight);
	getLastCudaError("init_WFF_matrices_kernel Launch Failed!");
	cudaEventRecord(pre);


	/* Insert this part inbetween to realize kind of CPU&GPU concurrent execution.
	map the wl: wi : wh interval to integers from  0 to size = (wyh - wyl)/wyi + 1 in order to divide the
	copmutations across threads, since threads indices are more conviniently controlled by integers 	    */
	int iwx = int((m_rWxh - m_rWxl)*(1 / m_rWxi)) + 1;
	int iwy = int((m_rWyh - m_rWyl)*(1 / m_rWyi)) + 1;

	init_WFF_matrices_kernel <<<blocksPadded, threads>>>(im_d_filtered, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("init_WFF_matrices_kernel Launch Failed!");


	for (int y = 0; y < iwy; y++)
	{
		// Now we have equal number of kernels executed in each stream
		for (int x = 0; x < iwx; x++)
		{			
			// Construct Fg
			compute_Fg_kernel <<<blocks1D, BLOCK_SIZE_256>> > (
				m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight,
				x, y, m_rWxi, m_rWyi, m_rWxl, m_rWyl,
				m_rSigmaX, m_rSigmaY, m_rGaussianNorm2, im_d_Fg);
			getLastCudaError("compute_Fg_kernel Launch Failed!");

			// Compute sf=ifft2(Ff.*Fg)
			complex_pointwise_multiplication_kernel <<<blocks1D, BLOCK_SIZE_256>> > (
				m_d_fPadded, im_d_Fg, m_iPaddedHeight*m_iPaddedWidth, im_d_Sf);
			getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
			checkCudaErrors(cufftExecC2C(m_planPadded, im_d_Sf, im_d_Sf, CUFFT_INVERSE));

			// Threshold the sf: sf=sf.*(abs(sf)>=thr); 
			threshold_sf_kernel <<<blocksPadded, threads>>> (im_d_Sf, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_rThr);
			getLastCudaError("threshold_sf_kernel Launch Failed!");

			// implement of IWFT: conv2(sf,w);
			checkCudaErrors(cufftExecC2C(m_planPadded, im_d_Sf, im_d_Sf, CUFFT_FORWARD));
			complex_pointwise_multiplication_kernel <<<blocks1D, BLOCK_SIZE_256 >>> (
				im_d_Sf, im_d_Fg, m_iPaddedHeight*m_iPaddedWidth, im_d_Sf);
			getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
			checkCudaErrors(cufftExecC2C(m_planPadded, im_d_Sf, im_d_Sf, CUFFT_INVERSE));

			// Update partial results im_d_filtered
			update_WFF_partial_filtered_kernel <<<blocksImg, threads >>> (im_d_Sf, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, im_d_filtered);
			getLastCudaError("update_WFF_partial_filtered_kernel Launch Failed!");
		}
		
	}

	update_WFF_final_filtered_kernel << <blocks1D, BLOCK_SIZE_256 >> >(im_d_filtered, m_iWidth*m_iHeight, d_z.m_d_filtered);
	getLastCudaError("update_WFF_final_filtered_kernel Launch Failed!");

	scale_WFF_final_filtered_kernel << <blocks1D, BLOCK_SIZE_256 >> >(d_z.m_d_filtered, m_iWidth*m_iHeight, m_rWxi, m_rWyi);
	getLastCudaError("scale_WFF_final_filtered_kernel Launch Failed!");

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	// Calculate the running time
	float t_precompute = 0;

	cudaEventElapsedTime(&t_precompute, start, pre);
	cudaEventElapsedTime(&m_fCoreTime, pre, end);

	m_fPrecomputeTime += t_precompute;

	cudaEventDestroy(start);
	cudaEventDestroy(pre);
	cudaEventDestroy(end);

	std::cout << "paWFF Precomputation Time: " << m_fPrecomputeTime << " ms" << std::endl;
	time = double(m_fCoreTime);
}

/* Private functions */
int WFF2_CUDAF::Initialize(WFT2_DeviceResultsF &d_z)
{
	cudaEvent_t start, ma, plan, end;
	cudaEventCreate(&start);
	cudaEventCreate(&ma);
	cudaEventCreate(&plan);
	cudaEventCreate(&end);

	/*----------------------------WFF&WFR Common parameters initialization-----------------------------*/
	// Half of the Gaussian Window size
	m_iSx = int(round(3 * m_rSigmaX));
	m_iSy = int(round(3 * m_rSigmaY));
	// Guassian Window Size
	m_iWinHeight = 2 * m_iSy + 1;
	m_iWinWidth = 2 * m_iSx + 1;

	// Calculate the initial padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + m_iWinHeight - 1;
	m_iPaddedWidth = m_iWidth + m_iWinWidth - 1;

	// Calculate the second padding in order to fit the optimized size for FFT
	int iH = getFirstGreater(m_iPaddedHeight);
	int iW = getFirstGreater(m_iPaddedWidth);
	if (-1 == iH || -1 == iW)
	{
		// Out of range
		return -1;
	}
	else
	{
		m_iPaddedHeight = OPT_FFT_SIZE[iH];
		m_iPaddedWidth = OPT_FFT_SIZE[iW];

		int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

		cudaEventRecord(start);
		/* Memory Preallocation on Device */
		// Allocate memory for input padded f which is pre-copmuted and remain unchanged
		checkCudaErrors(cudaMalloc((void**)&m_d_fPadded, sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&m_d_xf, sizeof(cufftReal)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&m_d_yf, sizeof(cufftReal)*iPaddedSize));
		cudaEventRecord(ma);

		/* Make the CUFFT plans */
		checkCudaErrors(cufftPlan2d(&m_planPadded, m_iPaddedHeight, m_iPaddedWidth, CUFFT_C2C));

		cudaEventRecord(plan);
		/* Construct the xf & yf */
		dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
		dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
		// Generate xf, yf
		gen_xf_yf_Kernel << <blocks, threads >> >(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("gen_xf_yf_Kernel Launch Failed!");

		// Shift xf, yf to match the FFT's results
		fftshift_xf_yf_kernel << <blocks, threads >> >(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("fftshift_xf_yf_kernel Launch Failed!");
		cudaEventRecord(end);
		cudaEventSynchronize(end);

		float f_time_MemAlloc = 0;
		float f_time_Pre = 0;
		
		cudaEventElapsedTime(&f_time_MemAlloc, start, ma);
		cudaEventElapsedTime(&f_time_Pre, plan, end);

		m_fMemTime += f_time_MemAlloc;
		m_fPrecomputeTime += f_time_Pre;

		cuWFF2_Init(d_z);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(ma);
	cudaEventDestroy(plan);
	cudaEventDestroy(end);

	return 0;
}

void WFF2_CUDAF::cuWFF2_Init(WFT2_DeviceResultsF &d_z)
{
	int iImageSize = m_iWidth * m_iHeight;
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	// Allocate memory for the final results
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_filtered, sizeof(cufftComplex)*iImageSize));

	// 1. Allocate memory for intermediate results
	checkCudaErrors(cudaMalloc((void**)&im_d_Fg, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_filtered, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_Sf, sizeof(cufftComplex)*iPaddedSize));
	
	// 2. Calculate threshold
	if (m_rThr <= 1e-6)
	{
		checkCudaErrors(cudaMalloc((void**)&m_d_rThr, sizeof(float)));
	}
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time = 0;

	cudaEventElapsedTime(&time, start, end);

	m_fMemTime += time;

	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void WFF2_CUDAF::cuWFF2_feed_fPadded(cufftComplex *d_f)
{
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);

	feed_fPadded_kernel <<<blocks, threads>>>(d_f, m_d_fPadded, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("feed_fPadded_kernel Launch Failed!");
}

void WFF2_CUDAF::cuWFF2_SetThreashold(cufftComplex *d_f)
{
	// Set the m_rThr if not set
	if (m_rThr < 0)
	{
		int iImgSize = m_iWidth * m_iHeight;

		// Launch the kernel to compute the threshold
		int blocks = std::min((iImgSize + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);
		compute_WFF_threshold_kernel << <blocks, BLOCK_SIZE_256 >> >(d_f, m_d_rThr, iImgSize);
		getLastCudaError("compute_WFF_threshold_kernel Launch Failed!");

		// Passing back to host
		checkCudaErrors(cudaMemcpy(&m_rThr, m_d_rThr, sizeof(float), cudaMemcpyDeviceToHost));
		m_rThr = 6 * sqrt(m_rThr *(1.0f / float(iImgSize)) / 3.0f);
	}
}