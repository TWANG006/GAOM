#ifndef WFF_H
#define WFF_H


#include <chrono>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "mem_manager.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

const int BLOCK_SIZE_256 = 256;
const int BLOCK_SIZE_128 = 128;
const int BLOCK_SIZE_64 = 64;
const int BLOCK_SIZE_16 = 16;

/* This is the LUT of the optimized FFT size based on the following 3  *
* features:														   *
*		1. The size is restricted up to 2048 in one dimension          *
*		2. The size satisfies: 2^a * 3^b * 5^c * 7^d, optimal for both *
*		   FFTW3 and CUFFT											   *
*		3. All the sizes are even numbers, eliminate the indexing bias *
*         of the FFT shift											   */
const std::vector<int> OPT_FFT_SIZE = {
	2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 50,
	54, 56, 60, 64, 70, 72, 80, 84, 90, 96, 98, 100, 108, 112, 120, 126, 128,
	140, 144, 150, 160, 162, 168, 180, 192, 196, 200, 210, 216, 224, 240, 250,
	252, 256, 270, 280, 288, 294, 300, 320, 324, 336, 350, 360, 378, 384, 392,
	400, 420, 432, 448, 450, 480, 486, 490, 500, 504, 512, 540, 560, 576, 588,
	600, 630, 640, 648, 672, 686, 700, 720, 750, 756, 768, 784, 800, 810, 840,
	864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1050, 1080, 1120, 1134,
	1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458,
	1470, 1500, 1512, 1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800,
	1890, 1920, 1944, 1960, 2000, 2016, 2048, 2058, 2100, 2160, 2240, 2250, 2268,
	2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688, 2700, 2744,
	2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402,
	3430, 3456, 3500, 3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032,
	4050, 4096 };

/* Sginle precision Data structure for CUDA WFT Results */
struct WFT2_DeviceResultsF
{
	/* For WFF */
	// 2D filtered signal
	// phase = angle(m_h_filtered) for fI and fIII
	// intensity = real(m_h_filtered) for fIII and fIV
	cufftComplex *m_d_filtered;

	/* for WFR */
	float *m_d_wx;			// local frequency in x
	float *m_d_wy;			// local frequency in y
	float *m_d_phase;		// phase
	float *m_d_phase_comp;	// compensated by estimation of c
	float *m_d_b;			// amplitude
	float *m_d_r;			// ridge value
	float *m_d_cxx;			// estimation of c in x
	float *m_d_cyy;			// estimation of c in y

	WFT2_DeviceResultsF();
	~WFT2_DeviceResultsF();
};

/*--------------------------Utility Functions----------------------------*/
/* FUNCTION: int getFirstGreater(int x) */
/* PURPOSE
*	Find the first element in the array OPT_FFT_SIZE that is greater than or *
*	equal to the input x. Use Binary Search since OPT_FFT_SIZE is ordered.   */
int getFirstGreater(int x);

#endif // !WFF_H
