#ifndef REDUCER_H
#define REDUCER_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

/*
	Implement the sum reduction within each block using shared memory
*/

template<unsigned int blockSize, class Real>
__device__ void
sumReduceBlock(volatile Real *sdata, Real mySum, const unsigned int tid)
{
	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared memory
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; }	__syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; }		__syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; }		__syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) { sdata[tid] = mySum = mySum + sdata[tid + 32]; }
		if (blockSize >= 32) { sdata[tid] = mySum = mySum + sdata[tid + 16]; }
		if (blockSize >= 16) { sdata[tid] = mySum = mySum + sdata[tid + 8]; }
		if (blockSize >= 8) { sdata[tid] = mySum = mySum + sdata[tid + 4]; }
		if (blockSize >= 4) { sdata[tid] = mySum = mySum + sdata[tid + 2]; }
		if (blockSize >= 2) { sdata[tid] = mySum = mySum + sdata[tid + 1]; }
	}
	__syncthreads();
}

#endif // !REDUCER_H
