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
