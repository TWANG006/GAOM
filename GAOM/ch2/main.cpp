#include <cuda_runtime.h>
#include "common.h"
#include "vec_add.h"
#include "vec_add.cuh"
#include "grad_calc.cuh"

using namespace std;
using namespace chrono;

void test_vecAdd()
{
	int n = 10000000;

	float *A, *B, *C;

	auto t1 = Time::now();
	cudaHostAlloc((void**)&A, n * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&B, n * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&C, n * sizeof(float), cudaHostAllocDefault);
	/*A = (float*)malloc(sizeof(float) * n);
	B = (float*)malloc(sizeof(float) * n);
	C = (float*)malloc(sizeof(float) * n);*/
	auto t2 = Time::now();
	fsec ma = t2 - t1;

	for (int i = 0; i < n; i++)
	{
		A[i] = float(i);
		B[i] = float(i);
		C[i] = 0;
	}

	auto t3 = Time::now();
	vecAdd_CPU(A, B, C, n);
	auto t4 = Time::now();
	fsec fs = t4 - t3;

	vecAdd(A, B, C, n);

	auto t5 = Time::now();
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	/*free(A); A = nullptr;
	free(B); B = nullptr;
	free(C); C = nullptr;*/
	auto t6 = Time::now();

	fsec md = t6 - t5;

	cout << "Vector size = " << n << endl;
	cout << "CPU Vector Addition overall running time: " << (ma + fs + md).count() * 1000.0 << " ms" << endl;
	cout << "CPU Vector Addition function running time: " << fs.count() * 1000.0 << " ms" << endl;
	cout << "Host memory allocation time: " << ma.count() * 1000.0 << " ms" << endl;
	cout << "Host memory deallocation time: " << md.count() * 1000.0 << " ms" << endl;
}

int main()
{
	test_vecAdd();
	gradientTest();

	return 0;
}