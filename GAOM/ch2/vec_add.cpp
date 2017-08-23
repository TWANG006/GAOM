#include "vec_add.h"

void vecAdd_CPU(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
	}
}