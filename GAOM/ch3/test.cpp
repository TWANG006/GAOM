#include "test.h"

#include "WFF.h"
#include "utils.h"
#include "cu_wff.h"

using namespace std;

void WFF_Test()
{
	/* Load the complex fringe pattern from disk */
	cufftComplex *f = nullptr;
	std::ifstream in("Test\\512.fp");
	int rows, cols;
	if (!cufftComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	in.close();

	/* Memory allocation for the restuls */
	cufftComplex *df;
	cudaMalloc((void**)&df, sizeof(cufftComplex)*cols*rows);
	cudaMemcpy(df, f, sizeof(cufftComplex)*cols*rows, cudaMemcpyHostToDevice);

	/* Test the parallel WFF algorithm */
	WFT2_DeviceResultsF z;
	WFF2_CUDAF cuwft(cols, rows,
		10, -1.0f, 1.0f, 0.1f,
		10, -1.0f, 1.0f, 0.1f, 5,
		z);
	double time = 0;
	cuwft(df, z, time);
	cout << "paWFF core algorithm Running time for " << rows << " x " << cols << " fringe pattern is " << time << " [ms]" << endl;

	/* Output results for verification */
	cufftComplex *h_zfiltered = (cufftComplex*)malloc(sizeof(cufftComplex) * cols * rows);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(h_zfiltered, z.m_d_filtered, sizeof(cufftComplex) *cols * rows, cudaMemcpyDeviceToHost));
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float f_timeTransfer = 0;
	cudaEventElapsedTime(&f_timeTransfer, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cout << "paWFF memory transfer time for " << " is " << f_timeTransfer << " [ms]" << endl;

	std::ofstream out;
	out.open("Test\\z_filtered_512.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (h_zfiltered[i * cols + j].y < 0)
				out << h_zfiltered[i * cols + j].x << h_zfiltered[i * cols + j].y << "i";
			else
				out << h_zfiltered[i * cols + j].x << "+" << h_zfiltered[i * cols + j].y << "i";
			if (j != cols - 1)
				out << ",";
		}
		if (i != rows - 1)
			out << "\n";
	}
	out.close();

	free(f);
	free(h_zfiltered);
	cudaFree(df);
}