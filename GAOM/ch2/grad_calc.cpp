#include "grad_calc.h"

void gradient(float *img, int iImgWidth, int iImgHeight, float *Gx, float *Gy, float *Gxy)
{
	int iWidth = iImgWidth - 2;
	int iHeight = iImgHeight - 2;

	for (int i = 1; i < iImgHeight - 1; i++)
	{
		for (int j = 1; j < iImgWidth - 1; j++)
		{			
			int id = (i - 1) * iWidth + j - 1; // Gradient id
			Gx[id] = 0.5 * (img[i*iImgWidth + j + 1] - img[i*iImgWidth + j - 1]);
			Gy[id] = 0.5 * (img[(i + 1)*iImgWidth + j] - img[(i - 1)*iImgWidth + j]);
			Gxy[id] = 0.25 * (img[(i + 1)*iImgWidth + j + 1] - img[(i - 1)*iImgWidth + j + 1]
				- img[(i + 1)*iImgWidth + j - 1] + img[(i - 1)*iImgWidth + j - 1]);
		}
	}
}