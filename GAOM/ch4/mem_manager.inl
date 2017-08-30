#ifdef MEM_MANAGER_H

//!- Host memory allo/deallo-cation methods.
template<typename T>
void hcreateptr(T*& ptr, 
				size_t size)
{
	ptr = (T*)calloc(size, sizeof(T));
}
template<typename T>
void hcreateptr(T**& ptr, 
				size_t row, 
				size_t col)
{
	T * ptr1d = (T*)calloc(row*col, sizeof(T));
	ptr = (T**)malloc(row*sizeof(T*));

	for (int i = 0; i < row; i++)
	{
		ptr[i] = ptr1d + i*col;
	}
}
template<typename T>
void hcreateptr(T***& ptr, 
				size_t row, 
				size_t col, 
				size_t height)
{
	T *ptr1d = (T*)calloc(row*col*height, sizeof(T));
	T**ptr2d = (T**)malloc(row*col*sizeof(T*));
	ptr = (T***)malloc(row*sizeof(T**));

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			ptr2d[i*col + j] = ptr1d + (i*col + j)*height;
		}
		ptr[i] = ptr2d + i*col;
	}
}
template<typename T>
void hcreateptr(T****& ptr,
				size_t a, 
				size_t b, 
				size_t c, 
				size_t d)
{
	T *ptr1d = (T*)calloc(a*b*c*d, sizeof(T));
	T**ptr2d = (T**)malloc(a*b*c*sizeof(T*));
	T***ptr3d = (T***)malloc(a*b*sizeof(T**));
	ptr = (T****)malloc(a*sizeof(T***));

	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < b; j++)
		{
			for (int k = 0; k < c; k++)
			{
				ptr2d[(i*b + j)*c + k] = ptr1d + ((i*b + j)*c + k)*d;
			}
			ptr3d[i*b + j] = ptr2d + (i*b + j)*c;
		}
		ptr[i] = ptr3d + i*b;
	}
}

template<typename T>
void hdestroyptr(T*& ptr)
{
	free(ptr);
	ptr = nullptr;
}
template<typename T>
void hdestroyptr(T**& ptr)
{
	free(ptr[0]);
	free(ptr);
	ptr = nullptr;
}
template<typename T>
void hdestroyptr(T***& ptr)
{
	free(ptr[0][0]);
	free(ptr[0]);
	free(ptr);
	ptr = nullptr;
}
template<typename T>
void hdestroyptr(T****& ptr)
{
	free(ptr[0][0][0]);
	free(ptr[0][0]);
	free(ptr[0]);
	free(ptr);
	ptr = nullptr;
}

//!- Pinned host memory allo/deallo-cation methods
template<typename T>
void cucreateptr(T*& ptr, 
				size_t size)
{
	cudaHostAlloc((void**)&ptr, size*sizeof(T), cudaHostAllocDefault);
}
template<typename T>
void cucreateptr(T**& ptr, 
				 size_t row, 
				 size_t col)
{
	T * ptr1d;
	cudaHostAlloc((void**)&ptr1d, row*col*sizeof(T), cudaHostAllocDefault);
	ptr = (T**)malloc(row*sizeof(T*));

	for (int i = 0; i < row; i++)
	{
		ptr[i] = ptr1d + i*col;
	}
}
template<typename T>
void cucreateptr(T***& ptr, 
				 size_t row, 
				 size_t col, 
				 size_t height)
{
	T * ptr1d;
	cudaHostAlloc((void**)&ptr1d, row*col*height*sizeof(T), cudaHostAllocDefault);
	T**ptr2d = (T**)malloc(row*col*sizeof(T*));
	ptr = (T***)malloc(row*sizeof(T**));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			ptr2d[i*col + j] = ptr1d + (i*col + j)*height;
		}
		ptr[i] = ptr2d + i*col;
	}
}
template<typename T>
void cucreateptr(T****& ptr, 
				 size_t a, 
				 size_t b, 
				 size_t c, 
				 size_t d)
{
	T *ptr1d;
	cudaHostAlloc((void**)&ptr1d, a*b*c*d*sizeof(T), cudaHostAllocDefault);
	T**ptr2d = (T**)malloc(a*b*c*sizeof(T*));
	T***ptr3d = (T***)malloc(a*b*sizeof(T**));
	ptr = (T****)malloc(a*sizeof(T***));

	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < b; j++)
		{
			for (int k = 0; k < c; k++)
			{
				ptr2d[(i*b + j)*c + k] = ptr1d + ((i*b + j)*c + k)*d;
			}
			ptr3d[i*b + j] = ptr2d + (i*b + j)*c;
		}
		ptr[i] = ptr3d + i*b;
	}
}

template<typename T>
void cudestroyptr(T*&ptr)
{
	cudaFreeHost(ptr);
	ptr = nullptr;
}
template<typename T>
void cudestroyptr(T**&ptr)
{
	cudaFreeHost(ptr[0]);
	free(ptr);
	ptr = nullptr;
}
template<typename T>
void cudestroyptr(T***&ptr)
{
	cudaFreeHost(ptr[0][0]);
	free(ptr[0]);
	free(ptr);
	ptr = nullptr;
}
template<typename T>
void cudestroyptr(T****&ptr)
{
	cudaFreeHost(ptr[0][0][0]);
	free(ptr[0][0]);
	free(ptr[0]);
	free(ptr);
	ptr = nullptr;
}

template<typename T>
void cudaSafeFree(T*&ptr)
{
	if(ptr!=NULL && ptr!=0 && ptr!=nullptr)
	{
		checkCudaErrors(cudaFree(ptr));
		ptr = nullptr;
	}
}

#endif