#ifndef UTILS_H
#define UTILS_H

#include "WFF.h"
#include <sstream>
#include <fstream>

/* CUFFT 2D Matrix I/O: matrix is stored in Row-Major */
bool cufftComplexMatRead2D(std::istream& in, cufftComplex *&f, int& rows, int& cols);
void cufftComplexMatWrite2D(std::ostream& out, cufftComplex *f, const int rows, const int cols);

/* Formatted output of CUFFT complex numbers */
void cufftComplexPrint(const cufftComplex& in);

#endif // !UTILS_H
