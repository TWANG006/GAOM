#include "utils.h"

bool cufftComplexMatRead2D(std::istream& in, cufftComplex *&f, int& rows, int& cols)
{
	std::string line;					// Real line-by-line
	std::getline(in, line);				// Read the first line to get the size
	std::stringstream lineStream(line);	// read the line into stirng stream
	std::string cell;					// string of each cell in one line
	std::vector<int> size;				// size information
	std::vector<float> data;			// data(real,imag) of each line

	while (std::getline(lineStream, cell, ','))
	{
		size.push_back(std::stoi(cell));
	}
	// Check if the size is read already
	if (size.size() != 2)
		return false;
	rows = size[0];
	cols = size[1];

	// Allocate memory for f
	f = (cufftComplex*)malloc(sizeof(cufftComplex) * rows * cols);

	int i = 0;	// Index

	while (std::getline(in, line))
	{
		data.clear();
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ','))
		{
			data.push_back(std::stof(cell));
		}
		if (data.size() != 2)
			return false;

		f[i].x = data[0];
		f[i].y = data[1];

		i++;
	}
	return true;
}

void cufftComplexMatWrite2D(std::ostream& out, cufftComplex *f, const int rows, const int cols)
{
	// First line contains total rows & cols of the matrix
	out << rows << "," << cols << "\n";
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			int id = y * cols + x;
			out << f[id].x << "," << f[id].y << "\n";
		}
	}
}

void cufftComplexPrint(const cufftComplex& in)
{
	std::cout << in.x << "+" << "(" << in.y << "i)";
}
