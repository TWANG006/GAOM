#ifndef COMMON_H
#define COMMON_H


#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

#define BLOCK_SIZE_256 256
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_64 64
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#endif // !COMMON_H
