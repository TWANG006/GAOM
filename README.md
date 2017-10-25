# GAOM - GPU-Acceleration for Optical Methods
## Introduction
This is the source code associated with the book *GPU-Acceleration for Optical Methods*, SPIE Spotlight, under peer review. The details within each project folder are explained as below.
  * ch2:
    * vec_add.h(.cpp)   CPU and GPU implementations of vector addition
    * grad_calc.h(.cpp) Pointwise and tiled implementations of image gradient calculation
  * ch3:
    * cu_wff.cu Parallel windowed Fourier filtering (WFF) algorithm
  * ch4:
    * icgn.cu           Parallel inverse Compositional Gauss-Newton (IC-GN) algorithm
    * precomputation.cu Parallelized precomputation for IC-GN
    * compute.cu        Helper functions for the above CUDA kernels
## Use the code
### Project dependencies:
  * [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive) for writing GPU code
  * [OpenCV 3.1](https://opencv.org/opencv-3-1.html) for convenient image I/O
### Compilation on Windows
  * Download and install [Visual Studio 2015+](https://www.visualstudio.com/vs/older-downloads/)
  * Download and install [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
  * Install OpenCV 3.1:
    1. Download [OpenCV 3.1](https://opencv.org/opencv-3-1.html)
    2. Right click "Computer" -> Advanced System Settings -> Advanced -> Environment variables, add a new virable to "System Variables" as
       "OPENCV_DIR" with the value 
         '<YOUR_OPENCV_INSTALLATION_PATH\build\x64\vc12(vc14)>'
    3. If you want your program to run directly, add "%OPENCV_DIR%\bin" to the "User Variables" 's "Path".
### Compilation on MAC OS or Linux
  * Please follow the [CUDA programming guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
