
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2\opencv.hpp>
#include "OpenCvTest.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__device__ void deviceBlur(const uchar *img_in, uchar *img_out, int img_w, int img_h, int kernel_size, int x, int y)
{
	int k_size_2 = int(kernel_size / 2);

	if (x < k_size_2 || x >= img_w - k_size_2)
		return;
	if (y < k_size_2 || y >= img_h - k_size_2)
		return;

	float sum = 0.0;
	for (int i = -k_size_2; i <= k_size_2; i++)
	{
		for (int j = -k_size_2; j <= k_size_2; j++)
		{
			sum += img_in[(y + i) * img_w + x + j];
		}
	}

	img_out[y * img_w + x] = sum / (kernel_size * kernel_size);
}

__global__ void Blur(const uchar *img_in, uchar *img_out, int img_w, int img_h, int kernel_size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	deviceBlur(img_in, img_out, img_w, img_h, kernel_size, idx, idy);
}

void cudaBlur(const uchar * img_in, uchar * img_out, int img_w, int img_h, int kernel_size)
{

	dim3 th(16, 16);
	dim3 blk(unsigned int(img_w / 16), unsigned int(img_h / 16));
	if (blk.x == 0)
		blk = 1;
	if (blk.y == 0)
		blk.y = 1;
 	Blur <<<blk , th >>>(img_in, img_out, img_w, img_h, kernel_size);
}


int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}
	cv::Mat elf = cv::imread("..\\Elf.jpg");
	//OpenCv::DisplayImage(image);
	//cv::Mat landscape = cv::imread("..\\Landscape.jpg");
	cv::Mat shaft = cv::imread("..\\SHAFT.bmp");
	cv::Mat out(shaft.rows, shaft.cols, CV_8UC1);
	//OpenCv::WhiteNoiseImage(shaft,1000);
	cv::Mat tmp(shaft.rows, shaft.cols, CV_8UC1);
	cv::cvtColor(shaft, tmp, cv::COLOR_BGR2GRAY);
	if (tmp.type() == CV_8UC1)
	{
		OpenCv::DisplayImage(shaft);
		uchar* devImg_tmp = 0;
		uchar* devImg_tmp_out = 0;
		int size = tmp.rows*tmp.cols;
		if (tmp.size().area() == size)
		{
			cudaMalloc((void**)&devImg_tmp, size);
			cudaMalloc((void**)&devImg_tmp_out, size);

			cudaMemcpy((void*)devImg_tmp, (void*)tmp.data, size, cudaMemcpyHostToDevice);

			cudaBlur(devImg_tmp, devImg_tmp_out, tmp.cols, tmp.rows, 32);
			cudaError_t cudaStatus= cudaGetLastError();
			if (cudaStatus == cudaSuccess)
			{
				cudaMemcpy((void*)out.data, (void*)devImg_tmp_out, size, cudaMemcpyDeviceToHost);
			}

			OpenCv::DisplayImage(out);
		}

	}
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
