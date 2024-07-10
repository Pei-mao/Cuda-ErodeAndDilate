#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

__global__ void erodeInCuda(unsigned char* dataIn, unsigned char* dataOut, Size erodeElement, int imgWidth, int imgHeight);
__global__ void dilateInCuda(unsigned char* dataIn, unsigned char* dataOut, Size dilateElement, int imgWidth, int imgHeight);

int main()
{
    Mat srcImg = imread("../images/finger.jpg"); //输入图片
    Mat grayImg = imread("../images/finger.jpg", 0); //输入的灰度图

    if (srcImg.empty() || grayImg.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return -1;
    }

    unsigned char* d_in; //輸入圖片在GPU内的内存
    unsigned char* d_out1; //腐蝕後輸出圖片在GPU内的内存
    unsigned char* d_out2; //膨脹後输出圖片在GPU内的内存

    int imgWidth = grayImg.cols;
    int imgHeight = grayImg.rows;

    Mat dstImg1(imgHeight, imgWidth, CV_8UC1, Scalar(0)); //腐蝕後輸出圖片在CPU内的内存
    Mat dstImg2(imgHeight, imgWidth, CV_8UC1, Scalar(0)); //膨脹後輸出圖片在CPU内的内存

    cudaMalloc((void**)&d_in, imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc((void**)&d_out1, imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc((void**)&d_out2, imgWidth * imgHeight * sizeof(unsigned char));

    cudaMemcpy(d_in, grayImg.data, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Size Element(1, 3);

    erodeInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out1, Element, imgWidth, imgHeight);
    dilateInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out2, Element, imgWidth, imgHeight);

    cudaMemcpy(dstImg1.data, d_out1, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(dstImg2.data, d_out2, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    imwrite("../images/erodeImg.jpg", dstImg1);
    imwrite("../images/dilateImg.jpg", dstImg2);

    //CPU内腐蝕（OpenCV實現）
    Mat erodeImg;
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 3));
    erode(grayImg, erodeImg, element);
    //CUDA膨脹
    dilateInCuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out2, Element, imgWidth, imgHeight);
    //將结果傳回CPU
    cudaMemcpy(dstImg2.data, d_out2, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //CPU内膨脹（OpenCV實現）
    Mat dilateImg;
    dilate(grayImg, dilateImg, element);

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}
