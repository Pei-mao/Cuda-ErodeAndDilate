#include <opencv2/opencv.hpp>

using namespace cv;

// 腐蝕
__global__ void erodeInCuda(unsigned char* dataIn, unsigned char* dataOut, Size erodeElement, int imgWidth, int imgHeight)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    int elementWidth = erodeElement.width;
    int elementHeight = erodeElement.height;
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;

    dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];

    if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
    {
        for (int i = -halfEH; i <= halfEH; i++)
        {
            for (int j = -halfEW; j <= halfEW; j++)
            {
                if (dataIn[(i + yIndex) * imgWidth + xIndex + j] < dataOut[yIndex * imgWidth + xIndex])
                {
                    dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
                }
            }
        }
    }
}

// 膨脹
__global__ void dilateInCuda(unsigned char* dataIn, unsigned char* dataOut, Size dilateElement, int imgWidth, int imgHeight)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    int elementWidth = dilateElement.width;
    int elementHeight = dilateElement.height;
    int halfEW = elementWidth / 2;
    int halfEH = elementHeight / 2;

    dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];

    if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
    {
        for (int i = -halfEH; i <= halfEH; i++)
        {
            for (int j = -halfEW; j <= halfEW; j++)
            {
                if (dataIn[(i + yIndex) * imgWidth + xIndex + j] > dataOut[yIndex * imgWidth + xIndex])
                {
                    dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
                }
            }
        }
    }
}
