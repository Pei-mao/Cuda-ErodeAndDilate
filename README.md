# Cuda-ErodeAndDilate
A Small Implementation of Image Processing Using CUDA.

| Raw Image | Erosion Image | Dilation Image |
|:---------:|:-------------:|:--------------:|
| ![Raw Image](images/finger.jpg) | ![Erosion Image](images/Erosion.jpg) | ![Dilation Image](images/Dilation.jpg) |

## Process

1. **Read the image**: Read the image from the specified path and convert it to grayscale.
2. **Allocate memory**: Allocate memory on the GPU to store the input and output image data.
3. **Set CUDA kernel parameters**: Define the number of threads per block and calculate the number of blocks needed.
4. **Execute CUDA kernel**: Execute the CUDA kernels for erosion and dilation operations.
5. **Retrieve results**: Copy the computed results from the GPU back to the CPU and save the resulting images to the specified path.
6. **Perform operations using OpenCV**: Perform erosion and dilation operations using OpenCV's `erode` and `dilate` functions, and save the results.
