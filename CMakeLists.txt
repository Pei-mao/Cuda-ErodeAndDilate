cmake_minimum_required(VERSION 3.10)
project(CudaErodeDilate)

# 找到 OpenCV 包並包含頭文件目錄
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 設置 CUDA 工具包路徑並找到 CUDA 包
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda) # 根據實際 CUDA 安裝路徑設置
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 包含源文件目錄
include_directories(src)

# 定義可執行文件及其源文件
cuda_add_executable(CudaErodeDilate src/main.cpp src/kernel.cu)

# 鏈接 OpenCV 和 CUDA 庫
target_link_libraries(CudaErodeDilate ${OpenCV_LIBS} ${CUDA_LIBRARIES})
