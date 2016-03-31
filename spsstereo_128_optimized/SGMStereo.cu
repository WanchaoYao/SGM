/*
    Copyright (C) 2014  Koichiro Yamaguchi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "SGMStereo.h"
#include <stack>
#include <algorithm>
#include <nmmintrin.h>
#include <time.h>

// Default parameters
#define MAX_DISPARITY 128
const int SGMSTEREO_DEFAULT_DISPARITY_TOTAL = MAX_DISPARITY;
const double SGMSTEREO_DEFAULT_DISPARITY_FACTOR = 256.0;
const int SGMSTEREO_DEFAULT_SOBEL_CAP_VALUE = 15;
const int SGMSTEREO_DEFAULT_CENSUS_WINDOW_RADIUS = 2;
const double SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR = 1.0/6.0;
const int SGMSTEREO_DEFAULT_AGGREGATION_WINDOW_RADIUS = 2;
const int SGMSTEREO_DEFAULT_SMOOTHNESS_PENALTY_SMALL = 100;
const int SGMSTEREO_DEFAULT_SMOOTHNESS_PENALTY_LARGE = 1600;
const int SGMSTEREO_DEFAULT_CONSISTENCY_THRESHOLD = 4;

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK 1024 //(THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y)
__global__ void gpuComputeCappedSobelImageAndCensusImgae(const unsigned char* image,
                                           const bool horizontalFlip,
                                           unsigned char* sobelImage,
                                           const int sobelCapValue_,
                                           int* censusImage,
                                           const int censusWindowRadius_,
                                           const int width_,
                                           const int height_);
__global__ void gpuCalcHalfPixelRightAll(const unsigned char* rightSobelImage, 
                                         unsigned char* gpuHalfPixelRightMinAll_, 
                                         unsigned char* gpuHalfPixelRightMaxAll_, 
                                         const int width_,
                                         const int height_);
__global__ void gpuCalcPixelwiseSADAndHamming(const unsigned char* gpuLeftSobelImage, const unsigned char* gpuRightSobelImage, 
                                              const unsigned char* gpuHalfPixelRightMinAll_, const unsigned char* gpuHalfPixelRightMaxAll_,
                                              const int* gpuLeftCensusImage, const int* gpuRightCensusImage,
                                              unsigned char* gpuPixelwiseCostRowAll_,
                                              double censusWeightFactor_,
                                              int disparityTotal_,
                                              int width_,
                                              int height_,
                                              int borderSize);
__global__ void gpuComputeRightCostImage(const unsigned short* gpuLeftCostImage_,
        unsigned short* gpuRightCostImage_,
        int disparityTotal_,
        int width_,
        int height_);
__global__ void gpuAggregateCost(unsigned short* gpuLeftCostImage_,
                                 unsigned char* gpuPixelwiseCostRowAll_,
                                 int disparityTotal_,
                                 int aggregationWindowRadius_,
                                 int width_,
                                 int height_);
 __global__ void performHorizontalSGM(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startX, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_);
__global__ void performVerticalSGM(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startY, int rowDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_);
__global__ void performDiagonalSGMAlongX(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startY, 
        int rowDiff, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_);
__global__ void performDiagonalSGMAlongY(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startX,
        int rowDiff, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_);
__device__ __inline__ void performPixelSGM(const unsigned short* costImage,
        unsigned short* disparityImage, 
        int x, int y, 
        int colDiff, int rowDiff, 
        int width_, int height_,
        int disparityTotal_, int d, unsigned short* disparityPreviousPixel, unsigned short* minimum,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_);
__global__ void sumDisparity(const unsigned short* disparityImageH, 
        const unsigned short* disparityImageV,
        const unsigned short* disparityImageD1,
        const unsigned short* disparityImageD2,
        unsigned short* disparityImageAll, 
        int width_, int height_, int disparityTotal_);
__global__ void selectDisparity(const unsigned short* disparityImage, 
        unsigned char* resultImage,
        int width_, int height_, int disparityTotal_);

// GPU relevant function added by WanchaoYao
void gpuAllocateDataBuffer(int disparityTotal_, int aggregationWindowRadius_, int width_, int height_, int totalBufferSize_, int pathRowBufferTotal_);
void gpuFreeDataBuffer();

// GPU relavant data added by WanchaoYao
unsigned char* leftGrayscaleImage;
unsigned char* rightGrayscaleImage;
unsigned char* leftResultImage;
unsigned char* rightResultImage;

unsigned char* gpuLeftGrayscaleImage;
unsigned char* gpuRightGrayscaleImage;
unsigned char* gpuLeftSobelImage;
unsigned char* gpuRightSobelImage;
int* gpuLeftCensusImage;
int* gpuRightCensusImage;
unsigned char* gpuHalfPixelRightMinAll_;
unsigned char* gpuHalfPixelRightMaxAll_;
unsigned char* gpuPixelwiseCostRowAll_;
unsigned short* gpuLeftCostImage_;
unsigned short* gpuRightCostImage_;
unsigned short* gpuLeftDisparityImage_;
unsigned short* gpuRightDisparityImage_;
unsigned char* gpuLeftResultImage_;
unsigned char* gpuRightResultImage_;

cudaStream_t stream0, stream1;

SGMStereo::SGMStereo() : disparityTotal_(SGMSTEREO_DEFAULT_DISPARITY_TOTAL),
						 disparityFactor_(SGMSTEREO_DEFAULT_DISPARITY_FACTOR),
						 sobelCapValue_(SGMSTEREO_DEFAULT_SOBEL_CAP_VALUE),
						 censusWindowRadius_(SGMSTEREO_DEFAULT_CENSUS_WINDOW_RADIUS),
						 censusWeightFactor_(SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR),
						 aggregationWindowRadius_(SGMSTEREO_DEFAULT_AGGREGATION_WINDOW_RADIUS),
						 smoothnessPenaltySmall_(SGMSTEREO_DEFAULT_SMOOTHNESS_PENALTY_SMALL),
						 smoothnessPenaltyLarge_(SGMSTEREO_DEFAULT_SMOOTHNESS_PENALTY_LARGE),
						 consistencyThreshold_(SGMSTEREO_DEFAULT_CONSISTENCY_THRESHOLD) {}

void SGMStereo::setDisparityTotal(const int disparityTotal) {
	if (disparityTotal <= 0 || disparityTotal%16 != 0) {
		throw std::invalid_argument("[SGMStereo::setDisparityTotal] the number of disparities must be a multiple of 16");
	}

	disparityTotal_ = disparityTotal;
}

void SGMStereo::setDisparityFactor(const double disparityFactor) {
	if (disparityFactor <= 0) {
		throw std::invalid_argument("[SGMStereo::setOutputDisparityFactor] disparity factor is less than zero");
	}

	disparityFactor_ = disparityFactor;
}

void SGMStereo::setDataCostParameters(const int sobelCapValue,
									  const int censusWindowRadius,
									  const double censusWeightFactor,
									  const int aggregationWindowRadius)
{
	sobelCapValue_ = std::max(sobelCapValue, 15);
	sobelCapValue_ = std::min(sobelCapValue_, 127) | 1;

	if (censusWindowRadius < 1 || censusWindowRadius > 2) {
		throw std::invalid_argument("[SGMStereo::setDataCostParameters] window radius of Census transform must be 1 or 2");
	}
	censusWindowRadius_ = censusWindowRadius;
	if (censusWeightFactor < 0) {
		throw std::invalid_argument("[SGMStereo::setDataCostParameters] weight of Census transform must be positive");
	}
	censusWeightFactor_ = censusWeightFactor;

	aggregationWindowRadius_ = aggregationWindowRadius;
}

void SGMStereo::setSmoothnessCostParameters(const int smoothnessPenaltySmall, const int smoothnessPenaltyLarge)
{
	if (smoothnessPenaltySmall < 0 || smoothnessPenaltyLarge < 0) {
		throw std::invalid_argument("[SGMStereo::setSmoothnessCostParameters] smoothness penalty value is less than zero");
	}
	if (smoothnessPenaltySmall >= smoothnessPenaltyLarge) {
		throw std::invalid_argument("[SGMStereo::setSmoothnessCostParameters] small value of smoothness penalty must be smaller than large penalty value");
	}

	smoothnessPenaltySmall_ = smoothnessPenaltySmall;
	smoothnessPenaltyLarge_ = smoothnessPenaltyLarge;
}

void SGMStereo::setConsistencyThreshold(const int consistencyThreshold) {
	if (consistencyThreshold < 0) {
		throw std::invalid_argument("[SGMStereo::setConsistencyThreshold] threshold for LR consistency must be positive");
	}
	consistencyThreshold_ = consistencyThreshold;
}

void SGMStereo::gpuPerformSGM(const unsigned short* gpuCostImage_, unsigned short* gpuDisparityImage, unsigned char* gpuResultImage, 
        unsigned char* resultImage, cudaStream_t stream)
{
    dim3 dimBlockForD(MAX_DISPARITY, 1);
    dim3 dimGridForD(width_, height_);

    cudaMemset(gpuDisparityImage, 0, width_*height_*disparityTotal_*sizeof(unsigned short));
    
    performHorizontalSGM<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 1,
        width_, height_, disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performHorizontalSGM<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        width_ - 1, -1,
        width_, height_, disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
    
    performVerticalSGM<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 1,
        width_, height_, disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performVerticalSGM<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        height_ - 1, -1,
        width_, height_, disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongX<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 
        1, 1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongX<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 
        1, -1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongX<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        height_-1, 
        -1, -1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongX<<<width_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        height_-1, 
        -1, 1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongY<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 
        1, 1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongY<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        width_-1, 
        1, -1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    performDiagonalSGMAlongY<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        width_-1, 
        -1, -1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
    
    performDiagonalSGMAlongY<<<height_, disparityTotal_, 0, stream>>>(gpuCostImage_, 
        gpuDisparityImage, 
        0, 
        -1, 1,
        width_, height_,
        disparityTotal_,
        smoothnessPenaltySmall_, smoothnessPenaltyLarge_);

    dim3 dimGrid(width_, height_);
    dim3 dimBlock(MAX_DISPARITY, 1);
    selectDisparity<<<dimGrid, dimBlock, 0, stream>>>(gpuDisparityImage, gpuResultImage, width_, height_, disparityTotal_);
    cudaError_t cudaStatus = cudaMemcpyAsync(resultImage, gpuResultImage, width_*height_*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess)
        puts("cudaMemcpy failed!");
}

void SGMStereo::compute(const png::image<png::rgb_pixel>& leftImage,
						const png::image<png::rgb_pixel>& rightImage,
						float* disparityImage)
{
    cudaSetDevice(2);

    clock_t start = clock();
	initialize(leftImage, rightImage);
    clock_t end = clock();
    printf("SGM initialize time: %.2lf\n", double(end - start) / CLOCKS_PER_SEC);

    clock_t begin = clock();
    start = begin;
	computeCostImage(leftImage, rightImage);
    //cudaDeviceSynchronize();
    end = clock();
    printf("SGM computeCostImage time: %.2lf\n", double(end - begin) / CLOCKS_PER_SEC);

    /*
    cudaError_t cudaStatus = cudaMemcpy(leftCostImage_, gpuLeftCostImage_, height_*width_*disparityTotal_*sizeof(unsigned short), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        puts("gpu copy left cost image cudaMemcpy failed!");

    unsigned char* pixelwiseCostRowAll_ = reinterpret_cast<unsigned char*>(malloc((height_+2*aggregationWindowRadius_)*(width_+2*aggregationWindowRadius_)*disparityTotal_*sizeof(unsigned char)));
    cudaStatus = cudaMemcpy(pixelwiseCostRowAll_, gpuPixelwiseCostRowAll_, (height_+2*aggregationWindowRadius_)*(width_+2*aggregationWindowRadius_)*disparityTotal_*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned char* pixelwiseCostRowAllWithoutTopBorder = pixelwiseCostRowAll_ + 
        aggregationWindowRadius_*(width_+2*aggregationWindowRadius_)*disparityTotal_ +aggregationWindowRadius_*disparityTotal_;
    
    for (int y = 0; y < height_; y++)
        if (y % 4 == 0)
        for (int x = 0; x < width_; x++)
            for (int d = 0; d < disparityTotal_; d++)
                printf("(%d, %d, %d): %d\n", y, x, d, 
                        pixelwiseCostRowAllWithoutTopBorder[y*(width_+2*aggregationWindowRadius_)*disparityTotal_+x*disparityTotal_+d]);
    
    for (int y = 2; y < height_ - 2; y++)
        for (int x = 2; x < width_ - 2; x++)
            for (int d = 0; d < disparityTotal_; d++)
            printf("(%d, %d, %d): %d\n", y, x, d, leftCostImage_[y*width_*disparityTotal_+x*disparityTotal_+d]);
    */

    begin = clock();
    gpuPerformSGM(gpuLeftCostImage_, gpuLeftDisparityImage_, gpuLeftResultImage_, leftResultImage, stream0);
    //cudaDeviceSynchronize();
    gpuPerformSGM(gpuRightCostImage_, gpuRightDisparityImage_, gpuRightResultImage_, rightResultImage, stream1);
    cudaDeviceSynchronize();
    end = clock();
    printf("SGM gpuPerformSGM time: %.2lf\n", double(end - begin) / CLOCKS_PER_SEC);

    begin = clock();
    myspeckleFilter(100, 2, leftResultImage);
    myspeckleFilter(100, 2, rightResultImage);
    end = clock();
    printf("SGM speckleFilter time: %.2lf\n", double(end - begin) / CLOCKS_PER_SEC);

    png::image<png::gray_pixel_16> outputImage;
    outputImage.resize(width_, height_);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            unsigned short estimatedDisparity = disparityFactor_*leftResultImage[width_*y + x] + 0.5;
            outputImage.set_pixel(x, y, estimatedDisparity);
        }
    }
    outputImage.write("my_left_result.png");

    /*
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            unsigned short estimatedDisparity = MAX_DISPARITY*rightResultImage[width_*y + x];
            outputImage.set_pixel(x, y, estimatedDisparity);
        }
    }
    outputImage.write("my_right_result.png");
    */

    myenforceLeftRightConsistency(leftResultImage, rightResultImage);
    end = clock();
    printf("In SGM compute time: %.2lf\n", double(end - start) / CLOCKS_PER_SEC);

    //png::image<png::gray_pixel_16> outputImage;
    //outputImage.resize(width_, height_);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            unsigned short estimatedDisparity = disparityFactor_*leftResultImage[width_*y + x] + 0.5;
            outputImage.set_pixel(x, y, estimatedDisparity);
        }
    }
    outputImage.write("my_final_result.png");

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			disparityImage[width_*y + x] = static_cast<float>(leftResultImage[width_*y + x]);
		}
	}

    /*
    unsigned short* leftDisparityImage = reinterpret_cast<unsigned short*>(malloc(width_*height_*sizeof(unsigned short)));
	unsigned short* rightDisparityImage = reinterpret_cast<unsigned short*>(malloc(width_*height_*sizeof(unsigned short)));

    cudaError_t cudaStatus = cudaMemcpy(rightCostImage_, gpuRightCostImage_, width_*height_*disparityTotal_*sizeof(unsigned short), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        puts("cudaMemcpy failed!");
    cudaStatus = cudaMemcpy(leftCostImage_, gpuLeftCostImage_, height_*width_*disparityTotal_*sizeof(unsigned short), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("gpu copy left cost image cudaMemcpy failed!");
    
    begin = clock();
	performSGM(leftCostImage_, leftDisparityImage);
	performSGM(rightCostImage_, rightDisparityImage);
    speckleFilter(100, static_cast<int>(2*disparityFactor_), leftDisparityImage);
	speckleFilter(100, static_cast<int>(2*disparityFactor_), rightDisparityImage);
	enforceLeftRightConsistency(leftDisparityImage, rightDisparityImage);
    end = clock();
    printf("SGM perform SGM time: %.2lf\n", double(end - begin) / CLOCKS_PER_SEC);
	
    png::image<png::gray_pixel_16> outputDisparityImage;
    outputDisparityImage.resize(width_, height_);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            float estimatedDisparity = disparityImage[width_*y + x];
            if (estimatedDisparity <= 0.0 || estimatedDisparity > 255.0) {
                outputDisparityImage.set_pixel(x, y, 0);
            } else {
                outputDisparityImage.set_pixel(x, y, static_cast<unsigned short>(estimatedDisparity*disparityFactor_ + 0.5));
            }
        }
    }
    outputDisparityImage.write("my_SGM_left_disparity.png");

	free(leftDisparityImage);
	free(rightDisparityImage);
    */
    
	freeDataBuffer();
    gpuFreeDataBuffer();
}


void SGMStereo::initialize(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage) {
	setImageSize(leftImage, rightImage);
	allocateDataBuffer();
    gpuAllocateDataBuffer(disparityTotal_, aggregationWindowRadius_, width_, height_, totalBufferSize_, pathRowBufferTotal_);
}

void SGMStereo::setImageSize(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage) {
	width_ = static_cast<int>(leftImage.get_width());
	height_ = static_cast<int>(leftImage.get_height());
	if ((int)rightImage.get_width() != width_ || (int)rightImage.get_height() != height_) {
		throw std::invalid_argument("[SGMStereo::setImageSize] sizes of left and right images are different");
	}
	widthStep_ = width_ + 15 - (width_ - 1)%16;
}

void gpuAllocateDataBuffer(int disparityTotal_, int aggregationWindowRadius_, int width_, int height_, int totalBufferSize_, int pathRowBufferTotal_)
{
    cudaHostAlloc((void**)&leftGrayscaleImage, width_*height_*sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&rightGrayscaleImage, width_*height_*sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&leftResultImage, width_*height_*sizeof(unsigned char), cudaHostAllocDefault);
    cudaHostAlloc((void**)&rightResultImage, width_*height_*sizeof(unsigned char), cudaHostAllocDefault);

    cudaMalloc((void**)&gpuLeftGrayscaleImage, width_*height_*sizeof(unsigned char));
    cudaMalloc((void**)&gpuRightGrayscaleImage, width_*height_*sizeof(unsigned char));
    cudaMalloc((void**)&gpuLeftSobelImage, width_*height_*sizeof(unsigned char));
    cudaMalloc((void**)&gpuRightSobelImage, width_*height_*sizeof(unsigned char));
    cudaMalloc((void**)&gpuLeftCensusImage, width_*height_*sizeof(int));
    cudaMalloc((void**)&gpuRightCensusImage, width_*height_*sizeof(int));

	cudaMalloc((void**)&gpuLeftCostImage_, width_*height_*disparityTotal_*sizeof(unsigned short));
	cudaMalloc((void**)&gpuRightCostImage_, width_*height_*disparityTotal_*sizeof(unsigned short));
	cudaMalloc((void**)&gpuLeftDisparityImage_, width_*height_*disparityTotal_*sizeof(unsigned short));
	cudaMalloc((void**)&gpuRightDisparityImage_, width_*height_*disparityTotal_*sizeof(unsigned short));
	cudaMalloc((void**)&gpuLeftResultImage_, height_*width_*sizeof(unsigned char));
	cudaMalloc((void**)&gpuRightResultImage_, height_*width_*sizeof(unsigned char));
	cudaMalloc((void**)&gpuHalfPixelRightMinAll_, height_*width_*sizeof(unsigned char));
	cudaMalloc((void**)&gpuHalfPixelRightMaxAll_, height_*width_*sizeof(unsigned char));
    cudaMalloc((void**)&gpuPixelwiseCostRowAll_, (height_+2*aggregationWindowRadius_)*(width_+2*aggregationWindowRadius_)*disparityTotal_*sizeof(unsigned char));
        
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
}

void gpuFreeDataBuffer()
{
    cudaFreeHost(leftGrayscaleImage);
    cudaFreeHost(rightGrayscaleImage);
    cudaFreeHost(leftResultImage);
    cudaFreeHost(rightResultImage);

    cudaFree(gpuLeftGrayscaleImage);
    cudaFree(gpuRightGrayscaleImage);
    cudaFree(gpuLeftSobelImage);
    cudaFree(gpuRightSobelImage);
    cudaFree(gpuLeftCensusImage);
    cudaFree(gpuRightCensusImage);

    cudaFree(gpuLeftCostImage_);
	cudaFree(gpuRightCostImage_);
	cudaFree(gpuLeftDisparityImage_);
	cudaFree(gpuRightDisparityImage_);
	cudaFree(gpuLeftResultImage_);
	cudaFree(gpuRightResultImage_);
	cudaFree(gpuHalfPixelRightMinAll_);
	cudaFree(gpuHalfPixelRightMaxAll_);
	cudaFree(gpuPixelwiseCostRowAll_);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}

void SGMStereo::allocateDataBuffer() {
	leftCostImage_ = reinterpret_cast<unsigned short*>(_mm_malloc(width_*height_*disparityTotal_*sizeof(unsigned short), 16));
	rightCostImage_ = reinterpret_cast<unsigned short*>(_mm_malloc(width_*height_*disparityTotal_*sizeof(unsigned short), 16));

	int pixelwiseCostRowBufferSize = width_*disparityTotal_;
	int rowAggregatedCostBufferSize = width_*disparityTotal_*(aggregationWindowRadius_*2+2);
	int halfPixelRightBufferSize = widthStep_;

	pixelwiseCostRow_ = reinterpret_cast<unsigned char*>(_mm_malloc(pixelwiseCostRowBufferSize*sizeof(unsigned char), 16));
	pixelwiseCostRowAll_ = reinterpret_cast<unsigned char*>(_mm_malloc(height_*pixelwiseCostRowBufferSize*sizeof(unsigned char), 16));
	rowAggregatedCost_ = reinterpret_cast<unsigned short*>(_mm_malloc(rowAggregatedCostBufferSize*sizeof(unsigned short), 16));
	halfPixelRightMin_ = reinterpret_cast<unsigned char*>(_mm_malloc(halfPixelRightBufferSize*sizeof(unsigned char), 16));
	halfPixelRightMax_ = reinterpret_cast<unsigned char*>(_mm_malloc(halfPixelRightBufferSize*sizeof(unsigned char), 16));

	pathRowBufferTotal_ = 2;
	disparitySize_ = disparityTotal_ + 16;
	pathTotal_ = 8;
	pathDisparitySize_ = pathTotal_*disparitySize_;

	costSumBufferRowSize_ = width_*disparityTotal_;
	costSumBufferSize_ = costSumBufferRowSize_*height_;
	pathMinCostBufferSize_ = (width_ + 2)*pathTotal_;
	pathCostBufferSize_ = pathMinCostBufferSize_*disparitySize_;
	totalBufferSize_ = (pathMinCostBufferSize_ + pathCostBufferSize_)*pathRowBufferTotal_ + costSumBufferSize_ + 16;

	sgmBuffer_ = reinterpret_cast<short*>(_mm_malloc(totalBufferSize_*sizeof(short), 16));
}

void SGMStereo::freeDataBuffer() {
	_mm_free(leftCostImage_);
	_mm_free(rightCostImage_);
	_mm_free(pixelwiseCostRow_);
	_mm_free(pixelwiseCostRowAll_);
	_mm_free(rowAggregatedCost_);
	_mm_free(halfPixelRightMin_);
	_mm_free(halfPixelRightMax_);
	_mm_free(sgmBuffer_);
}

void SGMStereo::computeCostImage(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage) {
	//unsigned char* leftGrayscaleImage = reinterpret_cast<unsigned char*>(malloc(width_*height_*sizeof(unsigned char)));
	//unsigned char* rightGrayscaleImage = reinterpret_cast<unsigned char*>(malloc(width_*height_*sizeof(unsigned char)));

	convertToGrayscale(leftImage, rightImage, leftGrayscaleImage, rightGrayscaleImage);

	//memset(leftCostImage_, 0, width_*height_*disparityTotal_*sizeof(unsigned short));
    clock_t begin = clock();
	computeLeftCostImage(leftGrayscaleImage, rightGrayscaleImage);
    cudaDeviceSynchronize();
    clock_t end = clock();
    printf("SGM computeLeftCostImage time: %.2lf\n", double(end - begin) / CLOCKS_PER_SEC);

    dim3 dimBlockForD(MAX_DISPARITY, 1);
    dim3 dimGridForD(width_, height_);
    //cudaMemset(gpuRightCostImage_, 0, width_*height_*disparityTotal_*sizeof(unsigned short));
	gpuComputeRightCostImage<<<dimGridForD, dimBlockForD>>>(gpuLeftCostImage_, gpuRightCostImage_, disparityTotal_, width_, height_);

	//free(leftGrayscaleImage);
	//free(rightGrayscaleImage);
}

void SGMStereo::convertToGrayscale(const png::image<png::rgb_pixel>& leftImage,
								   const png::image<png::rgb_pixel>& rightImage,
								   unsigned char* leftGrayscaleImage,
								   unsigned char* rightGrayscaleImage) const
{
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			png::rgb_pixel pix = leftImage.get_pixel(x, y);
			leftGrayscaleImage[width_*y + x] = static_cast<unsigned char>(0.299*pix.red + 0.587*pix.green + 0.114*pix.blue + 0.5);
			pix = rightImage.get_pixel(x, y);
			rightGrayscaleImage[width_*y + x] = static_cast<unsigned char>(0.299*pix.red + 0.587*pix.green + 0.114*pix.blue + 0.5);
		}
	}
}

void SGMStereo::computeLeftCostImage(const unsigned char* leftGrayscaleImage, const unsigned char* rightGrayscaleImage) {
    // GPU code
    {
        /*
        size_t pitch;
        unsigned char* gpuPitch;
        cudaMallocPitch((void**)&gpuPitch, &pitch, width_*sizeof(unsigned char), height_);
        printf("pitch: %d widthStep_: %d\n", pitch, widthStep_);
        */

        cudaError_t cudaStatus = cudaMemcpyAsync(gpuLeftGrayscaleImage, leftGrayscaleImage, width_*height_*sizeof(unsigned char), cudaMemcpyHostToDevice, stream0);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpyAsync(gpuRightGrayscaleImage, rightGrayscaleImage, width_*height_*sizeof(unsigned char), cudaMemcpyHostToDevice, stream1);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");

        /* Old code
        dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
        // Bellow line caused a significant bug that costed me a whole afternoon to find and fix!
        dim3 dimGrid((width_ + dimBlock.x - 1)/dimBlock.x, (height_ + dimBlock.y - 1)/dimBlock.y);
        */

        // New code
        dim3 dimBlock(width_);
        dim3 dimGrid(height_);

        cudaMemset(gpuLeftSobelImage, sobelCapValue_, width_*height_);
        cudaMemset(gpuRightSobelImage, sobelCapValue_, width_*height_);
        
        // Method 1
        gpuComputeCappedSobelImageAndCensusImgae<<<dimGrid, dimBlock, 0, stream1>>>(gpuRightGrayscaleImage, true, gpuRightSobelImage, sobelCapValue_, 
                gpuRightCensusImage, censusWindowRadius_, width_,  height_);
        gpuCalcHalfPixelRightAll<<<dimGrid, dimBlock, 0, stream1>>>(gpuRightSobelImage, gpuHalfPixelRightMinAll_, gpuHalfPixelRightMaxAll_, width_, height_);
        gpuComputeCappedSobelImageAndCensusImgae<<<dimGrid, dimBlock, 0, stream0>>>(gpuLeftGrayscaleImage, false, gpuLeftSobelImage, sobelCapValue_, 
                gpuLeftCensusImage, censusWindowRadius_, width_,  height_);
        cudaDeviceSynchronize();

        /* Method 2
        gpuComputeCappedSobelImageAndCensusImgae<<<dimGrid, dimBlock, 0, 0>>>(gpuLeftGrayscaleImage, false, gpuLeftSobelImage, sobelCapValue_, 
                gpuLeftCensusImage, censusWindowRadius_, width_,  height_);
        gpuComputeCappedSobelImageAndCensusImgae<<<dimGrid, dimBlock, 0, 0>>>(gpuRightGrayscaleImage, true, gpuRightSobelImage, sobelCapValue_, 
                gpuRightCensusImage, censusWindowRadius_, width_,  height_);
        gpuCalcHalfPixelRightAll<<<dimGrid, dimBlock, 0, 0>>>(gpuRightSobelImage, gpuHalfPixelRightMinAll_, gpuHalfPixelRightMaxAll_, width_, height_);
        */

        /*
        unsigned char* leftSobelImage = reinterpret_cast<unsigned char*>(_mm_malloc(widthStep_*height_*sizeof(unsigned char), 16));
        unsigned char* rightSobelImage = reinterpret_cast<unsigned char*>(_mm_malloc(widthStep_*height_*sizeof(unsigned char), 16));
        int* leftCensusImage = reinterpret_cast<int*>(malloc(width_*height_*sizeof(int)));
        int* rightCensusImage = reinterpret_cast<int*>(malloc(width_*height_*sizeof(int)));
        unsigned char* halfPixelRightMinAll_ =  reinterpret_cast<unsigned char*>(malloc(height_*width_*sizeof(unsigned char)));
        unsigned char* halfPixelRightMaxAll_ =  reinterpret_cast<unsigned char*>(malloc(height_*width_*sizeof(unsigned char)));

        cudaStatus = cudaMemcpy(leftSobelImage, gpuLeftSobelImage, widthStep_*height_*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpy(rightSobelImage, gpuRightSobelImage, widthStep_*height_*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpy(leftCensusImage, gpuLeftCensusImage, width_*height_*sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpy(rightCensusImage, gpuRightCensusImage, width_*height_*sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpy(halfPixelRightMinAll_, gpuHalfPixelRightMinAll_, width_*height_*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");
        cudaStatus = cudaMemcpy(halfPixelRightMaxAll_, gpuHalfPixelRightMaxAll_, width_*height_*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
            puts("cudaMemcpy failed!");   

        for (int y = 0; y < height_; y++)
            for (int x = 0; x < width_; x++)
                printf("(%d, %d: %d %d %d %d %d %d\n", y, x, leftSobelImage[y*width_+x], rightSobelImage[y*width_+x], leftCensusImage[y*width_+x], rightCensusImage[y*width_+x], halfPixelRightMinAll_[y*width_+x], halfPixelRightMaxAll_[y*width_+x]); 
        */

        dim3 dimBlockForD(MAX_DISPARITY, 1);
        dim3 dimGridForD(width_, height_);

        cudaMemset(gpuPixelwiseCostRowAll_, 0, (height_+2*aggregationWindowRadius_)*(width_+2*aggregationWindowRadius_)*disparityTotal_*sizeof(unsigned char));
        unsigned char* gpuPixelwiseCostRowAllWithoutTopBorder = gpuPixelwiseCostRowAll_ + 
            aggregationWindowRadius_*(width_+2*aggregationWindowRadius_)*disparityTotal_ +aggregationWindowRadius_*disparityTotal_;

        gpuCalcPixelwiseSADAndHamming<<<dimGridForD, dimBlockForD>>>(gpuLeftSobelImage, gpuRightSobelImage, 
                                      gpuHalfPixelRightMinAll_, gpuHalfPixelRightMaxAll_,
                                      gpuLeftCensusImage, gpuRightCensusImage,
                                      gpuPixelwiseCostRowAllWithoutTopBorder,
                                      censusWeightFactor_,
                                      disparityTotal_,
                                      width_,
                                      height_,
                                      aggregationWindowRadius_);

        cudaMemset(gpuLeftCostImage_, 0, width_*height_*disparityTotal_*sizeof(unsigned short));
        
        gpuAggregateCost<<<dimGridForD, dimBlockForD>>>(gpuLeftCostImage_,
                         gpuPixelwiseCostRowAllWithoutTopBorder,
                         disparityTotal_,
                         aggregationWindowRadius_,
                         width_,
                         height_);
    }
}

__global__ void gpuComputeCappedSobelImageAndCensusImgae(const unsigned char* image,
                                           const bool horizontalFlip,
                                           unsigned char* sobelImage,
                                           const int sobelCapValue_,
                                           int* censusImage,
                                           const int censusWindowRadius_,
                                           const int width_,
                                           const int height_)
{
    // Odd code
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;

    // New code
    int y = blockIdx.x;
    int x = threadIdx.x;

    __shared__ unsigned char subImage[5*640];
    if (y >= 2)
        subImage[x] = image[(y-2)*width_+x];
    if (y >= 1)
        subImage[width_+x] = image[(y-1)*width_+x];
    subImage[2*width_+x] = image[y*width_+x];
    if (y <= height_ - 2)
        subImage[3*width_+x] = image[(y+1)*width_+x];
    if (y <= height_ - 3)
        subImage[4*width_+x] = image[(y+2)*width_+x];
    __syncthreads();

    unsigned char centerValue = subImage[width_*2 + x];
    unsigned char window[5][5];

    int censusCode = 0;
    #pragma unroll 5
    for (int offsetY = -censusWindowRadius_; offsetY <= censusWindowRadius_; ++offsetY) {
        #pragma unroll 5
        for (int offsetX = -censusWindowRadius_; offsetX <= censusWindowRadius_; ++offsetX) {
            censusCode = censusCode << 1;
            if (y + offsetY >= 0 && y + offsetY < height_ && x + offsetX >= 0 && x + offsetX < width_)
            {
                window[2+offsetY][2+offsetX] = subImage[width_*(2 + offsetY) + x + offsetX];
                if (window[2+offsetY][2+offsetX] >= centerValue)
                    censusCode += 1;
            }
        }
    }
    censusImage[width_*y + x] = censusCode;

    if (y > 0 && x > 0 && y < (height_ - 1) && x < (width_ - 1))
    {
        int sobelValue = window[1][3] - window[1][1] + 2*(window[2][3] - window[2][1]) + window[3][3] - window[3][1];
        //int sobelValue = (image[width_*(y - 1) + x + 1] + 2*image[width_*y + x + 1] + image[width_*(y + 1) + x + 1])
        //    - (image[width_*(y - 1) + x - 1] + 2*image[width_*y + x - 1] + image[width_*(y + 1) + x - 1]);
        if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
        else if (sobelValue < -sobelCapValue_) sobelValue = 0;
        else sobelValue += sobelCapValue_;

        if (horizontalFlip)
            sobelImage[width_*y + width_ - x - 1] = sobelValue;
        else
            sobelImage[width_*y + x] = sobelValue;
    }
}

void SGMStereo::computeCappedSobelImage(const unsigned char* image, const bool horizontalFlip, unsigned char* sobelImage) const {
	memset(sobelImage, sobelCapValue_, widthStep_*height_);

	if (horizontalFlip) {
		for (int y = 1; y < height_ - 1; ++y) {
			for (int x = 1; x < width_ - 1; ++x) {
				int sobelValue = (image[width_*(y - 1) + x + 1] + 2*image[width_*y + x + 1] + image[width_*(y + 1) + x + 1])
					- (image[width_*(y - 1) + x - 1] + 2*image[width_*y + x - 1] + image[width_*(y + 1) + x - 1]);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				sobelImage[widthStep_*y + width_ - x - 1] = sobelValue;
			}
		}
	} else {
		for (int y = 1; y < height_ - 1; ++y) {
			for (int x = 1; x < width_ - 1; ++x) {
				int sobelValue = (image[width_*(y - 1) + x + 1] + 2*image[width_*y + x + 1] + image[width_*(y + 1) + x + 1])
					- (image[width_*(y - 1) + x - 1] + 2*image[width_*y + x - 1] + image[width_*(y + 1) + x - 1]);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				sobelImage[widthStep_*y + x] = sobelValue;
			}
		}
	}
}

void SGMStereo::computeCensusImage(const unsigned char* image, int* censusImage) const {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			unsigned char centerValue = image[width_*y + x];

			int censusCode = 0;
			for (int offsetY = -censusWindowRadius_; offsetY <= censusWindowRadius_; ++offsetY) {
				for (int offsetX = -censusWindowRadius_; offsetX <= censusWindowRadius_; ++offsetX) {
					censusCode = censusCode << 1;
					if (y + offsetY >= 0 && y + offsetY < height_
						&& x + offsetX >= 0 && x + offsetX < width_
						&& image[width_*(y + offsetY) + x + offsetX] >= centerValue) censusCode += 1;
				}
			}
			censusImage[width_*y + x] = censusCode;
		}
	}
}

/*
__global__ void gpuAggregateCost(unsigned short* gpuLeftCostImage_,
                                 unsigned char* gpuPixelwiseCostRowAll_,
                                 int disparityTotal_,
                                 int aggregationWindowRadius_,
                                 int width_,
                                 int height_)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.x;

    if (y >= 0 && x >= 0 && y < height_ && x < width_)
    {
        unsigned short* costPixel = gpuLeftCostImage_ + y*width_*disparityTotal_ + x*disparityTotal_;

        if (y >= 1)
        {
            if (x == 0)
            {
                costPixel[d] = 0;
                return;
            }
        }

        int rowStart = y - aggregationWindowRadius_;
        int colStart = x - aggregationWindowRadius_;
        int rowEnd = y + aggregationWindowRadius_;
        int colEnd = x + aggregationWindowRadius_;

        if (rowStart < 0)
        {
            if (colStart < 0)
            {
                costPixel[d] += rowStart*colStart * gpuPixelwiseCostRowAll_[d];
                for (int row = 0; row <= rowEnd; row++)
                    costPixel[d] += -colStart * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + d];
                colStart = 0;
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += -rowStart * gpuPixelwiseCostRowAll_[col*disparityTotal_ + d];
            }
            else if (colEnd >= width_)
            {
                costPixel[d] += -rowStart*(colEnd - width_ + 1) * gpuPixelwiseCostRowAll_[(width_-1)*disparityTotal_ + d];
                for (int row = 0; row <= rowEnd; row++)
                    costPixel[d] += (colEnd - width_ + 1) * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + (width_-1)*disparityTotal_ + d];
                colEnd = width_ - 1;
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += -rowStart * gpuPixelwiseCostRowAll_[col*disparityTotal_ + d];
            }
            else
            {
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += -rowStart * gpuPixelwiseCostRowAll_[col*disparityTotal_ + d];
            }
            rowStart = 0;
        }
        else if (rowEnd >= height_)
        {
            if (colStart < 0)
            {
                costPixel[d] += -colStart*(rowEnd - height_ + 1) * gpuPixelwiseCostRowAll_[(height_-1)*width_*disparityTotal_ + d];
                for (int row = rowStart; row <= height_ - 1; row++)
                    costPixel[d] += -colStart * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + d];
                colStart = 0;
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += (rowEnd - height_ + 1) * gpuPixelwiseCostRowAll_[(height_-1)*width_*disparityTotal_ + col*disparityTotal_ + d];
            }
            else if (colEnd >= width_)
            {
                costPixel[d] += (colEnd - width_ + 1)*(rowEnd - height_ + 1) * gpuPixelwiseCostRowAll_[(height_-1)*width_*disparityTotal_ + (width_-1)*disparityTotal_ + d];
                for (int row = rowStart; row <= height_ - 1; row++)
                    costPixel[d] += (colEnd - width_ + 1) * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + (width_-1)*disparityTotal_ + d];
                colEnd = width_ - 1;
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += (rowEnd - height_ + 1) * gpuPixelwiseCostRowAll_[(height_-1)*width_*disparityTotal_ + col*disparityTotal_ + d];
            }
            else
            {
                for (int col = colStart; col <= colEnd; col++)
                    costPixel[d] += (rowEnd - height_ + 1) * gpuPixelwiseCostRowAll_[(height_-1)*width_*disparityTotal_ + col*disparityTotal_ + d];
            }
            rowEnd = height_ - 1;
        }
        else if (colStart < 0)
        {
            for (int row = rowStart; row <= rowEnd; row++)
                costPixel[d] += -colStart * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + d];
            colStart = 0;
        }
        else if (colEnd >= width_)
        {
            for (int row = rowStart; row <= rowEnd; row++)
                costPixel[d] += (colEnd - width_ + 1) * gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + (width_-1)*disparityTotal_ + d];
            colEnd = width_ - 1;
        }

        for (int row = rowStart; row <= rowEnd; row++)
            for (int col = colStart; col <= colEnd; col++)
            {
                costPixel[d] += gpuPixelwiseCostRowAll_[row*width_*disparityTotal_ + col*disparityTotal_ + d];
            }
    }
}
*/


__global__ void gpuAggregateCost(unsigned short* gpuLeftCostImage_,
                                 unsigned char* gpuPixelwiseCostRowAll_,
                                 int disparityTotal_,
                                 int aggregationWindowRadius_,
                                 int width_,
                                 int height_)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.x;

    unsigned short* costPixel = gpuLeftCostImage_ + y*width_*disparityTotal_ + x*disparityTotal_;
    int widthWithBorders = width_+2*aggregationWindowRadius_;

    int rowStart = y - aggregationWindowRadius_;
    int colStart = x - aggregationWindowRadius_;
    int rowEnd = y + aggregationWindowRadius_;
    int colEnd = x + aggregationWindowRadius_;

    #pragma unroll 5
    for (int row = rowStart; row <= rowEnd; row++)
        #pragma unroll 5
        for (int col = colStart; col <= colEnd; col++)
        {
            costPixel[d] += gpuPixelwiseCostRowAll_[row*widthWithBorders*disparityTotal_ + col*disparityTotal_ + d];
        }
}

void SGMStereo::mycalcTopRowCost(unsigned char*& leftSobelRow, int*& leftCensusRow,
							   unsigned char*& rightSobelRow, int*& rightCensusRow,
                               unsigned char*& pixelwiseCostRow,
							   unsigned short* costImageRow)
{
	for (int rowIndex = 0; rowIndex <= aggregationWindowRadius_; ++rowIndex) {
		int rowAggregatedCostIndex = std::min(rowIndex, height_ - 1)%(aggregationWindowRadius_*2+2);
		unsigned short* rowAggregatedCostCurrent = rowAggregatedCost_ + rowAggregatedCostIndex*width_*disparityTotal_;

		memset(rowAggregatedCostCurrent, 0, disparityTotal_*sizeof(unsigned short));
		// x = 0
		for (int x = 0; x <= aggregationWindowRadius_; ++x) {
			int scale = x == 0 ? aggregationWindowRadius_ + 1 : 1;
			for (int d = 0; d < disparityTotal_; ++d) {
				rowAggregatedCostCurrent[d] += static_cast<unsigned short>(pixelwiseCostRow[disparityTotal_*x + d]*scale);
			}
		}
		// x = 1...width-1
		for (int x = 1; x < width_; ++x) {
			const unsigned char* addPixelwiseCost = pixelwiseCostRow
				+ std::min((x + aggregationWindowRadius_)*disparityTotal_, (width_ - 1)*disparityTotal_);
			const unsigned char* subPixelwiseCost = pixelwiseCostRow
				+ std::max((x - aggregationWindowRadius_ - 1)*disparityTotal_, 0);

			for (int d = 0; d < disparityTotal_; ++d) {
				rowAggregatedCostCurrent[disparityTotal_*x + d]
					= static_cast<unsigned short>(rowAggregatedCostCurrent[disparityTotal_*(x - 1) + d]
					+ addPixelwiseCost[d] - subPixelwiseCost[d]);
			}
		}

		// Add to cost
		int scale = rowIndex == 0 ? aggregationWindowRadius_ + 1 : 1;
		for (int i = 0; i < width_*disparityTotal_; ++i) {
			costImageRow[i] += rowAggregatedCostCurrent[i]*scale;
		}

		leftSobelRow += widthStep_;
		rightSobelRow += widthStep_;
		leftCensusRow += width_;
		rightCensusRow += width_;
	    pixelwiseCostRow += width_*disparityTotal_;
	}
}

void SGMStereo::mycalcRowCosts(unsigned char*& leftSobelRow, int*& leftCensusRow,
							   unsigned char*& rightSobelRow, int*& rightCensusRow,
                               unsigned char*& pixelwiseCostRow,
							   unsigned short* costImageRow)
{
	const int widthStepCost = width_*disparityTotal_;
	const __m128i registerZero = _mm_setzero_si128();

	for (int y = 1; y < height_; ++y) {
		int addRowIndex = y + aggregationWindowRadius_;
		int addRowAggregatedCostIndex = std::min(addRowIndex, height_ - 1)%(aggregationWindowRadius_*2 + 2);
		unsigned short* addRowAggregatedCost = rowAggregatedCost_ + width_*disparityTotal_*addRowAggregatedCostIndex;

		if (addRowIndex < height_) {
			memset(addRowAggregatedCost, 0, disparityTotal_*sizeof(unsigned short));
			// x = 0
			for (int x = 0; x <= aggregationWindowRadius_; ++x) {
				int scale = x == 0 ? aggregationWindowRadius_ + 1 : 1;
				for (int d = 0; d < disparityTotal_; ++d) {
					addRowAggregatedCost[d] += static_cast<unsigned short>(pixelwiseCostRow[disparityTotal_*x + d]*scale);
				}
			}
			// x = 1...width-1
			int subRowAggregatedCostIndex = std::max(y - aggregationWindowRadius_ - 1, 0)%(aggregationWindowRadius_*2 + 2);
			const unsigned short* subRowAggregatedCost = rowAggregatedCost_ + width_*disparityTotal_*subRowAggregatedCostIndex;
			const unsigned short* previousCostRow = costImageRow - widthStepCost;
			for (int x = 1; x < width_; ++x) {
				const unsigned char* addPixelwiseCost = pixelwiseCostRow
					+ std::min((x + aggregationWindowRadius_)*disparityTotal_, (width_ - 1)*disparityTotal_);
				const unsigned char* subPixelwiseCost = pixelwiseCostRow
					+ std::max((x - aggregationWindowRadius_ - 1)*disparityTotal_, 0);

				for (int d = 0; d < disparityTotal_; d += 16) {
					__m128i registerAddPixelwiseLow = _mm_load_si128(reinterpret_cast<const __m128i*>(addPixelwiseCost + d));
					__m128i registerAddPixelwiseHigh = _mm_unpackhi_epi8(registerAddPixelwiseLow, registerZero);
					registerAddPixelwiseLow = _mm_unpacklo_epi8(registerAddPixelwiseLow, registerZero);
					__m128i registerSubPixelwiseLow = _mm_load_si128(reinterpret_cast<const __m128i*>(subPixelwiseCost + d));
					__m128i registerSubPixelwiseHigh = _mm_unpackhi_epi8(registerSubPixelwiseLow, registerZero);
					registerSubPixelwiseLow = _mm_unpacklo_epi8(registerSubPixelwiseLow, registerZero);

					// Low
					__m128i registerAddAggregated = _mm_load_si128(reinterpret_cast<const __m128i*>(addRowAggregatedCost
						+ disparityTotal_*(x - 1) + d));
					registerAddAggregated = _mm_adds_epi16(_mm_subs_epi16(registerAddAggregated, registerSubPixelwiseLow),
														   registerAddPixelwiseLow);
					__m128i registerCost = _mm_load_si128(reinterpret_cast<const __m128i*>(previousCostRow + disparityTotal_*x + d));
					registerCost = _mm_adds_epi16(_mm_subs_epi16(registerCost,
						_mm_load_si128(reinterpret_cast<const __m128i*>(subRowAggregatedCost + disparityTotal_*x + d))),
						registerAddAggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(addRowAggregatedCost + disparityTotal_*x + d), registerAddAggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(costImageRow + disparityTotal_*x + d), registerCost);

					// High
					registerAddAggregated = _mm_load_si128(reinterpret_cast<const __m128i*>(addRowAggregatedCost + disparityTotal_*(x-1) + d + 8));
					registerAddAggregated = _mm_adds_epi16(_mm_subs_epi16(registerAddAggregated, registerSubPixelwiseHigh),
														   registerAddPixelwiseHigh);
					registerCost = _mm_load_si128(reinterpret_cast<const __m128i*>(previousCostRow + disparityTotal_*x + d + 8));
					registerCost = _mm_adds_epi16(_mm_subs_epi16(registerCost,
						_mm_load_si128(reinterpret_cast<const __m128i*>(subRowAggregatedCost + disparityTotal_*x + d + 8))),
						registerAddAggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(addRowAggregatedCost + disparityTotal_*x + d + 8), registerAddAggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(costImageRow + disparityTotal_*x + d + 8), registerCost);
				}
			}
		}

		leftSobelRow += widthStep_;
		rightSobelRow += widthStep_;
		leftCensusRow += width_;
		rightCensusRow += width_;
        pixelwiseCostRow += widthStepCost;
		costImageRow += widthStepCost;
	}
}

__global__ void gpuCalcPixelwiseSADAndHamming(const unsigned char* gpuLeftSobelImage, const unsigned char* gpuRightSobelImage, 
                                              const unsigned char* gpuHalfPixelRightMinAll_, const unsigned char* gpuHalfPixelRightMaxAll_,
                                              const int* gpuLeftCensusImage, const int* gpuRightCensusImage,
                                              unsigned char* gpuPixelwiseCostRowAll_,
                                              double censusWeightFactor_,
                                              int disparityTotal_,
                                              int width_,
                                              int height_,
                                              int borderSize)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.x;

    int widthWithBorders = width_+2*borderSize;
    const unsigned char* gpuLeftSobelRow = gpuLeftSobelImage + y*width_;
    const unsigned char* gpuRightSobelRow = gpuRightSobelImage + y*width_;
    const unsigned char* gpuHalfPixelRightMin_ = gpuHalfPixelRightMinAll_ + y*width_;
    const unsigned char* gpuHalfPixelRightMax_ = gpuHalfPixelRightMaxAll_ + y*width_;
    
    unsigned char* gpuPixelwiseCostRow_ = gpuPixelwiseCostRowAll_ + y*widthWithBorders*disparityTotal_;

    int leftCenterValue = gpuLeftSobelRow[x];
    int leftHalfLeftValue = x > 0 ? (leftCenterValue + gpuLeftSobelRow[x - 1])/2 : leftCenterValue;
    int leftHalfRightValue = x < width_ - 1 ? (leftCenterValue + gpuLeftSobelRow[x + 1])/2 : leftCenterValue;
    int leftMinValue = min(leftHalfLeftValue, leftHalfRightValue);
    leftMinValue = min(leftMinValue, leftCenterValue);
    int leftMaxValue = max(leftHalfLeftValue, leftHalfRightValue);
    leftMaxValue = max(leftMaxValue, leftCenterValue);

    const int* gpuLeftCensusRow = gpuLeftCensusImage + y*width_;
    const int* gpuRightCensusRow = gpuRightCensusImage + y*width_;

    int leftCencusCode = gpuLeftCensusRow[x];
    int hammingDistance = 0;

    bool flag = (x < disparityTotal_ && d <= x) || x >= disparityTotal_;
    if (flag) {
        int rightCenterValue = gpuRightSobelRow[width_ - 1 - x + d];
        int rightMinValue = gpuHalfPixelRightMin_[width_ - 1 - x + d];
        int rightMaxValue = gpuHalfPixelRightMax_[width_ - 1 - x + d];

        int costLtoR = max(0, leftCenterValue - rightMaxValue);
        costLtoR = max(costLtoR, rightMinValue - leftCenterValue);
        int costRtoL = max(0, rightCenterValue - leftMaxValue);
        costRtoL = max(costRtoL, leftMinValue - rightCenterValue);
        int costValue = min(costLtoR, costRtoL);

        int rightCensusCode = gpuRightCensusRow[x - d];
        int n = static_cast<unsigned int>(leftCencusCode^rightCensusCode);
        #pragma enroll 25
        for (int i  = 0; i < 25; i++)
        {
            hammingDistance += n & 1;
            n >>= 1;
        }

        gpuPixelwiseCostRow_[disparityTotal_*x + d] = costValue + static_cast<unsigned char>(hammingDistance*censusWeightFactor_);
    }
    __syncthreads();
    if (!flag) {
        // Method 1
        gpuPixelwiseCostRow_[disparityTotal_*x + d] = gpuPixelwiseCostRow_[disparityTotal_*x + x];
    }
}

__global__ void gpuCalcHalfPixelRightAll(const unsigned char* rightSobelImage, 
                                         unsigned char* gpuHalfPixelRightMinAll_, 
                                         unsigned char* gpuHalfPixelRightMaxAll_, 
                                         const int width_,
                                         const int height_)
{    
    // Odd code
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;

    // New code
    int y = blockIdx.x;
    int x = threadIdx.x;

    const unsigned char* rightSobelRow = rightSobelImage + y * width_;
    int centerValue = rightSobelRow[x];
    int leftHalfValue = x > 0 ? (centerValue + rightSobelRow[x - 1])/2 : centerValue;
    int rightHalfValue = x < width_ - 1 ? (centerValue + rightSobelRow[x + 1])/2 : centerValue;
    int minValue = min(leftHalfValue, rightHalfValue);
    minValue = min(minValue, centerValue);
    int maxValue = max(leftHalfValue, rightHalfValue);
    maxValue = max(maxValue, centerValue);
    
    unsigned char* gpuHalfPixelRightMin_ = gpuHalfPixelRightMinAll_ + y * width_;
    unsigned char* gpuHalfPixelRightMax_ = gpuHalfPixelRightMaxAll_ + y * width_;
    gpuHalfPixelRightMin_[x] = minValue;
    gpuHalfPixelRightMax_[x] = maxValue;
}

__global__ void gpuComputeRightCostImage(const unsigned short* gpuLeftCostImage_,
        unsigned short* gpuRightCostImage_,
        int disparityTotal_,
        int width_,
        int height_) {
    int d = threadIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;

    const int widthStepCost = width_*disparityTotal_;

    const unsigned short* leftCostRow = gpuLeftCostImage_ + widthStepCost*y;
    unsigned short* rightCostRow = gpuRightCostImage_ + widthStepCost*y;
    
    if ((x < disparityTotal_ && d <= x) || x >= disparityTotal_)
    {
        const unsigned short* leftCostPointer = leftCostRow + disparityTotal_*x + d;
        unsigned short* rightCostPointer = rightCostRow + disparityTotal_*(x-d) + d;
        *(rightCostPointer) = *(leftCostPointer);
    }

    if (x > width_ - disparityTotal_ && d >= width_ - x)
    {
        int maxDisparityIndex = width_ - x;
        //unsigned short lastValue = *(rightCostRow + disparityTotal_*x + maxDisparityIndex - 1);
        // Below line is buggy code
        //unsigned short lastValue = *(leftCostRow + disparityTotal_*(x + maxDisparityIndex - 1) + maxDisparityIndex - 1);
        unsigned short lastValue = *(leftCostRow + disparityTotal_*x + maxDisparityIndex - 1);
        unsigned short* rightCostPointer = rightCostRow + disparityTotal_*x + d;
        *(rightCostPointer) = lastValue;
    }
}

void SGMStereo::computeRightCostImage() {
	const int widthStepCost = width_*disparityTotal_;

	for (int y = 0; y < height_; ++y) {
		unsigned short* leftCostRow = leftCostImage_ + widthStepCost*y;
		unsigned short* rightCostRow = rightCostImage_ + widthStepCost*y;

		for (int x = 0; x < disparityTotal_; ++x) {
			unsigned short* leftCostPointer = leftCostRow + disparityTotal_*x;
			unsigned short* rightCostPointer = rightCostRow + disparityTotal_*x;
			for (int d = 0; d <= x; ++d) {
				*(rightCostPointer) = *(leftCostPointer);
				rightCostPointer -= disparityTotal_ - 1;
				++leftCostPointer;
			}
		}

		for (int x = disparityTotal_; x < width_; ++x) {
			unsigned short* leftCostPointer = leftCostRow + disparityTotal_*x;
			unsigned short* rightCostPointer = rightCostRow + disparityTotal_*x;
			for (int d = 0; d < disparityTotal_; ++d) {
				*(rightCostPointer) = *(leftCostPointer);
				rightCostPointer -= disparityTotal_ - 1;
				++leftCostPointer;
			}
		}

		for (int x = width_ - disparityTotal_ + 1; x < width_; ++x) {
			int maxDisparityIndex = width_ - x;
			unsigned short lastValue = *(rightCostRow + disparityTotal_*x + maxDisparityIndex - 1);

			unsigned short* rightCostPointer = rightCostRow + disparityTotal_*x + maxDisparityIndex;
			for (int d = maxDisparityIndex; d < disparityTotal_; ++d) {
				*(rightCostPointer) = lastValue;
				++rightCostPointer;
			}
		}
	}
}

__global__ void sumDisparity(const unsigned short* disparityImageH, 
        const unsigned short* disparityImageV,
        const unsigned short* disparityImageD1,
        const unsigned short* disparityImageD2,
        unsigned short* disparityImageAll, 
        int width_, int height_, int disparityTotal_)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.x;
    
    unsigned short* disparityPixelAll = disparityImageAll + y*width_*disparityTotal_ + x*disparityTotal_;

    disparityPixelAll[d] += *(disparityImageH + y*width_*disparityTotal_ + x*disparityTotal_ + d) + 
        *(disparityImageV + y*width_*disparityTotal_ + x*disparityTotal_ + d) + 
        *(disparityImageD1 + y*width_*disparityTotal_ + x*disparityTotal_ + d) + 
        *(disparityImageD2 + y*width_*disparityTotal_ + x*disparityTotal_ + d);
}

__global__ void selectDisparity(const unsigned short* disparityImage, 
        unsigned char* resultImage, 
        int width_, int height_, int disparityTotal_)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.x;

    //printf("x: %d y: %d d:%d\n", x, y, d);
    //if (y == 0 && x == 0)
    __shared__ unsigned short minimum[MAX_DISPARITY], index[MAX_DISPARITY];
    const unsigned short* disparityPixel = disparityImage + y*width_*disparityTotal_ + x*disparityTotal_;
    minimum[d] = disparityPixel[d];
    index[d] = d;
    //printf("[%d]:%d(%d) ", d, minimum[d], index[d]);
    __syncthreads();

    int dis = 2;
    // 7 - max disparity 128
    // 8 - max disparity 256
    #pragma unroll 7
    for (int i = 0; i < 7; i++)
    {
        //if (d == 0)
        //    printf("\ndis: %d **********\n", dis);
        //__syncthreads();
        if ((d & (dis-1)) == 0 && minimum[d] > minimum[d+dis/2])
        {
            minimum[d] = minimum[d+dis/2];
            index[d] = index[d+dis/2];
        }
        //printf("[%d]:%d(%d) ", d, minimum[d], index[d]);
        __syncthreads();
        dis *= 2;
    }
    
    /*
    unsigned short minDisparityCost = disparityPixel[0];
    unsigned char bestDisparity = 0;
    for (int d = 1; d < disparityTotal_; d++)
    {
        if (disparityPixel[d] < minDisparityCost)
        {
            minDisparityCost = disparityPixel[d];
            bestDisparity = d;
        }
    }
    */
        
    resultImage[y*width_ + x] = index[0];
    //if (d == 0)
    //    printf("resultImage[%d][%d]: %d\n", y, x, resultImage[y*width_ + x]);
}

__global__ void performVerticalSGM(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startY, int rowDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
    int startX = blockIdx.x;
    int d = threadIdx.x; 

    //printf("startY: %d\n", startY);
    __shared__ unsigned short disparityPreviousPixel[MAX_DISPARITY], minimum[MAX_DISPARITY];
    disparityPreviousPixel[d] = *(costImage + startY * width_ * disparityTotal_ + startX * disparityTotal_ + d);
    __syncthreads();
    
    unsigned short* disparityPixel = disparityImage + startY * width_ * disparityTotal_ + startX * disparityTotal_;
    disparityPixel[d] += disparityPreviousPixel[d];

    startY += rowDiff;

    #pragma unroll 480
    for (int y = startY; y < height_ && y > -1; y+=rowDiff)
        performPixelSGM(costImage, disparityImage, startX, y, 0, rowDiff, width_, height_, disparityTotal_, d, disparityPreviousPixel, minimum, smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
}

__global__ void performHorizontalSGM(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startX, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
    int startY = blockIdx.x;
    int d = threadIdx.x; 

    //printf("startY: %d\n", startY);
    __shared__ unsigned short disparityPreviousPixel[MAX_DISPARITY], minimum[MAX_DISPARITY];
    disparityPreviousPixel[d] = *(costImage + startY * width_ * disparityTotal_ + startX * disparityTotal_ + d);
    __syncthreads();
    
    unsigned short* disparityPixel = disparityImage + startY * width_ * disparityTotal_ + startX * disparityTotal_;
    disparityPixel[d] += disparityPreviousPixel[d];

    startX += colDiff;

    #pragma unroll 640
    for (int x = startX; x < width_ && x > -1; x+=colDiff)
        performPixelSGM(costImage, disparityImage, x, startY, colDiff, 0, width_, height_, disparityTotal_, d, disparityPreviousPixel, minimum, smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
}

__global__ void performDiagonalSGMAlongX(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startY, 
        int rowDiff, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
    int startX = blockIdx.x;
    int d = threadIdx.x; 

    //printf("startY: %d\n", startY);
    __shared__ unsigned short disparityPreviousPixel[MAX_DISPARITY], minimum[MAX_DISPARITY];
    disparityPreviousPixel[d] = *(costImage + startY * width_ * disparityTotal_ + startX * disparityTotal_ + d);
    __syncthreads();
    
    unsigned short* disparityPixel = disparityImage + startY * width_ * disparityTotal_ + startX * disparityTotal_;
    disparityPixel[d] += disparityPreviousPixel[d];

    startY += rowDiff;
    startX += colDiff;

    for (int y = startY, x = startX; y < height_ && y > -1 && x < width_ && x > -1; y+=rowDiff, x+=colDiff)
        performPixelSGM(costImage, disparityImage, x, y, colDiff, rowDiff, width_, height_, disparityTotal_, d, disparityPreviousPixel, minimum, smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
}

__global__ void performDiagonalSGMAlongY(const unsigned short* costImage, 
        unsigned short* disparityImage, 
        int startX,
        int rowDiff, int colDiff,
        int width_, int height_,
        int disparityTotal_,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
    int startY = blockIdx.x;
    int d = threadIdx.x; 

    //printf("startY: %d\n", startY);
    __shared__ unsigned short disparityPreviousPixel[MAX_DISPARITY], minimum[MAX_DISPARITY];
    disparityPreviousPixel[d] = *(costImage + startY * width_ * disparityTotal_ + startX * disparityTotal_ + d);
    __syncthreads();
    
    unsigned short* disparityPixel = disparityImage + startY * width_ * disparityTotal_ + startX * disparityTotal_;
    disparityPixel[d] += disparityPreviousPixel[d];

    startY += rowDiff;
    startX += colDiff;

    for (int y = startY, x = startX; y < height_ && y > -1 && x < width_ && x > -1; y+=rowDiff, x+=colDiff)
        performPixelSGM(costImage, disparityImage, x, y, colDiff, rowDiff, width_, height_, disparityTotal_, d, disparityPreviousPixel, minimum, smoothnessPenaltySmall_, smoothnessPenaltyLarge_);
}

__device__ __inline__ void performPixelSGM(const unsigned short* costImage,
        unsigned short* disparityImage, 
        int x, int y, 
        int colDiff, int rowDiff, 
        int width_, int height_,
        int disparityTotal_, int d, unsigned short* disparityPreviousPixel, unsigned short* minimum,
        int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
    //printf("In performPixelSGM x: %d y: %d\n", x, y);

    const unsigned short* costPixel = costImage + y * width_ * disparityTotal_ + x * disparityTotal_;
    unsigned short* disparityPixel = disparityImage + y * width_ * disparityTotal_ + x * disparityTotal_;
    
    unsigned short secondTerm = disparityPreviousPixel[d];

    if (d != 0)
        secondTerm = min(secondTerm, (unsigned short)(disparityPreviousPixel[d-1] + smoothnessPenaltySmall_));
    if (d != disparityTotal_ - 1)
        secondTerm = min(secondTerm, (unsigned short)(disparityPreviousPixel[d+1] + smoothnessPenaltySmall_));
    
    int dis = 2;

    minimum[d] = disparityPreviousPixel[d];
    __syncthreads();

    // 7 - max disparity 128
    // 8 - max disparity 256
    #pragma unroll 7
    for (int i = 0; i < 7; i++)
    {
        if ((d & (dis-1)) == 0)
            minimum[d] = min(minimum[d], minimum[d+dis/2]);
        __syncthreads();
        dis *= 2;
    }
    unsigned short thirdTerm = minimum[0];
    
    secondTerm = min(secondTerm, (unsigned short)(thirdTerm + smoothnessPenaltyLarge_));

    disparityPreviousPixel[d] = costPixel[d] + secondTerm - thirdTerm;
    __syncthreads();

    disparityPixel[d] += disparityPreviousPixel[d];
}

void SGMStereo::performSGM(unsigned short* costImage, unsigned short* disparityImage) {
	const short costMax = SHRT_MAX;

	int widthStepCostImage = width_*disparityTotal_;

	short* costSums = sgmBuffer_;
	memset(costSums, 0, costSumBufferSize_*sizeof(short));

	short** pathCosts = new short*[pathRowBufferTotal_];
	short** pathMinCosts = new short*[pathRowBufferTotal_];

	const int processPassTotal = 2;
	for (int processPassCount = 0; processPassCount < processPassTotal; ++processPassCount) {
		int startX, endX, stepX;
		int startY, endY, stepY;
		if (processPassCount == 0) {
			startX = 0; endX = width_; stepX = 1;
			startY = 0; endY = height_; stepY = 1;
		} else {
			startX = width_ - 1; endX = -1; stepX = -1;
			startY = height_ - 1; endY = -1; stepY = -1;
		}

		for (int i = 0; i < pathRowBufferTotal_; ++i) {
			pathCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*i + pathDisparitySize_ + 8;
			memset(pathCosts[i] - pathDisparitySize_ - 8, 0, pathCostBufferSize_*sizeof(short));
			pathMinCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
				+ pathMinCostBufferSize_*i + pathTotal_*2;
			memset(pathMinCosts[i] - pathTotal_, 0, pathMinCostBufferSize_*sizeof(short));
		}

		for (int y = startY; y != endY; y += stepY) {
			unsigned short* pixelCostRow = costImage + widthStepCostImage*y;
			short* costSumRow = costSums + costSumBufferRowSize_*y;

			memset(pathCosts[0] - pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathCosts[0] + width_*pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_*sizeof(short));
			memset(pathMinCosts[0] + width_*pathTotal_, 0, pathTotal_*sizeof(short));

			for (int x = startX; x != endX; x += stepX) {
				int pathMinX = x*pathTotal_;
				int pathX = pathMinX*disparitySize_;

				int previousPathMin0 = pathMinCosts[0][pathMinX - stepX*pathTotal_] + smoothnessPenaltyLarge_;
				int previousPathMin2 = pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;

				short* previousPathCosts0 = pathCosts[0] + pathX - stepX*pathDisparitySize_;
				short* previousPathCosts2 = pathCosts[1] + pathX + disparitySize_*2;

				previousPathCosts0[-1] = previousPathCosts0[disparityTotal_] = costMax;
				previousPathCosts2[-1] = previousPathCosts2[disparityTotal_] = costMax;

				short* pathCostCurrent = pathCosts[0] + pathX;
				const unsigned short* pixelCostCurrent = pixelCostRow + disparityTotal_*x;
				short* costSumCurrent = costSumRow + disparityTotal_*x;

				__m128i regPenaltySmall = _mm_set1_epi16(static_cast<short>(smoothnessPenaltySmall_));

				__m128i regPathMin0, regPathMin2;
				regPathMin0 = _mm_set1_epi16(static_cast<short>(previousPathMin0));
				regPathMin2 = _mm_set1_epi16(static_cast<short>(previousPathMin2));
				__m128i regNewPathMin = _mm_set1_epi16(costMax);

				for (int d = 0; d < disparityTotal_; d += 8) {
					__m128i regPixelCost = _mm_load_si128(reinterpret_cast<const __m128i*>(pixelCostCurrent + d));

					__m128i regPathCost0, regPathCost2;
					regPathCost0 = _mm_load_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d));
					regPathCost2 = _mm_load_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d));

					regPathCost0 = _mm_min_epi16(regPathCost0,
												 _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d - 1)),
												 regPenaltySmall));
					regPathCost0 = _mm_min_epi16(regPathCost0,
												 _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d + 1)),
												 regPenaltySmall));
					regPathCost2 = _mm_min_epi16(regPathCost2,
												 _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d - 1)),
												 regPenaltySmall));
					regPathCost2 = _mm_min_epi16(regPathCost2,
												 _mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d + 1)),
												 regPenaltySmall));

                    //puts("* performSGM *");
                    /*
                    short result[8];
					_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost0);
                    //for (int i = 0; i < 8; i++)
                    //    printf("after 2 min pathCost0[%d]: %d\n", i, result[i]);
					_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost2);
                    //for (int i = 0; i < 8; i++)
                    //    printf("after 4 min pathCost2[%d]: %d\n", i, result[i]);

					_mm_store_si128(reinterpret_cast<__m128i*>(result), regPixelCost);
                    //for (int i = 0; i < 8; i++)
                    //    printf("regPixelCost[%d]: %d\n", i, result[i]);
                    */
					regPathCost0 = _mm_min_epi16(regPathCost0, regPathMin0);
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost0);
                    //for (int i = 0; i < 8; i++)
                    //    printf("after 1 min pathCost0[%d]: %d\n", i, result[i]);
					regPathCost0 = _mm_adds_epi16(_mm_subs_epi16(regPathCost0, regPathMin0), regPixelCost);
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost0);
                    //for (int i = 0; i < 8; i++)
                    //    printf("after 1 add pathCost0[%d]: %d\n", i, result[i]);
					regPathCost2 = _mm_min_epi16(regPathCost2, regPathMin2);
					regPathCost2 = _mm_adds_epi16(_mm_subs_epi16(regPathCost2, regPathMin2), regPixelCost);

					_mm_store_si128(reinterpret_cast<__m128i*>(pathCostCurrent + d), regPathCost0);
					_mm_store_si128(reinterpret_cast<__m128i*>(pathCostCurrent + d + disparitySize_*2), regPathCost2);

                    //_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost2);
                    //for (int i = 0; i < 8; i++)
                        //printf("before regMin02 regPathCost2[%d]: %d\n", i, result[i]);
                    __m128i regMin02 = _mm_min_epi16(_mm_unpacklo_epi16(regPathCost0, regPathCost2),
													 _mm_unpackhi_epi16(regPathCost0, regPathCost2));
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regMin02);
                    //for (int i = 0; i < 8; i++)
                        //printf("after 1 min regMin02[%d]: %d\n", i, result[i]);

					regMin02 = _mm_min_epi16(_mm_unpacklo_epi16(regMin02, regMin02),
											 _mm_unpackhi_epi16(regMin02, regMin02));
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regMin02);
                    //for (int i = 0; i < 8; i++)
                        //printf("after 2 min regMin02[%d]: %d\n", i, result[i]);
					regNewPathMin = _mm_min_epi16(regNewPathMin, regMin02);
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regNewPathMin);
                    //for (int i = 0; i < 8; i++)
                        //printf("regNewPathMin[%d]: %d\n", i, result[i]);

					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost0);
                    //for (int i = 0; i < 8; i++)
                        //printf("pathCost0[%d]: %d\n", i, result[i]);
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regPathCost2);
                    //for (int i = 0; i < 8; i++)
                        //printf("pathCost2[%d]: %d\n", i, result[i]);
					__m128i regCostSum = _mm_load_si128(reinterpret_cast<const __m128i*>(costSumCurrent + d));
					regCostSum = _mm_adds_epi16(regCostSum, regPathCost0);
					regCostSum = _mm_adds_epi16(regCostSum, regPathCost2);
					//_mm_store_si128(reinterpret_cast<__m128i*>(result), regCostSum);
                    //for (int i = 0; i < 8; i++)
                        //printf("regCostSum[%d]: %d\n", i, result[i]);
                    
					_mm_store_si128(reinterpret_cast<__m128i*>(costSumCurrent + d), regCostSum);
                    //for (int i = 0; i < 8; i++)
                        //printf("costSumCurrent[%d]: %d\n", d+i, costSumCurrent[d+i]);
				}

				regNewPathMin = _mm_min_epi16(regNewPathMin, _mm_srli_si128(regNewPathMin, 8));
                //short result[8];
                //_mm_store_si128(reinterpret_cast<__m128i*>(result), regNewPathMin);
                //for (int i = 0; i < 8; i++)
                    //printf("regNewPathMin before shift[%d]: %d\n", i, result[i]);
                //_mm_store_si128(reinterpret_cast<__m128i*>(result), _mm_srli_si128(regNewPathMin, 8));
                //for (int i = 0; i < 8; i++)
                    //printf("regNewPathMin after shift[%d]: %d\n", i, result[i]);
                //_mm_store_si128(reinterpret_cast<__m128i*>(result), regNewPathMin);
                //for (int i = 0; i < 8; i++)
                    //printf("regNewPathMin after comp[%d]: %d\n", i, result[i]);
				_mm_storel_epi64(reinterpret_cast<__m128i*>(&pathMinCosts[0][pathMinX]), regNewPathMin);
                //for (int i = 0; i < 8; i++)
                    //printf("pathMinCosts[%d]: %d\n", i, pathMinCosts[0][pathMinX+i]);
			}

			if (processPassCount == processPassTotal - 1) {
				unsigned short* disparityRow = disparityImage + width_*y;

				for (int x = 0; x < width_; ++x) {
					short* costSumCurrent = costSumRow + disparityTotal_*x;
					int bestSumCost = costSumCurrent[0];
					int bestDisparity = 0;
					for (int d = 1; d < disparityTotal_; ++d) {
						if (costSumCurrent[d] < bestSumCost) {
							bestSumCost = costSumCurrent[d];
							bestDisparity = d;
						}
					}

					if (bestDisparity > 0 && bestDisparity < disparityTotal_ - 1) {
						int centerCostValue = costSumCurrent[bestDisparity];
						int leftCostValue = costSumCurrent[bestDisparity - 1];
						int rightCostValue = costSumCurrent[bestDisparity + 1];
						if (rightCostValue < leftCostValue) {
							bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
															 + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - leftCostValue)/2.0*disparityFactor_ + 0.5);
						} else {
							bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
															 + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - rightCostValue)/2.0*disparityFactor_ + 0.5);
						}
					} else {
						bestDisparity = static_cast<int>(bestDisparity*disparityFactor_);
					}

					disparityRow[x] = static_cast<unsigned short>(bestDisparity);
				}
			}

			std::swap(pathCosts[0], pathCosts[1]);
			std::swap(pathMinCosts[0], pathMinCosts[1]);
		}
	}
	delete[] pathCosts;
	delete[] pathMinCosts;
}

void SGMStereo::myperformSGMStep1(unsigned short* costImage, unsigned short* disparityImage) {
	const short costMax = SHRT_MAX;

	int widthStepCostImage = width_*disparityTotal_;

	short* costSums = sgmBuffer_;
	memset(costSums, 0, costSumBufferSize_*sizeof(short));
	
    pathCosts = new short*[pathRowBufferTotal_];
	pathMinCosts = new short*[pathRowBufferTotal_];

    int startX, endX, stepX;
    int startY, endY, stepY;
    startX = 0; endX = width_; stepX = 1;
    startY = 0; endY = height_; stepY = 1;

    for (int y = startY; y < endY; y += stepY) {
        for (int x = startX; x < endX; x += stepX) {
            unsigned short* pixelCostRow = costImage + widthStepCostImage*y;
            short* costSumRow = costSums + costSumBufferRowSize_*y;

            if (y % 2 ==0)
            {
                pathCosts[0] = costSums + costSumBufferSize_ + pathCostBufferSize_*0 + pathDisparitySize_ + 8;
                pathMinCosts[0] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
                    + pathMinCostBufferSize_*0 + pathTotal_*2;
                pathCosts[1] = costSums + costSumBufferSize_ + pathCostBufferSize_*1 + pathDisparitySize_ + 8;
                pathMinCosts[1] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
                    + pathMinCostBufferSize_*1 + pathTotal_*2;
            }
            else
            {
                pathCosts[1] = costSums + costSumBufferSize_ + pathCostBufferSize_*0 + pathDisparitySize_ + 8;
                pathMinCosts[1] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
                    + pathMinCostBufferSize_*0 + pathTotal_*2;
                pathCosts[0] = costSums + costSumBufferSize_ + pathCostBufferSize_*1 + pathDisparitySize_ + 8;
                pathMinCosts[0] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
                    + pathMinCostBufferSize_*1 + pathTotal_*2;
            }

            /*
            memset(pathCosts[0] - pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
            memset(pathCosts[0] + width_*pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
            memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_*sizeof(short));
            memset(pathMinCosts[0] + width_*pathTotal_, 0, pathTotal_*sizeof(short));
            */
            for (int i = 0; i < pathDisparitySize_; i++)
                *(pathCosts[0] - pathDisparitySize_ - 8 + i) = *(pathCosts[0] + width_*pathDisparitySize_ - 8) = 0;
            for (int i = 0; i < pathTotal_; i++)
                *(pathMinCosts[0] - pathTotal_) = *(pathMinCosts[0] + width_*pathTotal_) = 0;

            int pathMinX = x*pathTotal_;
            int pathX = pathMinX*disparitySize_;

            int previousPathMin0 = pathMinCosts[0][pathMinX - stepX*pathTotal_] + smoothnessPenaltyLarge_;
            int previousPathMin2 = pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;

            short* previousPathCosts0 = pathCosts[0] + pathX - stepX*pathDisparitySize_;
            short* previousPathCosts2 = pathCosts[1] + pathX + disparitySize_*2;

            previousPathCosts0[-1] = previousPathCosts0[disparityTotal_] = costMax;
            previousPathCosts2[-1] = previousPathCosts2[disparityTotal_] = costMax;

            short* pathCostCurrent = pathCosts[0] + pathX;
            const unsigned short* pixelCostCurrent = pixelCostRow + disparityTotal_*x;
            short* costSumCurrent = costSumRow + disparityTotal_*x;
           
            short pathMin0, pathMin2;
            pathMin0 = previousPathMin0;
            pathMin2 = previousPathMin2;
            short newPathMin[8];
            for (int i = 0; i < 8; i++)
                newPathMin[i] = costMax;

            for (int d = 0; d < disparityTotal_; d += 8) {
                const unsigned short* pixelCost = pixelCostCurrent + d;

                short tmp_pathCost0[8], tmp_pathCost2[8];
                for (int i = 0; i < 8; i++)
                {
                    tmp_pathCost0[i] = min(previousPathCosts0[d+i], previousPathCosts0[d-1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost0[i] = min(tmp_pathCost0[i], previousPathCosts0[d+1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost0[i] = min(tmp_pathCost0[i], pathMin0);
                    tmp_pathCost0[i] = short(pixelCost[i]) + tmp_pathCost0[i] - pathMin0;
                    
                    tmp_pathCost2[i] = min(previousPathCosts2[d+i], previousPathCosts2[d-1+i] + smoothnessPenaltySmall_);
                    //printf("cpu pathCosts2[%d]: %d previousPathCosts2[%d]: %d\n", i, previousPathCost2[d+i], d-1+i, previousPathCosts2[d-1+i]);
                    //printf("cpu comp: %d\n", costSums[costSumBufferSize_ + pathCostBufferSize_*1 + pathDisparitySize_ + 8 + x*pathTotal_*disparitySize_ + disparitySize_*2
                    //        +d-1+i]);
                    tmp_pathCost2[i] = min(tmp_pathCost2[i], previousPathCosts2[d+1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost2[i] = min(tmp_pathCost2[i], pathMin2);
                    tmp_pathCost2[i] = short(pixelCost[i]) + tmp_pathCost2[i] - pathMin2;

                    pathCostCurrent[d+i] = tmp_pathCost0[i];
                    //printf("cpu pathCostCurrent[%d]: %d costSums[%d]: %d\n", d+i, pathCostCurrent[d+i], pathCostCurrent+d+i-costSums, costSums[pathCostCurrent+d+i-costSums]);
                    pathCostCurrent[d+disparitySize_*2+i] = tmp_pathCost2[i];
                    //printf("cpu pathCostCurrent[%d]: %d\n", d+disparitySize_*2+i, pathCostCurrent[d+disparitySize_*2+i]);
                    
                    costSumCurrent[d+i] += tmp_pathCost0[i] + tmp_pathCost2[i];
                    //printf("cpu costSumCurrent[%d]: %d\n", d+i, costSumCurrent[d+i]);
                }
                //return;

                short min02[8], tmp_min02[8];
                for (int i = 0; i < 4; i++)
                {
                    tmp_min02[2*i] = min(tmp_pathCost0[i], tmp_pathCost0[i+4]);
                    tmp_min02[2*i+1] = min(tmp_pathCost2[i], tmp_pathCost2[i+4]);
                }
                for (int i = 0; i < 4; i++)
                {
                    min02[2*i] = min02[2*i+1] = min(tmp_min02[i], tmp_min02[i+4]);
                }
                for (int i = 0; i < 8; i++)
                {
                    newPathMin[i] = min(newPathMin[i], min02[i]);
                }
            }

            for (int i = 0; i < 4; i++)
            {
                newPathMin[i] = min(newPathMin[i], newPathMin[4+i]);
                pathMinCosts[0][pathMinX+i] = newPathMin[i];
            }
            for (int i = 0; i < 4; i++)
            {
                newPathMin[4+i] = min(newPathMin[4+i], 0);
            }
        }

        //swap(pathCosts[0], pathCosts[1]);
        //swap(pathMinCosts[0], pathMinCosts[1]);
        /*
        short* tmp;
        tmp = pathCosts[0];
        pathCosts[0] = pathCosts[1];
        pathCosts[1] = tmp;
        tmp = pathMinCosts[0];
        pathMinCosts[0] = pathMinCosts[1];
        pathMinCosts[1] = tmp;

        printf("y: %d pathMinCosts[0]: %p\n", y, pathMinCosts[0]);
        printf("y: %d pathMinCosts[1]: %p\n", y, pathMinCosts[1]);
        */
    }
}

void SGMStereo::myperformSGMStep2(unsigned short* costImage, unsigned short* disparityImage) {
	const short costMax = SHRT_MAX;

	int widthStepCostImage = width_*disparityTotal_;
	
    short* costSums = sgmBuffer_;

    int startX, endX, stepX;
    int startY, endY, stepY;
    startX = width_ - 1; endX = -1; stepX = -1;
    startY = height_ - 1; endY = -1; stepY = -1;

    for (int i = 0; i < pathRowBufferTotal_; ++i) {
        pathCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*i + pathDisparitySize_ + 8;
        memset(pathCosts[i] - pathDisparitySize_ - 8, 0, pathCostBufferSize_*sizeof(short));
        pathMinCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
            + pathMinCostBufferSize_*i + pathTotal_*2;
        memset(pathMinCosts[i] - pathTotal_, 0, pathMinCostBufferSize_*sizeof(short));
    }

    for (int y = startY; y != endY; y += stepY) {
        unsigned short* pixelCostRow = costImage + widthStepCostImage*y;
        short* costSumRow = costSums + costSumBufferRowSize_*y;

        memset(pathCosts[0] - pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
        memset(pathCosts[0] + width_*pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
        memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_*sizeof(short));
        memset(pathMinCosts[0] + width_*pathTotal_, 0, pathTotal_*sizeof(short));

        for (int x = startX; x != endX; x += stepX) {
            int pathMinX = x*pathTotal_;
            int pathX = pathMinX*disparitySize_;

            int previousPathMin0 = pathMinCosts[0][pathMinX - stepX*pathTotal_] + smoothnessPenaltyLarge_;
            int previousPathMin2 = pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;

            short* previousPathCosts0 = pathCosts[0] + pathX - stepX*pathDisparitySize_;
            short* previousPathCosts2 = pathCosts[1] + pathX + disparitySize_*2;

            previousPathCosts0[-1] = previousPathCosts0[disparityTotal_] = costMax;
            previousPathCosts2[-1] = previousPathCosts2[disparityTotal_] = costMax;

            short* pathCostCurrent = pathCosts[0] + pathX;
            const unsigned short* pixelCostCurrent = pixelCostRow + disparityTotal_*x;
            short* costSumCurrent = costSumRow + disparityTotal_*x;
           
            short pathMin0, pathMin2;
            pathMin0 = previousPathMin0;
            pathMin2 = previousPathMin2;
            short newPathMin[8];
            for (int i = 0; i < 8; i++)
                newPathMin[i] = costMax;

            for (int d = 0; d < disparityTotal_; d += 8) {
                const unsigned short* pixelCost = pixelCostCurrent + d;

                short* pathCost0 = previousPathCosts0 + d;
                short* pathCost2 = previousPathCosts2 + d;

                short tmp_pathCost0[8], tmp_pathCost2[8];
                for (int i = 0; i < 8; i++)
                {
                    tmp_pathCost0[i] = min(pathCost0[i], previousPathCosts0[d-1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost0[i] = min(tmp_pathCost0[i], previousPathCosts0[d+1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost0[i] = min(tmp_pathCost0[i], pathMin0);
                    tmp_pathCost0[i] = short(pixelCost[i]) + tmp_pathCost0[i] - pathMin0;

                    tmp_pathCost2[i] = min(pathCost2[i], previousPathCosts2[d-1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost2[i] = min(tmp_pathCost2[i], previousPathCosts2[d+1+i] + smoothnessPenaltySmall_);
                    tmp_pathCost2[i] = min(tmp_pathCost2[i], pathMin2);
                    tmp_pathCost2[i] = short(pixelCost[i]) + tmp_pathCost2[i] - pathMin2;

                    pathCostCurrent[d+i] = tmp_pathCost0[i];
                    pathCostCurrent[d+disparitySize_*2+i] = tmp_pathCost2[i];

                    costSumCurrent[d+i] += tmp_pathCost0[i] + tmp_pathCost2[i];
                }

                short min02[8], tmp_min02[8];
                for (int i = 0; i < 4; i++)
                {
                    tmp_min02[2*i] = min(tmp_pathCost0[i], tmp_pathCost0[i+4]);
                    tmp_min02[2*i+1] = min(tmp_pathCost2[i], tmp_pathCost2[i+4]);
                }
                for (int i = 0; i < 4; i++)
                {
                    min02[2*i] = min02[2*i+1] = min(tmp_min02[i], tmp_min02[i+4]);
                }
                for (int i = 0; i < 8; i++)
                {
                    newPathMin[i] = min(newPathMin[i], min02[i]);
                }
            }

            for (int i = 0; i < 4; i++)
            {
                newPathMin[i] = min(newPathMin[i], newPathMin[4+i]);
                pathMinCosts[0][pathMinX+i] = newPathMin[i];
            }
            for (int i = 0; i < 4; i++)
            {
                newPathMin[4+i] = min(newPathMin[4+i], 0);
            }
        }

        unsigned short* disparityRow = disparityImage + width_*y;

        for (int x = 0; x < width_; ++x) {
            short* costSumCurrent = costSumRow + disparityTotal_*x;
            int bestSumCost = costSumCurrent[0];
            int bestDisparity = 0;
            for (int d = 1; d < disparityTotal_; ++d) {
                if (costSumCurrent[d] < bestSumCost) {
                    bestSumCost = costSumCurrent[d];
                    bestDisparity = d;
                }
            }

            if (bestDisparity > 0 && bestDisparity < disparityTotal_ - 1) {
                int centerCostValue = costSumCurrent[bestDisparity];
                int leftCostValue = costSumCurrent[bestDisparity - 1];
                int rightCostValue = costSumCurrent[bestDisparity + 1];
                if (rightCostValue < leftCostValue) {
                    bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
                                                     + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - leftCostValue)/2.0*disparityFactor_ + 0.5);
                } else {
                    bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
                                                     + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - rightCostValue)/2.0*disparityFactor_ + 0.5);
                }
            } else {
                bestDisparity = static_cast<int>(bestDisparity*disparityFactor_);
            }

            disparityRow[x] = static_cast<unsigned short>(bestDisparity);
        }

        std::swap(pathCosts[0], pathCosts[1]);
        std::swap(pathMinCosts[0], pathMinCosts[1]);
    }

	delete[] pathCosts;
	delete[] pathMinCosts;
}


void SGMStereo::myperformSGM(unsigned short* costImage, unsigned short* disparityImage) {
	const short costMax = SHRT_MAX;

	int widthStepCostImage = width_*disparityTotal_;

	short* costSums = sgmBuffer_;
	memset(costSums, 0, costSumBufferSize_*sizeof(short));

	short** pathCosts = new short*[pathRowBufferTotal_];
	short** pathMinCosts = new short*[pathRowBufferTotal_];

	const int processPassTotal = 2;
	for (int processPassCount = 0; processPassCount < processPassTotal; ++processPassCount) {
		int startX, endX, stepX;
		int startY, endY, stepY;
		if (processPassCount == 0) {
			startX = 0; endX = width_; stepX = 1;
			startY = 0; endY = height_; stepY = 1;
		} else {
			startX = width_ - 1; endX = -1; stepX = -1;
			startY = height_ - 1; endY = -1; stepY = -1;
		}

		for (int i = 0; i < pathRowBufferTotal_; ++i) {
			pathCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*i + pathDisparitySize_ + 8;
			memset(pathCosts[i] - pathDisparitySize_ - 8, 0, pathCostBufferSize_*sizeof(short));
			pathMinCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_
				+ pathMinCostBufferSize_*i + pathTotal_*2;
			memset(pathMinCosts[i] - pathTotal_, 0, pathMinCostBufferSize_*sizeof(short));
		}

		for (int y = startY; y != endY; y += stepY) {
			unsigned short* pixelCostRow = costImage + widthStepCostImage*y;
			short* costSumRow = costSums + costSumBufferRowSize_*y;

			memset(pathCosts[0] - pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathCosts[0] + width_*pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_*sizeof(short));
			memset(pathMinCosts[0] + width_*pathTotal_, 0, pathTotal_*sizeof(short));

			for (int x = startX; x != endX; x += stepX) {
				int pathMinX = x*pathTotal_;
				int pathX = pathMinX*disparitySize_;

				int previousPathMin0 = pathMinCosts[0][pathMinX - stepX*pathTotal_] + smoothnessPenaltyLarge_;
				int previousPathMin2 = pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;

				short* previousPathCosts0 = pathCosts[0] + pathX - stepX*pathDisparitySize_;
				short* previousPathCosts2 = pathCosts[1] + pathX + disparitySize_*2;

				previousPathCosts0[-1] = previousPathCosts0[disparityTotal_] = costMax;
				previousPathCosts2[-1] = previousPathCosts2[disparityTotal_] = costMax;

				short* pathCostCurrent = pathCosts[0] + pathX;
				const unsigned short* pixelCostCurrent = pixelCostRow + disparityTotal_*x;
				short* costSumCurrent = costSumRow + disparityTotal_*x;
               
                short pathMin0, pathMin2;
                pathMin0 = previousPathMin0;
                pathMin2 = previousPathMin2;
                short newPathMin[8];
                for (int i = 0; i < 8; i++)
                    newPathMin[i] = costMax;

				for (int d = 0; d < disparityTotal_; d += 8) {
                    const unsigned short* pixelCost = pixelCostCurrent + d;

                    short* pathCost0 = previousPathCosts0 + d;
                    short* pathCost2 = previousPathCosts2 + d;

                    short tmp_pathCost0[8], tmp_pathCost2[8];
                    for (int i = 0; i < 8; i++)
                    {
                        tmp_pathCost0[i] = min(pathCost0[i], previousPathCosts0[d-1+i] + smoothnessPenaltySmall_);
                        tmp_pathCost0[i] = min(tmp_pathCost0[i], previousPathCosts0[d+1+i] + smoothnessPenaltySmall_);
                        tmp_pathCost2[i] = min(pathCost2[i], previousPathCosts2[d-1+i] + smoothnessPenaltySmall_);
                        tmp_pathCost2[i] = min(tmp_pathCost2[i], previousPathCosts2[d+1+i] + smoothnessPenaltySmall_);
                    }

                    for (int i = 0; i < 8; i++)
                    {
                        tmp_pathCost0[i] = min(tmp_pathCost0[i], pathMin0);
                        tmp_pathCost0[i] = short(pixelCost[i]) + tmp_pathCost0[i] - pathMin0;
                        tmp_pathCost2[i] = min(tmp_pathCost2[i], pathMin2);
                        tmp_pathCost2[i] = short(pixelCost[i]) + tmp_pathCost2[i] - pathMin2;
                    }

                    for (int i = 0; i < 8; i++)
                    {
                        pathCostCurrent[d+i] = tmp_pathCost0[i];
                        pathCostCurrent[d+disparitySize_*2+i] = tmp_pathCost2[i];
                    }

                    short min02[8], tmp_min02[8];
                    for (int i = 0; i < 4; i++)
                    {
                        tmp_min02[2*i] = min(tmp_pathCost0[i], tmp_pathCost0[i+4]);
                        tmp_min02[2*i+1] = min(tmp_pathCost2[i], tmp_pathCost2[i+4]);
                    }
                    for (int i = 0; i < 4; i++)
                    {
                        min02[2*i] = min02[2*i+1] = min(tmp_min02[i], tmp_min02[i+4]);
                    }
                    for (int i = 0; i < 8; i++)
                    {
                        newPathMin[i] = min(newPathMin[i], min02[i]);
                    }

                    for (int i = 0; i < 8; i++)
                    {
                        costSumCurrent[d+i] += tmp_pathCost0[i] + tmp_pathCost2[i];
                    }
				}

                for (int i = 0; i < 4; i++)
                {
                    newPathMin[i] = min(newPathMin[i], newPathMin[4+i]);
                    pathMinCosts[0][pathMinX+i] = newPathMin[i];
                }
                for (int i = 0; i < 4; i++)
                {
                    newPathMin[4+i] = min(newPathMin[4+i], 0);
                }
			}

			if (processPassCount == processPassTotal - 1) {
				unsigned short* disparityRow = disparityImage + width_*y;

				for (int x = 0; x < width_; ++x) {
					short* costSumCurrent = costSumRow + disparityTotal_*x;
					int bestSumCost = costSumCurrent[0];
					int bestDisparity = 0;
					for (int d = 1; d < disparityTotal_; ++d) {
						if (costSumCurrent[d] < bestSumCost) {
							bestSumCost = costSumCurrent[d];
							bestDisparity = d;
						}
					}

					if (bestDisparity > 0 && bestDisparity < disparityTotal_ - 1) {
						int centerCostValue = costSumCurrent[bestDisparity];
						int leftCostValue = costSumCurrent[bestDisparity - 1];
						int rightCostValue = costSumCurrent[bestDisparity + 1];
						if (rightCostValue < leftCostValue) {
							bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
															 + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - leftCostValue)/2.0*disparityFactor_ + 0.5);
						} else {
							bestDisparity = static_cast<int>(bestDisparity*disparityFactor_
															 + static_cast<double>(rightCostValue - leftCostValue)/(centerCostValue - rightCostValue)/2.0*disparityFactor_ + 0.5);
						}
					} else {
						bestDisparity = static_cast<int>(bestDisparity*disparityFactor_);
					}

					disparityRow[x] = static_cast<unsigned short>(bestDisparity);
				}
			}

			std::swap(pathCosts[0], pathCosts[1]);
			std::swap(pathMinCosts[0], pathMinCosts[1]);
		}
	}
	delete[] pathCosts;
	delete[] pathMinCosts;
}

void SGMStereo::speckleFilter(const int maxSpeckleSize, const int maxDifference, unsigned short* image) const {
	std::vector<int> labels(width_*height_, 0);
	std::vector<bool> regionTypes(1);
	regionTypes[0] = false;

	int currentLabelIndex = 0;

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelIndex = width_*y + x;
			if (image[width_*y + x] != 0) {
				if (labels[pixelIndex] > 0) {
					if (regionTypes[labels[pixelIndex]]) {
						image[width_*y + x] = 0;
					}
				} else {
					std::stack<int> wavefrontIndices;
					wavefrontIndices.push(pixelIndex);
					++currentLabelIndex;
					regionTypes.push_back(false);
					int regionPixelTotal = 0;
					labels[pixelIndex] = currentLabelIndex;

					while (!wavefrontIndices.empty()) {
						int currentPixelIndex = wavefrontIndices.top();
						wavefrontIndices.pop();
						int currentX = currentPixelIndex%width_;
						int currentY = currentPixelIndex/width_;
						++regionPixelTotal;
						unsigned short pixelValue = image[width_*currentY + currentX];

						if (currentX < width_ - 1 && labels[currentPixelIndex + 1] == 0
							&& image[width_*currentY + currentX + 1] != 0
							&& std::abs(pixelValue - image[width_*currentY + currentX + 1]) <= maxDifference)
						{
							labels[currentPixelIndex + 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + 1);
						}

						if (currentX > 0 && labels[currentPixelIndex - 1] == 0
							&& image[width_*currentY + currentX - 1] != 0
							&& std::abs(pixelValue - image[width_*currentY + currentX - 1]) <= maxDifference)
						{
							labels[currentPixelIndex - 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - 1);
						}

						if (currentY < height_ - 1 && labels[currentPixelIndex + width_] == 0
							&& image[width_*(currentY + 1) + currentX] != 0
							&& std::abs(pixelValue - image[width_*(currentY + 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex + width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + width_);
						}

						if (currentY > 0 && labels[currentPixelIndex - width_] == 0
							&& image[width_*(currentY - 1) + currentX] != 0
							&& std::abs(pixelValue - image[width_*(currentY - 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex - width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - width_);
						}
					}

					if (regionPixelTotal <= maxSpeckleSize) {
						regionTypes[currentLabelIndex] = true;
						image[width_*y + x] = 0;
					}
				}
			}
		}
	}
}

void SGMStereo::enforceLeftRightConsistency(unsigned short* leftDisparityImage, unsigned short* rightDisparityImage) const {
	// Check left disparity image
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (leftDisparityImage[width_*y + x] == 0) continue;

			int leftDisparityValue = static_cast<int>(static_cast<double>(leftDisparityImage[width_*y + x])/disparityFactor_ + 0.5);
			if (x - leftDisparityValue < 0) {
				leftDisparityImage[width_*y + x] = 0;
				continue;
			}

			int rightDisparityValue = static_cast<int>(static_cast<double>(rightDisparityImage[width_*y + x-leftDisparityValue])/disparityFactor_ + 0.5);
			if (rightDisparityValue == 0 || abs(leftDisparityValue - rightDisparityValue) > consistencyThreshold_) {
				leftDisparityImage[width_*y + x] = 0;
			}
		}
	}

	// Check right disparity image
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (rightDisparityImage[width_*y + x] == 0)  continue;

			int rightDisparityValue = static_cast<int>(static_cast<double>(rightDisparityImage[width_*y + x])/disparityFactor_ + 0.5);
			if (x + rightDisparityValue >= width_) {
				rightDisparityImage[width_*y + x] = 0;
				continue;
			}

			int leftDisparityValue = static_cast<int>(static_cast<double>(leftDisparityImage[width_*y + x+rightDisparityValue])/disparityFactor_ + 0.5);
			if (leftDisparityValue == 0 || abs(rightDisparityValue - leftDisparityValue) > consistencyThreshold_) {
				rightDisparityImage[width_*y + x] = 0;
			}
		}
	}
}

void SGMStereo::myenforceLeftRightConsistency(unsigned char* leftDisparityImage, unsigned char* rightDisparityImage) const {
	// Check left disparity image
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (leftDisparityImage[width_*y + x] == 0) continue;

			int leftDisparityValue = leftDisparityImage[width_*y + x];
			if (x - leftDisparityValue < 0) {
				leftDisparityImage[width_*y + x] = 0;
				continue;
			}

			int rightDisparityValue = rightDisparityImage[width_*y + x-leftDisparityValue];
			if (rightDisparityValue == 0 || abs(leftDisparityValue - rightDisparityValue) > consistencyThreshold_) {
				leftDisparityImage[width_*y + x] = 0;
			}
		}
	}

	// Check right disparity image
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (rightDisparityImage[width_*y + x] == 0)  continue;

			int rightDisparityValue = rightDisparityImage[width_*y + x];
			if (x + rightDisparityValue >= width_) {
				rightDisparityImage[width_*y + x] = 0;
				continue;
			}

			int leftDisparityValue = leftDisparityImage[width_*y + x+rightDisparityValue];
			if (leftDisparityValue == 0 || abs(rightDisparityValue - leftDisparityValue) > consistencyThreshold_) {
				rightDisparityImage[width_*y + x] = 0;
			}
		}
	}
}

void SGMStereo::myspeckleFilter(const int maxSpeckleSize, const int maxDifference, unsigned char* image) const {
	std::vector<int> labels(width_*height_, 0);
	std::vector<bool> regionTypes(1);
	regionTypes[0] = false;

	int currentLabelIndex = 0;

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelIndex = width_*y + x;
			if (image[width_*y + x] != 0) {
				if (labels[pixelIndex] > 0) {
					if (regionTypes[labels[pixelIndex]]) {
						image[width_*y + x] = 0;
					}
				} else {
					std::stack<int> wavefrontIndices;
					wavefrontIndices.push(pixelIndex);
					++currentLabelIndex;
					regionTypes.push_back(false);
                    int regionPixelTotal = 0;
					labels[pixelIndex] = currentLabelIndex;

					while (!wavefrontIndices.empty()) {
						int currentPixelIndex = wavefrontIndices.top();
						wavefrontIndices.pop();
						int currentX = currentPixelIndex%width_;
						int currentY = currentPixelIndex/width_;
						++regionPixelTotal;
						unsigned char pixelValue = image[width_*currentY + currentX];

						if (currentX < width_ - 1 && labels[currentPixelIndex + 1] == 0
							&& image[width_*currentY + currentX + 1] != 0
							&& std::abs(pixelValue - image[width_*currentY + currentX + 1]) <= maxDifference)
						{
							labels[currentPixelIndex + 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + 1);
						}

						if (currentX > 0 && labels[currentPixelIndex - 1] == 0
							&& image[width_*currentY + currentX - 1] != 0
							&& std::abs(pixelValue - image[width_*currentY + currentX - 1]) <= maxDifference)
						{
							labels[currentPixelIndex - 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - 1);
						}

						if (currentY < height_ - 1 && labels[currentPixelIndex + width_] == 0
							&& image[width_*(currentY + 1) + currentX] != 0
							&& std::abs(pixelValue - image[width_*(currentY + 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex + width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + width_);
						}

						if (currentY > 0 && labels[currentPixelIndex - width_] == 0
							&& image[width_*(currentY - 1) + currentX] != 0
							&& std::abs(pixelValue - image[width_*(currentY - 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex - width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - width_);
						}
					}

					if (regionPixelTotal <= maxSpeckleSize) {
						regionTypes[currentLabelIndex] = true;
						image[width_*y + x] = 0;
					}
				}
			}
		}
	}
}

