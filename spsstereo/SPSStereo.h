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

#pragma once

#include <vector>
#include <stack>
#include <png++/png.hpp>

class SPSStereo {
public:
	SPSStereo();

	void setOutputDisparityFactor(const double outputDisparityFactor);
	void setIterationTotal(const int outerIterationTotal, const int innerIterationTotal);
	void setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight);
	void setInlierThreshold(const double inlierThreshold);
	void setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty);

	void compute(const int superpixelTotal,
				 const png::image<png::rgb_pixel>& leftImage,
				 const png::image<png::rgb_pixel>& rightImage,
				 png::image<png::gray_pixel_16>& segmentImage,
				 png::image<png::gray_pixel_16>& disparityImage,
				 std::vector< std::vector<double> >& disparityPlaneParameters,
				 std::vector< std::vector<int> >& boundaryLabels);

private:
	class Segment {
	public:
		Segment() {
			pixelTotal_ = 0;
			colorSum_[0] = 0;  colorSum_[1] = 0;  colorSum_[2] = 0;
			positionSum_[0] = 0;  positionSum_[1] = 0;
			disparityPlane_[0] = 0;  disparityPlane_[1] = 0;  disparityPlane_[2] = -1;
		}

		void addPixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
			pixelTotal_ += 1;
			colorSum_[0] += colorL;  colorSum_[1] += colorA;  colorSum_[2] += colorB;
			positionSum_[0] += x;  positionSum_[1] += y;
		}
		void removePixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
			pixelTotal_ -= 1;
			colorSum_[0] -= colorL;  colorSum_[1] -= colorA;  colorSum_[2] -= colorB;
			positionSum_[0] -= x;  positionSum_[1] -= y;
		}
		void setDisparityPlane(const double planeGradientX, const double planeGradientY, const double planeConstant) {
			disparityPlane_[0] = planeGradientX;
			disparityPlane_[1] = planeGradientY;
			disparityPlane_[2] = planeConstant;
		}

		int pixelTotal() const { return pixelTotal_; }
		double color(const int colorIndex) const { return colorSum_[colorIndex]/pixelTotal_; }
		double position(const int coordinateIndex) const { return positionSum_[coordinateIndex]/pixelTotal_; }
		double estimatedDisparity(const double x, const double y) const { return disparityPlane_[0]*x + disparityPlane_[1]*y + disparityPlane_[2]; }
		bool hasDisparityPlane() const { if (disparityPlane_[0] != 0.0 || disparityPlane_[1] != 0.0 || disparityPlane_[2] != -1.0) return true; else return false; }

		void clearConfiguration() {
			neighborSegmentIndices_.clear();
			boundaryIndices_.clear();
			for (int i = 0; i < 9; ++i) polynomialCoefficients_[i] = 0;
			for (int i = 0; i < 6; ++i) polynomialCoefficientsAll_[i] = 0;
		}
		void appendBoundaryIndex(const int boundaryIndex) { boundaryIndices_.push_back(boundaryIndex); }
		void appendSegmentPixel(const int x, const int y) {
			polynomialCoefficientsAll_[0] += x*x;
			polynomialCoefficientsAll_[1] += y*y;
			polynomialCoefficientsAll_[2] += x*y;
			polynomialCoefficientsAll_[3] += x;
			polynomialCoefficientsAll_[4] += y;
			polynomialCoefficientsAll_[5] += 1;
		}
		void appendSegmentPixelWithDisparity(const int x, const int y, const double d) {
			polynomialCoefficients_[0] += x*x;
			polynomialCoefficients_[1] += y*y;
			polynomialCoefficients_[2] += x*y;
			polynomialCoefficients_[3] += x;
			polynomialCoefficients_[4] += y;
			polynomialCoefficients_[5] += x*d;
			polynomialCoefficients_[6] += y*d;
			polynomialCoefficients_[7] += d;
			polynomialCoefficients_[8] += 1;
		}

		int neighborTotal() const { return static_cast<int>(neighborSegmentIndices_.size()); }
		int neighborIndex(const int index) const { return neighborSegmentIndices_[index]; }

		int boundaryTotal() const { return static_cast<int>(boundaryIndices_.size()); }
		int boundaryIndex(const int index) const { return boundaryIndices_[index]; }

		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }
		double polynomialCoefficientAll(const int index) const { return polynomialCoefficientsAll_[index]; }

		double planeParameter(const int index) const { return disparityPlane_[index]; }

	private:
		int pixelTotal_;
		double colorSum_[3];
		double positionSum_[2];
		double disparityPlane_[3];

		std::vector<int> neighborSegmentIndices_;
		std::vector<int> boundaryIndices_;
		double polynomialCoefficients_[9];
		double polynomialCoefficientsAll_[6];
	};
	class Boundary {
	public:
		Boundary() { segmentIndices_[0] = -1; segmentIndices_[1] = -1; clearCoefficients(); }
		Boundary(const int firstSegmentIndex, const int secondSegmentIndex) {
			if (firstSegmentIndex < secondSegmentIndex) {
				segmentIndices_[0] = firstSegmentIndex; segmentIndices_[1] = secondSegmentIndex;
			} else {
				segmentIndices_[0] = secondSegmentIndex; segmentIndices_[1] = firstSegmentIndex;
			}
			clearCoefficients();
		}

		void clearCoefficients() {
			for (int i = 0; i < 6; ++i) polynomialCoefficients_[i] = 0;
		}

		void setType(const int typeIndex) { type_ = typeIndex; }
		void appendBoundaryPixel(const double x, const double y) {
			boundaryPixelXs_.push_back(x);
			boundaryPixelYs_.push_back(y);
			polynomialCoefficients_[0] += x*x;
			polynomialCoefficients_[1] += y*y;
			polynomialCoefficients_[2] += x*y;
			polynomialCoefficients_[3] += x;
			polynomialCoefficients_[4] += y;
			polynomialCoefficients_[5] += 1;
		}

		int type() const { return type_; }
		int segmentIndex(const int index) const { return segmentIndices_[index]; }
		bool consistOf(const int firstSegmentIndex, const int secondSegmentIndex) const {
			if ((firstSegmentIndex == segmentIndices_[0] && secondSegmentIndex == segmentIndices_[1])
				|| (firstSegmentIndex == segmentIndices_[1] && secondSegmentIndex == segmentIndices_[0]))
			{
				return true;
			}
			return false;
		}
		int include(const int segmentIndex) const {
			if (segmentIndex == segmentIndices_[0]) return 0;
			else if (segmentIndex == segmentIndices_[1]) return 1;
			else return -1;
		}
		int boundaryPixelTotal() const { return static_cast<int>(boundaryPixelXs_.size()); }
		double boundaryPixelX(const int index) const { return boundaryPixelXs_[index]; }
		double boundaryPixelY(const int index) const { return boundaryPixelYs_[index]; }

		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }

	private:
		int type_;
		int segmentIndices_[2];
		std::vector<double> boundaryPixelXs_;
		std::vector<double> boundaryPixelYs_;

		double polynomialCoefficients_[6];
	};


	void allocateBuffer();
	void freeBuffer();
	void setInputData(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
	void setLabImage(const png::image<png::rgb_pixel>& leftImage);
	void computeInitialDisparityImage(const png::image<png::rgb_pixel>& leftImage, const png::image<png::rgb_pixel>& rightImage);
	void initializeSegment(const int superpixelTotal);
	void makeGridSegment(const int superpixelTotal);
	void assignLabel();
	void extractBoundaryPixel(std::stack<int>& boundaryPixelIndices);
	bool isBoundaryPixel(const int x, const int y) const;
	bool isUnchangeable(const int x, const int y) const;
	int findBestSegmentLabel(const int x, const int y) const;
	std::vector<int> getNeighborSegmentIndices(const int x, const int y) const;
	double computePixelEnergy(const int x, const int y, const int segmentIndex) const;
	double computeBoundaryLengthEnergy(const int x, const int y, const int segmentIndex) const;
	void changeSegmentLabel(const int x, const int y, const int newSegmentIndex);
	void addNeighborBoundaryPixel(const int x, const int y, std::stack<int>& boundaryPixelIndices) const;
	void initialFitDisparityPlane();
	void estimateDisparityPlaneRANSAC(const float* disparityImage);
	void solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
							 const double x2, const double y2, const double z2, const double d2,
							 const double x3, const double y3, const double z3, const double d3,
							 std::vector<double>& planeParameter) const;
	int computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const;
	void interpolateDisparityImage(float* interpolatedDisparityImage) const;
	void initializeOutlierFlagImage();
	void performSmoothingSegmentation();
	void buildSegmentConfiguration();
	bool isHorizontalBoundary(const int x, const int y) const;
	bool isVerticalBoundary(const int x, const int y) const;
	int appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex);
	void planeSmoothing();
	void estimateBoundaryLabel();
	void estimateSmoothFitting();
	void makeOutputImage(png::image<png::gray_pixel_16>& segmentImage, png::image<png::gray_pixel_16>& segmentDisparityImage) const;
	void makeSegmentBoundaryData(std::vector< std::vector<double> >& disparityPlaneParameters, std::vector< std::vector<int> >& boundaryLabels) const;


	// Parameter
	double outputDisparityFactor_;
	int outerIterationTotal_;
	int innerIterationTotal_;
	double positionWeight_;
	double disparityWeight_;
	double boundaryLengthWeight_;
	double smoothRelativeWeight_;
	double inlierThreshold_;
	double hingePenalty_;
	double occlusionPenalty_;
	double impossiblePenalty_;

	// Input data
	int width_;
	int height_;
	float* inputLabImage_;
	float* initialDisparityImage_;

	// Superpixel segments
	int segmentTotal_;
	std::vector<Segment> segments_;
	int stepSize_;
	int* labelImage_;
	unsigned char* outlierFlagImage_;
	unsigned char* boundaryFlagImage_;
	std::vector<Boundary> boundaries_;
	std::vector< std::vector<int> > boundaryIndexMatrix_;
};
