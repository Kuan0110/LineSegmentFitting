#pragma once

#include <iostream>
#include <vector>

#include <pcl/common/common.h>

#include <Eigen/Dense>

namespace line_fitting {

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point>::Ptr PointCloudPtr;
typedef std::array<double, 3> LineCoefficients;

class LineSegment2D {
 public:
	static std::vector<std::pair<int, double>> getPointIndicesCloseToLine(
		const PointCloudPtr& cloud, const LineCoefficients& coeffs, const double distance_thresh,
		const std::vector<char>& ignore_point_indices);

	static double getDistancePoint2Line(const Eigen::Vector2d& point, const LineCoefficients& coeffs);

	LineSegment2D();
	
	LineSegment2D(const PointCloudPtr& cloud, 
		const LineCoefficients& coeffs, const std::array<double, 4>& range);

	LineSegment2D(const LineSegment2D& line_segment);

	bool refine(const double distance_Thresh, const std::vector<char>& ignore_indices);

	bool PCALineFit(const Eigen::MatrixXd& point_matrix);

	void clipLineSegment(const Eigen::Vector2d& point);

	double inline getSegmentLength() {return std::sqrt(std::pow(endpoints_[0] - endpoints_[2], 2) + std::pow(endpoints_[1] - endpoints_[3], 2));}

	const auto& coeffs() const {return coeffs_;}

	const auto& endpoints() const {return endpoints_;}

	const auto& inlierIndices() const {return inlier_indices_;}

 private:
	template <typename T>
	bool almostEquals(const T val, const T correctVal, const T epsilon = std::numeric_limits<T>::epsilon()) {
		const T maxXYOne = std::max({static_cast<T>(1.0d), std::fabs(val), std::fabs(correctVal)});
		return std::fabs(val - correctVal) <= epsilon * maxXYOne;
	}

 private:
	const PointCloudPtr& cloud_;
	LineCoefficients coeffs_;
	std::array<double, 4> endpoints_;
	std::vector<int> inlier_indices_;
};
}  // namespace line_fitting

