#pragma once

#include <pcl/common/common.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <vector>

namespace line_fitting {

typedef Eigen::Vector2d Point2d;
typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point>::Ptr PointCloudPtr;
typedef std::array<double, 3> LineCoefficients;

class LineSegment2D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  static std::vector<std::pair<int, double>> getPointIndicesCloseToLine(
      const PointCloudPtr& cloud, const LineCoefficients& coeffs,
      const double distance_thresh,
      const std::vector<char>& ignore_point_indices);

  static double getDistancePoint2Line(const Point2d& point,
                                      const LineCoefficients& coeffs);

  LineSegment2D(const PointCloudPtr& cloud, const std::vector<Point2d>& range);

  LineSegment2D(const PointCloudPtr& cloud, const LineCoefficients& coeffs,
                const std::vector<Point2d>& range);

  LineSegment2D(const LineSegment2D& line_segment);

  bool refine(const double distance_Thresh,
              const std::vector<char>& ignore_indices);

  bool PCALineFit(const Eigen::MatrixXd& point_matrix);

  bool fitLineTLS(const Eigen::MatrixXd& point_matrix);

  void clipLineSegment(const Point2d& point);

	Point2d getProjection(const Point2d& point) const;

  bool isOnLineSegment(const Point2d& point) const;

  void inline addInliers(const Point2d& inlier) {
    raw_points_.push_back(inlier);
  }

  double inline getSegmentLength() {
    return std::sqrt(std::pow(endpoints_[0].x() - endpoints_[1].x(), 2) +
                     std::pow(endpoints_[0].y() - endpoints_[1].y(), 2));
  }

  void inline setEndpoints(int idx, const Point2d& p) { endpoints_[idx] = p; }

  const auto& coeffs() const { return coeffs_; }

  const auto& endpoints() const { return endpoints_; }

  const auto& inliers() const { return raw_points_; }

 private:
  template <typename T>
  bool almostEquals(const T val, const T correctVal,
                    const T epsilon = std::numeric_limits<T>::epsilon()) {
    const T maxXYOne =
        std::max({static_cast<T>(1.0d), std::fabs(val), std::fabs(correctVal)});
    return std::fabs(val - correctVal) <= epsilon * maxXYOne;
  }

 private:
  const PointCloudPtr& cloud_;
  LineCoefficients coeffs_;
  std::vector<Point2d> endpoints_;
  std::vector<Point2d> raw_points_;
};
}  // namespace line_fitting
