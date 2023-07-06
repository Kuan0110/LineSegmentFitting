#include "line_segment.h"

namespace line_fitting {

LineSegment2D::LineSegment2D(const PointCloudPtr& cloud,
                             const std::vector<Point2d>& range)
    : cloud_(cloud),
      coeffs_({0.0, 0.0, 0.0}),
      endpoints_(range),
      raw_points_() {}

LineSegment2D::LineSegment2D(const PointCloudPtr& cloud,
                             const LineCoefficients& coeffs,
                             const std::vector<Point2d>& range)
    : cloud_(cloud), coeffs_(coeffs), raw_points_(), endpoints_(range) {}

LineSegment2D::LineSegment2D(const LineSegment2D& line_segment)
    : cloud_(line_segment.cloud_),
      coeffs_(line_segment.coeffs_),
      raw_points_(line_segment.raw_points_),
      endpoints_(line_segment.endpoints_) {}

std::vector<std::pair<int, double>> LineSegment2D::getPointIndicesCloseToLine(
    const PointCloudPtr& cloud, const LineCoefficients& coeffs,
    const double distance_thresh,
    const std::vector<char>& ignore_point_indices) {
  if (cloud->empty()) {
    return {};
  }

  Point2d eigen_point;
  std::vector<std::pair<int, double>> point_indices;

  for (std::size_t i = 0; i < cloud->size(); ++i) {
    if (ignore_point_indices[i]) {
      continue;
    }

    const auto& curPoint = cloud->points[i];
    eigen_point << curPoint.x, curPoint.y;
    double distance = getDistancePoint2Line(eigen_point, coeffs);

    if (distance < distance_thresh) {
      point_indices.emplace_back(std::make_pair(i, distance));
    }
  }

  return std::move(point_indices);
}

double LineSegment2D::getDistancePoint2Line(const Point2d& point,
                                            const LineCoefficients& coeffs) {
  if (coeffs[1] == 0) return std::abs(point[0] + coeffs[2]);

  Point2d hypot;
  hypot << 0.0, (-point[0] * coeffs[0] - coeffs[2]) / coeffs[1] - point[1];

  Point2d norm;
  norm << coeffs[0], coeffs[1];

  return std::abs(hypot.dot(norm));
}

bool LineSegment2D::refine(const double distance_Thresh,
                           const std::vector<char>& ignore_indices) {
  const auto point_indices = LineSegment2D::getPointIndicesCloseToLine(
      cloud_, coeffs_, distance_Thresh, ignore_indices);

  if (point_indices.size() < 15) {
    std::cout << "point closed to line not enough" << std::endl;
    return false;
  }

  // analyze principal components of inlier point cloud
  Eigen::MatrixXd point_matrix =
      Eigen::MatrixXd::Constant(point_indices.size(), 2, 0);

  int count = 0;
  for (const auto& index : point_indices) {
    const auto& cur_point = cloud_->points[index.first];
    point_matrix.row(count++) << cur_point.x, cur_point.y;
  }

  fitLineTLS(point_matrix);

  std::cout << "refined line: " << coeffs_[0] << "," << coeffs_[1] << ","
            << coeffs_[2] << std::endl;
  std::cout << "" << std::endl;

  return true;
}

bool LineSegment2D::PCALineFit(const Eigen::MatrixXd& point_matrix) {
  // normalize point cloud
  auto centroid = point_matrix.colwise().mean();
  Eigen::MatrixXd centered_matrix = point_matrix.rowwise() - centroid;
  Eigen::MatrixXd scatter_matrix =
      (centered_matrix.adjoint() * centered_matrix);

  // second eigen value cant be zero
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(scatter_matrix);
  if (almostEquals<double>(eig.eigenvalues()(1), 0)) {
    return false;
  }

  Eigen::MatrixXd eig_vecs = eig.eigenvectors();
  coeffs_[0] = eig_vecs(1, 1);
  coeffs_[1] = -eig_vecs(0, 1);
  coeffs_[2] = eig_vecs(0, 1) * centroid.y() - eig_vecs(1, 1) * centroid.x();

  return true;
}

bool LineSegment2D::fitLineTLS(const Eigen::MatrixXd& point_matrix) {
  // normalize point cloud
  auto centroid = point_matrix.colwise().mean();
  Eigen::MatrixXd centered_matrix = point_matrix.rowwise() - centroid;
  Eigen::MatrixXd scatter_matrix =
      (centered_matrix.transpose() * centered_matrix);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      scatter_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd V = svd.matrixV();

  Point2d lastColumn = V.col(V.cols() - 1);

  coeffs_[0] = lastColumn[0];
  coeffs_[1] = lastColumn[1];
  coeffs_[2] = -(lastColumn[1] * centroid.y() + lastColumn[0] * centroid.x());

  return true;
}

void LineSegment2D::clipLineSegment(const Point2d& point) {
  if (coeffs_[1] != 0.0) {
    double proj_y = (point[0] * coeffs_[0] + coeffs_[2]) / -coeffs_[1];

    Point2d u, v;
    v << -coeffs_[1], coeffs_[0];
    u << 0, point[1] - proj_y;
    double scale = u.dot(v) / v.dot(v);

    Point2d proj_point;
    proj_point << point[0] + scale * v[0], proj_y + scale * v[1];

    if (proj_point[0] < endpoints_[0].x()) {
      endpoints_[0].x() = proj_point[0];
      endpoints_[0].y() = proj_point[1];
    }
    if (proj_point[0] > endpoints_[1].x()) {
      endpoints_[1].x() = proj_point[0];
      endpoints_[1].y() = proj_point[1];
    }
  } else {
    if (point[1] < endpoints_[0].y()) {
      endpoints_[0].x() = -coeffs_[2] / coeffs_[0];
      endpoints_[0].y() = point[1];
    }
    if (point[1] > endpoints_[1].y()) {
      endpoints_[1].x() = -coeffs_[2] / coeffs_[0];
      endpoints_[1].y() = point[1];
    }
  }

  return;
}

Point2d LineSegment2D::getProjection(const Point2d& point) const {
  Point2d proj_point;
  if (coeffs_[1] != 0.0) {
    double proj_y = (point[0] * coeffs_[0] + coeffs_[2]) / -coeffs_[1];

    Point2d u, v;
    v << -coeffs_[1], coeffs_[0];
    u << 0, point[1] - proj_y;
    double scale = u.dot(v) / v.dot(v);

    proj_point << point[0] + scale * v[0], proj_y + scale * v[1];
  } else {
    proj_point << -coeffs_[2] / coeffs_[0], point[1];
  }

  return proj_point;
}

bool LineSegment2D::isOnLineSegment(const Point2d& point) const {
  double x_min = endpoints_[0].x();
  double x_max = endpoints_[1].x();

  if (point.x() >= x_min && point.x() <= x_max) return true;

  return false;
}

}  // namespace line_fitting
