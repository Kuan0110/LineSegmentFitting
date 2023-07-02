#pragma once 

#include <pcl/search/impl/kdtree.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>

#include "line_segment.h"

namespace line_fitting {
//theta range
#define THETA_BOTTOM -90
#define THETA_TOP 89
#define PI 3.14159265358979

using Vertex = std::array<double, 2>;
using LineSegments = std::vector<LineSegment2D>;
using ClusterIndices = std::vector<pcl::PointIndices>;
using NeighborIndices = std::vector<std::pair<int, double>>;
using SegmentClusters = std::vector<std::vector<int>>;

class HoughTransform {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
	static constexpr int THETA_BIN = (THETA_TOP-THETA_BOTTOM)*2;

	static std::vector<Vertex> createCircleLookUpTable();

	HoughTransform(int min_num_vote, int rho_num, double distance_thresh);

	~HoughTransform() {};

	bool run(const PointCloudPtr& cloud, LineSegments& line_segments);

 private:
	void performHT(const PointCloudPtr& cloud, LineSegments& result);

	void printVote(const double min_range, const double delta_range, 
		const std::vector<int>& accumulator_cell_indices);

	void intersectLineSegments(LineSegments& line_segments);

	void votePoint(const Point& point, 
			const double delta_range, const double min_range, bool to_add);

	int getLines(
		std::vector<LineCoefficients>& candi_hough_lines, 
		const double min_range, const double delta_range, 
		const std::vector<std::size_t>& accumulator_cell_indices);

	LineCoefficients findBestHoughLine(
		const PointCloudPtr& cloud,
		std::vector<LineCoefficients> hough_lines, 
		const std::vector<char>& ignore_indices);

	SegmentClusters seperateDistributedPoints(const PointCloudPtr& points, 
			const NeighborIndices& point_indices);

	Point2d getIntersection(
			const LineSegment2D& line1, const LineSegment2D& line2);

	double inline calcNorm(const Point& point) const {
		return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
	}

	double inline calcDist(const Point2d& p1, const Point2d& p2) const {
		return std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.y() - p2.y(), 2));
	}

	void inline removeVote(const int index) {accumulator_[index] = 0;}

 private:
	std::vector<Vertex> circle_; // [cos theta, sin theta]
	std::vector<int> accumulator_;
	std::vector<int> top_vote_indices_;

	int rho_num_;
	int theta_num_;
	int peak_num_;
	int min_num_vote_;
	double distance_thresh_;
};

}