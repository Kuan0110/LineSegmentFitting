#pragma once 

#include <unordered_map>
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
public:
	static constexpr int THETA_BIN = (THETA_TOP-THETA_BOTTOM)*2;

	static std::vector<Vertex> createCircleLookUpTable();

	HoughTransform(int min_num_vote, int rho_num, double distance_thresh);

	~HoughTransform() {};

	bool run(const PointCloudPtr& cloud, LineSegments& line_segments);

private:
	void performHT(const PointCloudPtr& cloud, LineSegments& result);

	void votePoint(const Point& point, 
			const double delta_range, const double min_range, bool to_add);

	int getLine(LineCoefficients& coeffs, 
			const double min_range, const double delta_range,
			const std::vector<std::size_t>& accumulator_cell_indices) const;

	SegmentClusters seperateDistributedPoints(const PointCloudPtr& points, 
			const NeighborIndices& point_indices);

	double calcNorm(const Point& point) const;

	std::vector<Vertex> circle_; // [cos theta, sin theta]

	int rho_num_;
	int theta_num_;
	int peak_num_;
	int min_num_vote_;
	double distance_thresh_;

	std::vector<int> accumulator_;
};

}