#include "hough_transform.h"

namespace line_fitting {

std::vector<Vertex> HoughTransform::createCircleLookUpTable() {
	std::vector<double> theta_vec;
	theta_vec.resize(THETA_BIN);

	theta_vec[0] = THETA_BOTTOM;
	for(int i = 1; i < THETA_BIN; i++) {
		theta_vec[i] = (double)theta_vec[i - 1] + 0.5; 
	}

	std::vector<Vertex> table;
	table.resize(THETA_BIN);
	for(int i = 0; i < THETA_BIN; i++) {
		table[i][0] = cos((double)(theta_vec[i]) / 180.0 * PI);
		table[i][1] = sin((double)(theta_vec[i]) / 180.0 * PI);
	}

	return table;
}

const std::vector<Vertex> LOOK_UP_TABLE = HoughTransform::createCircleLookUpTable();

HoughTransform::HoughTransform(int min_num_vote, int rho_num, double distance_thresh) 
	: circle_(LOOK_UP_TABLE) 
	, min_num_vote_(min_num_vote)	
	, theta_num_(THETA_BIN)
	, rho_num_(rho_num == 0 ? THETA_BIN : rho_num)
	, distance_thresh_(distance_thresh) 
{
}

bool HoughTransform::run(const PointCloudPtr& cloud, LineSegments& line_segments) {
	// pcl::visualization::PCLVisualizer viewer("Region Growing Clustering");

	// point cloud pre-clustering
	ClusterIndices clusters;
	pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<Point> ec;
  ec.setClusterTolerance(0.15);
  ec.setMinClusterSize(100);
  ec.setMaxClusterSize(10000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);

  ec.extract(clusters);
	
	if (clusters.size() == 0) {
		std::cout << "no clusters found" << std::endl;
		return false;
	}

	std::cout << "Clusters size: " << clusters.size() << std::endl;

	// perform Hough Transform for each cluster
  for (size_t i = 0; i < clusters.size(); ++i) {
    PointCloudPtr cluster_cloud(new pcl::PointCloud<Point>);
    cluster_cloud->points.resize(clusters[i].indices.size());

    for (size_t j = 0; j < clusters[i].indices.size(); ++j) {
      int index = clusters[i].indices[j];
      cluster_cloud->points[j].x = cloud->points[index].x;
      cluster_cloud->points[j].y = cloud->points[index].y;
      cluster_cloud->points[j].z = 0.0;
    }

		// // Generate a random color for the cluster
    // int r = rand() % 256;
    // int g = rand() % 256;
    // int b = rand() % 256;
    // pcl::visualization::PointCloudColorHandlerCustom<Point> colorHandler(cluster_cloud, r, g, b);
    // std::cout << "cluster " << i << " size: " << clusters[i].indices.size() << std::endl;

    // // Add the cluster to the viewer
    // viewer.addPointCloud(cluster_cloud, colorHandler, "lane" + std::to_string(i));

		performHT(cluster_cloud, line_segments);
  }

	// // Start visualization
  // while (!viewer.wasStopped())
  //   viewer.spin();

	return true;
}

void HoughTransform::performHT(const PointCloudPtr& cloud, LineSegments& result) {
	// find bounding box of point cloud
	Point max_bound, min_bound;
	pcl::getMinMax3D(*cloud, min_bound, max_bound);
	double max_range = std::max<double>(calcNorm(min_bound), calcNorm(max_bound));
	double min_range = -max_range;
	double delta_range = 2 * max_range / rho_num_;
	accumulator_.resize(theta_num_ * rho_num_, 0);	

	// calculate the vote frequency of every cell in hough space
	for (const auto& point : cloud->points) {
		votePoint(point, delta_range, min_range, true);
	}

	// only consider cells whose number of votes surpass threshold
	std::vector<std::size_t> accumulator_cell_indices;
	for (std::size_t i = 0; i < accumulator_.size(); ++i) {
		if (accumulator_[i] >= min_num_vote_) {
			accumulator_cell_indices.emplace_back(i);
		}
	}

	LineCoefficients coeffs;
	std::size_t remain_points = cloud->size();
	std::vector<char> ignore_indices(cloud->size(), false);
	while (remain_points >= min_num_vote_) {
		// get candidate line has top number of votes 
		std::size_t num_votes = getLine(
			coeffs, min_range, delta_range, accumulator_cell_indices);
		if (num_votes < min_num_vote_) {
			std::cout << "num_votes: " << num_votes << std::endl;
			break;
		}

		// refine line with points closed to line
		LineSegment2D candi_segment(cloud, coeffs, 
			{max_bound.x, max_bound.y, min_bound.x, min_bound.y});
		if (!candi_segment.refine(distance_thresh_, ignore_indices)) {
			std::cout << "refine failed" << std::endl;
			break;
		}

		// remove points around current fitted line
		NeighborIndices points_to_remove = 
			LineSegment2D::getPointIndicesCloseToLine(
				cloud, candi_segment.coeffs(), distance_thresh_, ignore_indices);

		// seperate line segment has distributed points
		SegmentClusters cluster_to_remove =
			seperateDistributedPoints(cloud, points_to_remove);

		if (cluster_to_remove.size() == 1) {
			LineSegment2D line_segment(candi_segment);

			for (int j = 0; j < cluster_to_remove[0].size(); ++j) {
				const auto& point_idx = points_to_remove[cluster_to_remove[0][j]];
				const auto& curPoint = cloud->points[point_idx.first];
				
				Eigen::Vector2d point;
				point << curPoint.x, curPoint.y;

				// clip line segment from line
				if (point_idx.second < 0.2) {
					line_segment.clipLineSegment(point);
				}

				// discard current vote cell
				votePoint(curPoint, delta_range, min_range, false);
				ignore_indices[point_idx.first] = true;
			}

			if (line_segment.getSegmentLength() > 1.0) 
				result.emplace_back(std::move(line_segment));
			
			remain_points -= cluster_to_remove[0].size();

		} else if (cluster_to_remove.size() > 1) {
			for (int i = 0; i < cluster_to_remove.size(); i++) {
				int num_points = cluster_to_remove[i].size();

				// re-fit line for each distributed cluster 
				Eigen::MatrixXd point_matrix = 
					Eigen::MatrixXd::Constant(num_points, 2, 0);

				int count = 0;
				int removed_count = 0;
				for (auto index : cluster_to_remove[i]) {
					const auto& cur_point = cloud->points[points_to_remove[index].first];
					point_matrix.row(count++) << cur_point.x, cur_point.y;
					// std::cout << "cluster point: " << cur_point.x << "," << cur_point.y << std::endl;
				}

				LineSegment2D line_segment(candi_segment);
				if (!line_segment.PCALineFit(point_matrix)) 
					continue;

				std::cout << "cluster refined line: " << line_segment.coeffs()[0] << " " << line_segment.coeffs()[1] << " " << line_segment.coeffs()[2] << std::endl;

				for (int j = 0; j < num_points; ++j) {
					const auto& point_idx = points_to_remove[cluster_to_remove[i][j]];
					const auto& curPoint = cloud->points[point_idx.first];
					
					Eigen::Vector2d point;
					point << curPoint.x, curPoint.y;

					double new_distance = 
						line_segment.getDistancePoint2Line(point, line_segment.coeffs());

					// clip line segment from line
					if (new_distance < 0.3) {
						line_segment.clipLineSegment(point);

						// discard current vote cell
						votePoint(curPoint, delta_range, min_range, false);
						ignore_indices[point_idx.first] = true;
						removed_count++;
					}
				}

				if (line_segment.getSegmentLength() > 1.0) 
					result.emplace_back(std::move(line_segment));
				
				remain_points -= removed_count;
			}
		}

		std::cout << "remaining points: " << remain_points << std::endl;
	}

	return;
}

void HoughTransform::votePoint(const Point& point, const double delta_range, const double min_range, bool to_add) {
	// vote the range frequency of each point in all theta
	for(int k = 0; k < theta_num_; k++) {
		double rho_cal = point.x * circle_[k][0] + point.y * circle_[k][1];
		int index = std::round((rho_cal - min_range) / delta_range) * theta_num_ + k;

		if (index < accumulator_.size()) {
			to_add ? accumulator_[index]++ : accumulator_[index]--;
		}
	}

	return;
}

int HoughTransform::getLine(LineCoefficients& coeffs, 
		const double min_range, const double delta_range, 
		const std::vector<std::size_t>& accumulator_cell_indices) const {
	if (accumulator_cell_indices.empty()) {
		return -1;
	}

	// get the top voted line
	auto max_indices = std::max_element(
		accumulator_cell_indices.begin(), accumulator_cell_indices.end(),
		[this](const std::size_t idx1, const std::size_t idx2) { 
			return accumulator_[idx1] < accumulator_[idx2]; 
		}
	);

	// convert line coefficient from hough space to cartesian space
	std::size_t index = *max_indices;
	const int num_vote = accumulator_[index];

	int y = index / theta_num_;
	double range = min_range + y * delta_range; 
	// std::cout << "range: " << range << std::endl;

	int x = index % theta_num_;
	double theta = ((double)x / 2 + THETA_BOTTOM) / 180.0 * PI;
	// std::cout << "theta: " << theta << std::endl;

  if (std::abs(theta) < 0.02) {
		coeffs[0] = 1;
		coeffs[1] = 0;
		coeffs[2] = -range;
	} else {
		coeffs[0] = cos(theta) / sin(theta);
		coeffs[1] = 1;
		coeffs[2] = - range / sin(theta);
	}

	// std::cout << "canditate line: " << coeffs[0] << " " << coeffs[1] << " " << coeffs[2] << ", num vote: " << num_vote << std::endl;
	// std::cout << "" << std::endl;

	return num_vote;
}

SegmentClusters HoughTransform::seperateDistributedPoints(
		const PointCloudPtr& cloud, const NeighborIndices& point_indices) {
	PointCloudPtr candi_cloud(new pcl::PointCloud<Point>);
	candi_cloud->points.reserve(point_indices.size());

	// std::unordered_map<int, int> index_map;
	for (int i = 0; i < point_indices.size(); i++) {
		candi_cloud->points.push_back(cloud->points[point_indices[i].first]);
		// index_map[i] = point_indices[i].first;
	}
	std::cout << "candidates: " << candi_cloud->points.size() << std::endl;
	pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  tree->setInputCloud(candi_cloud);

  pcl::EuclideanClusterExtraction<Point> ec;
  ec.setClusterTolerance(1);
  ec.setMinClusterSize(5);
  ec.setMaxClusterSize(10000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(candi_cloud);

	ClusterIndices clusters;
  ec.extract(clusters);
	
	
	SegmentClusters point_clusters;
	point_clusters.reserve(clusters.size());
	for (int i = 0; i < clusters.size(); i++) {
		std::vector<int> points;
		points.reserve(clusters[i].indices.size());
		for (int j = 0; j < clusters[i].indices.size(); j++) {
      points.push_back(clusters[i].indices[j]);
    }
		point_clusters.push_back(points);
  }

	std::cout << "line segment cluster number: " << point_clusters.size() << std::endl;
	return std::move(point_clusters);
}

double HoughTransform::calcNorm(const Point& point) const {
  return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

}