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
  ec.setMinClusterSize(70);
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

		intersectLineSegments(line_segments);

		std::cout << "cluster ends" << std::endl;
  }

	// // Start visualization
  // while (!viewer.wasStopped())
  //   viewer.spin();

	return true;
}

void HoughTransform::printVote(const double min_range, const double delta_range, 
		const std::vector<int>& accumulator_cell_indices) {
  LineCoefficients coeffs;
	auto compare = [&accumulator_cell_indices](int i, int j) {
		return accumulator_cell_indices[i] > accumulator_cell_indices[j];
	};

	std::vector<int> indices(accumulator_cell_indices.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), compare);

	for (int i = 0; i < 10; ++i) {
		// convert line coefficient from hough space to cartesian space
		int cur_vote_index = indices[i];
		std::cout << "num vote: " << accumulator_cell_indices[cur_vote_index] << std::endl;

		int y = cur_vote_index / theta_num_;
		double range = min_range + y * delta_range; 

		int x = cur_vote_index % theta_num_;
		double theta = ((double)x / 2 + THETA_BOTTOM) / 180.0 * PI;

		if (std::abs(theta) < 0.02) {
			coeffs[0] = 1;
			coeffs[1] = 0;
			coeffs[2] = -range;
		} else {
			coeffs[0] = cos(theta) / sin(theta);
			coeffs[1] = 1;
			coeffs[2] = - range / sin(theta);
		}

		std::cout << "print line: " << coeffs[0] << "," << coeffs[1] << "," << coeffs[2] << std::endl;
		std::cout << "" << std::endl;
	}
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

	// printVote(min_range, delta_range, accumulator_);

	std::size_t remain_points = cloud->size();
	std::vector<char> ignore_indices(cloud->size(), false);
	while (remain_points >= min_num_vote_) {
		top_vote_indices_.clear();
		// get candidate line has top number of votes 
		std::vector<LineCoefficients> candi_coeffs;
		std::size_t num_votes = getLines(candi_coeffs, min_range, 
				delta_range, accumulator_cell_indices);

		if (num_votes < min_num_vote_) {
			std::cout << "num_votes: " << num_votes << std::endl;
			break;
		}

		LineCoefficients coeffs = findBestHoughLine(
				cloud, candi_coeffs, ignore_indices);

		// refine line with points closed to line
		LineSegment2D candi_segment(cloud, coeffs, 
			{Point2d(max_bound.x, max_bound.y), 
			 Point2d(min_bound.x, min_bound.y)});
		if (!candi_segment.refine(distance_thresh_, ignore_indices)) {
			std::cout << "refine failed" << std::endl;
			for (const int idx : top_vote_indices_) 
				removeVote(idx);
			continue;
		}

		// remove points around current fitted line
		NeighborIndices points_to_remove = 
			LineSegment2D::getPointIndicesCloseToLine(
				cloud, candi_segment.coeffs(), distance_thresh_, ignore_indices);

		// seperate line segment has distributed points
		SegmentClusters cluster_to_remove =
			seperateDistributedPoints(cloud, points_to_remove);

		std::cout << "print line: " << coeffs[0] << "," << coeffs[1] << "," << coeffs[2] << std::endl;

		if (cluster_to_remove.size() == 0) {
			for (const auto& point_idx : points_to_remove) {
				const auto& cur_point = cloud->points[point_idx.first];
				
				Point2d point;
				point << cur_point.x, cur_point.y;
				std::cout << "0 closed points: " << cur_point.x << "," << cur_point.y << std::endl;

				// discard current vote cell
				// votePoint(cur_point, delta_range, min_range, false);
				for (const int idx : top_vote_indices_) 
					removeVote(idx);
			}			
		} else if (cluster_to_remove.size() == 1) {
			LineSegment2D line_segment(candi_segment);

			for (const auto& point_idx : points_to_remove) {
				const auto& cur_point = cloud->points[point_idx.first];
				
				Point2d point;
				point << cur_point.x, cur_point.y;
				// clip line segment from line
				if (point_idx.second < 0.3) {
					line_segment.addInliers({point.x(), point.y()});
					line_segment.clipLineSegment(point);
				}

				// discard current vote cell
				votePoint(cur_point, delta_range, min_range, false);
				ignore_indices[point_idx.first] = true;
				std::cout << "1 closed points: " << cur_point.x << "," << cur_point.y << std::endl;
			}
			if (line_segment.getSegmentLength() > 1) {
				std::cout << "find line endpoint: " << line_segment.endpoints()[0].x() << "," << line_segment.endpoints()[0].y() << "," 
							<< line_segment.endpoints()[1].x() << "," << line_segment.endpoints()[1].y() << std::endl;
				result.emplace_back(std::move(line_segment));
			}
			
			remain_points -= points_to_remove.size();

		} else if (cluster_to_remove.size() > 1) {
			for (int i = 0; i < cluster_to_remove.size(); i++) {
				int num_points = cluster_to_remove[i].size();

				if (num_points < 8) {
					for (auto index : cluster_to_remove[i]) {
						if (points_to_remove[index].second < 0.2) {
							const auto& cur_point = cloud->points[points_to_remove[index].first];
							votePoint(cur_point, delta_range, min_range, false);
						}				
					}
					continue;
				}

				// re-fit line for each distributed cluster 
				Eigen::MatrixXd point_matrix = 
					Eigen::MatrixXd::Constant(num_points, 2, 0);

				int count = 0;
				int removed_count = 0;
				for (auto index : cluster_to_remove[i]) {
					if (points_to_remove[index].second < 0.2) {
						const auto& cur_point = cloud->points[points_to_remove[index].first];
						point_matrix(count, 0) = cur_point.x;
						point_matrix(count++, 1) = cur_point.y;
					}
				}
				if (count == 0) continue;
				point_matrix.conservativeResize(count, 2);
				// for (int i = 0; i < point_matrix.rows(); ++i) {
				// 	std::cout << "matrix point: " << point_matrix(i,0) << "," << point_matrix(i,1) << std::endl;
				// }
				LineSegment2D line_segment(candi_segment);
				if (!line_segment.fitLineTLS(point_matrix)) 
					continue;

				std::cout << "cluster refined line: " << line_segment.coeffs()[0] << "," << line_segment.coeffs()[1] << "," << line_segment.coeffs()[2] << std::endl;

				if (candi_segment.coeffs()[1] != 0 && line_segment.coeffs()[1] != 0) {
					double angle_candi = std::atan2(-candi_segment.coeffs()[0], candi_segment.coeffs()[1]);
					double angle_new = std::atan2(-line_segment.coeffs()[0], line_segment.coeffs()[1]);
					std::cout << "angle diff: " << std::abs(angle_candi-angle_new) / PI * 180.0 << std::endl;
					if (std::abs(angle_candi-angle_new) > 0.03492) {
						// for (auto index : cluster_to_remove[i]) {
						// 	if (points_to_remove[index].second < 0.2) {
						// 		const auto& cur_point = cloud->points[points_to_remove[index].first];
						// 		votePoint(cur_point, delta_range, min_range, false);
						// 		std::cout << "delete points: " << cur_point.x << "," << cur_point.y << std::endl;
						// 	}					
						// }
						for (const int idx : top_vote_indices_) 
							removeVote(idx);
						continue;
					}
				} else if (candi_segment.coeffs()[1] == 0) {
					double angle_new = std::atan2(-line_segment.coeffs()[0], line_segment.coeffs()[1]);
					std::cout << "angle diff: " << std::abs(PI/2-angle_new) / PI * 180.0 << std::endl;
					if (std::abs(PI/2-angle_new) > 0.03492) {
						// for (auto index : cluster_to_remove[i]) {
						// 	if (points_to_remove[index].second < 0.2) {
						// 		const auto& cur_point = cloud->points[points_to_remove[index].first];
						// 		votePoint(cur_point, delta_range, min_range, false);
						// 		std::cout << "delete points: " << cur_point.x << "," << cur_point.y << std::endl;
						// 	}					
						// }
						for (const int idx : top_vote_indices_) 
							removeVote(idx);
						continue;
					}
				} else if (line_segment.coeffs()[1] == 0) {
					double angle_candi = std::atan2(-candi_segment.coeffs()[0], candi_segment.coeffs()[1]);
					std::cout << "angle diff: " << std::abs(PI/2-angle_candi) / PI * 180.0 << std::endl;
					if (std::abs(PI/2-angle_candi) > 0.03492) {
						// for (auto index : cluster_to_remove[i]) {
						// 	if (points_to_remove[index].second < 0.2) {
						// 		const auto& cur_point = cloud->points[points_to_remove[index].first];
						// 		votePoint(cur_point, delta_range, min_range, false);
						// 		std::cout << "delete points: " << cur_point.x << "," << cur_point.y << std::endl;
						// 	}				
						// }
						for (const int idx : top_vote_indices_) 
							removeVote(idx);
						continue;
					}				
				}

				for (int j = 0; j < num_points; ++j) {
					const auto& point_idx = points_to_remove[cluster_to_remove[i][j]];
					const auto& cur_point = cloud->points[point_idx.first];
					
					Point2d point;
					point << cur_point.x, cur_point.y;

					double new_distance = 
						line_segment.getDistancePoint2Line(point, line_segment.coeffs());

					// clip line segment from line
					if (new_distance < 0.3) {
						line_segment.clipLineSegment(point);
						// std::cout << "end point: " << curPoint.x << "," << curPoint.y << std::endl;
					}

						// discard current vote cell
						votePoint(cur_point, delta_range, min_range, false);
						ignore_indices[point_idx.first] = true;
						std::cout << "2 closed points: " << cur_point.x << "," << cur_point.y << std::endl;
						removed_count++;
					// }
				}

				if (line_segment.getSegmentLength() > 1.0) {
					std::cout << "find line endpoint: " << line_segment.endpoints()[0].x() << "," << line_segment.endpoints()[0].y() << "," 
                << line_segment.endpoints()[1].x() << "," << line_segment.endpoints()[1].y() << std::endl;
					result.emplace_back(std::move(line_segment));
				}
				
				remain_points -= removed_count;
			}
		}
		std::cout << "remained " << remain_points << " points" << std::endl;
		std::cout << "end\n" << std::endl;
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

int HoughTransform::getLines(
		std::vector<LineCoefficients>& candi_hough_lines, 
		const double min_range, const double delta_range, 
		const std::vector<std::size_t>& accumulator_cell_indices) {
	if (accumulator_cell_indices.empty()) {
		return -1;
	}

	auto compare = [&b=accumulator_cell_indices,
			&a=accumulator_] (int i, int j) {return a[b[i]] > a[b[j]];};

	std::vector<int> indices(accumulator_cell_indices.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), compare);
	int num_top_vote = accumulator_[accumulator_cell_indices[indices[0]]];

	for (const int idx : indices) {
		int cell_idx = accumulator_cell_indices[idx];
		if (accumulator_[cell_idx] < (num_top_vote-10) ||
				accumulator_[cell_idx] < min_num_vote_)
			break;

		// convert line coefficient from hough space to cartesian space
		int y = cell_idx / theta_num_;
		double range = min_range + y * delta_range; 

		int x = cell_idx % theta_num_;
		double theta = ((double)x / 2 + THETA_BOTTOM) / 180.0 * PI;

		LineCoefficients candi_coeffs;
		if (std::abs(theta) < 0.02) {
			candi_coeffs[0] = 1;
			candi_coeffs[1] = 0;
			candi_coeffs[2] = -range;
		} else {
			candi_coeffs[0] = cos(theta) / sin(theta);
			candi_coeffs[1] = 1;
			candi_coeffs[2] = - range / sin(theta);
		}

		candi_hough_lines.push_back(std::move(candi_coeffs));
		top_vote_indices_.push_back(cell_idx);
	}

	return num_top_vote;
}

LineCoefficients HoughTransform::findBestHoughLine(const PointCloudPtr& cloud,
		std::vector<LineCoefficients> hough_lines, const std::vector<char>& ignore_indices) {
	std::vector<size_t> num_inliers;
	for (const auto& coeffs : hough_lines) {
		const auto point_indices = LineSegment2D::getPointIndicesCloseToLine(
				cloud, coeffs, distance_thresh_, ignore_indices);
		num_inliers.push_back(point_indices.size());
	}

	auto iter = std::max_element(num_inliers.begin(), num_inliers.end());
	int idx = iter - num_inliers.begin();

	return hough_lines[idx];
}

SegmentClusters HoughTransform::seperateDistributedPoints(
		const PointCloudPtr& cloud, const NeighborIndices& point_indices) {
	PointCloudPtr candi_cloud(new pcl::PointCloud<Point>);
	candi_cloud->points.reserve(point_indices.size());

	for (int i = 0; i < point_indices.size(); i++) {
		candi_cloud->points.push_back(cloud->points[point_indices[i].first]);
	}
	// std::cout << "candidates: " << candi_cloud->points.size() << std::endl;
	pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  tree->setInputCloud(candi_cloud);

  pcl::EuclideanClusterExtraction<Point> ec;
  ec.setClusterTolerance(0.5);
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

void HoughTransform::addLineSegment(const LineSegment2D& new_line, LineSegments& lines) {
	// for (auto& line : lines) {
	// 	Point2d point1, point2;
	// 	point1 << new_line.endpoints()[0], new_line.endpoints()[1];
	// 	point2 << new_line.endpoints()[2], new_line.endpoints()[3];

	// 	double distance1 = getDistancePoint2Line(point1, line.coeffs());
	// 	double distance2 = getDistancePoint2Line(point2, line.coeffs());

	// 	if (distance1 < 0.8 && distance2 < 0.8 && std::abs(distance1 - distance2) < 0.3) {
	// 		Eigen::MatrixXd point_matrix(new_line.inliers().size() + line.inliers().size(), 2);

	// 		for ()
	// 	}
	// }
}

void HoughTransform::intersectLineSegments(LineSegments& line_segments) {
	int num_endpoint = 2*line_segments.size();
	Eigen::MatrixXd dis_matrix = 
		Eigen::MatrixXd::Constant(num_endpoint, num_endpoint, 100.0);

	for (size_t j = 0; j < line_segments.size(); j++) {
		std::vector<Point2d> endpoints = line_segments[j].endpoints();
		for (size_t k = j + 1; k < line_segments.size(); k++) {
			std::vector<Point2d> candi_endpoints = line_segments[k].endpoints();
			double dist1 = calcDist(candi_endpoints[0], endpoints[0]);
			double dist2 = calcDist(candi_endpoints[1], endpoints[0]);
			dis_matrix(2*j,2*k) = dist1;
			dis_matrix(2*j,2*k+1) = dist2;

			double dist3 = calcDist(candi_endpoints[0], endpoints[1]);
			double dist4 = calcDist(candi_endpoints[1], endpoints[1]);
			dis_matrix(2*j+1,2*k) = dist3;
			dis_matrix(2*j+1,2*k+1) = dist4;			
		}
	}

	// std::cout << "dist matrix:\n" << dis_matrix << std::endl;
	std::vector<double> distances;
	std::vector<std::pair<int, int>> closed_pairs;
	for (int i = 0; i < dis_matrix.rows(); ++i) {
		Eigen::VectorXd::Index min_row;
		double min_val = dis_matrix.row(i).minCoeff(&min_row);
		if (min_val > 2.0) continue;
		closed_pairs.push_back(std::make_pair(i, min_row));
		distances.push_back(min_val);
	}

	for (int i = 0; i < closed_pairs.size(); ++i) {
		// std::cout << "pair index: " << closed_pairs[i].first << "," << closed_pairs[i].second << std::endl;
		Point2d cross_point = getIntersection(
			line_segments[closed_pairs[i].first/2], line_segments[closed_pairs[i].second/2]);
		// std::cout << "cross point: " << cross_point.x() << "," << cross_point.y() << std::endl;
		
		Point2d p1 = line_segments[closed_pairs[i].first/2].endpoints()[closed_pairs[i].first%2];
		Point2d p2 = line_segments[closed_pairs[i].second/2].endpoints()[closed_pairs[i].second%2];
		double d1 = calcDist(cross_point, p1);
		double d2 = calcDist(cross_point, p2);
		// std::cout << "distance: " << d1 << "," << d2 << std::endl;
		if (d1 < 2.0 && d2 < 2.0) {
			if (distances[i] < 1.0) {
				line_segments[closed_pairs[i].first/2].setEndpoints(closed_pairs[i].first%2, cross_point);
				line_segments[closed_pairs[i].second/2].setEndpoints(closed_pairs[i].second%2, cross_point);
			} else {
				PointCloudPtr cloud(new pcl::PointCloud<Point>);
				std::vector<Point2d> range = {p1, p2};
				LineSegment2D new_line_segment(cloud, range);
				line_segments.push_back(std::move(new_line_segment));
			}
		} else {
			Point2d mid = (p1 + p2) / 2.0;
			line_segments[closed_pairs[i].first/2].setEndpoints(closed_pairs[i].first%2, mid);
			line_segments[closed_pairs[i].second/2].setEndpoints(closed_pairs[i].second%2, mid);
		}
	}

	return;
}

Point2d HoughTransform::getIntersection(
		const LineSegment2D& line1, const LineSegment2D& line2) {
	if (line1.coeffs()[1] == 0) {
		double x = - line1.coeffs()[2] / line1.coeffs()[0];
		double y = - (line2.coeffs()[2] + line2.coeffs()[0]*x) / line2.coeffs()[1];
		return Point2d(x, y);
	} else if (line2.coeffs()[1] == 0) {
		double x = - line2.coeffs()[2] / line2.coeffs()[0];
		double y = - (line1.coeffs()[2] + line1.coeffs()[0]*x) / line1.coeffs()[1];
		return Point2d(x, y);
	} else {
		double a0 = line1.coeffs()[0];
		double b0 = line1.coeffs()[1];
		double c0 = line1.coeffs()[2];
		double a1 = line2.coeffs()[0];
		double b1 = line2.coeffs()[1];
		double c1 = line2.coeffs()[2];
		double D = a0*b1 - a1*b0;
		double x = (b0*c1 - b1*c0)/D;
		double y = (a1*c0 - a0*c1)/D;
		return Point2d(x, y);
	}
}

}