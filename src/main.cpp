#include <iostream>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "line_segment.h"
#include "hough_transform.h"

int main(int argc, char* argv[]) {
  const std::string pclFilePath = argv[1];
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(pclFilePath, *cloud) == -1) {
    std::cerr << "Failed to load pcl file" << std::endl;
    return EXIT_FAILURE;
  }

  // 创建体素化滤波器对象
  pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
  voxelGrid.setInputCloud(cloud);
  voxelGrid.setLeafSize(0.1f, 0.1f, 0.1f); // 设置体素的尺寸

  // 执行体素化滤波
  pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
  voxelGrid.filter(*filteredCloud);

  pcl::io::savePCDFile("merged_cloud6.pcd", *filteredCloud);

  std::cout << "Number of original cloud: " << filteredCloud->points.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  line_fitting::HoughTransform transformer(5,800,0.5);

  line_fitting::LineSegments line_segments;
  if (transformer.run(cloud, line_segments)) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time = duration.count();
    std::cout << "Line fitting time: " << time << " ms" << std::endl;
    std::cout << "" << std::endl;
    for (const auto& line : line_segments)
      std::cout << "line function: " 
                << line.endpoints()[0] << "," << line.endpoints()[1] << "," 
                << line.endpoints()[2] << "," << line.endpoints()[3] << std::endl;
  } else {
    std::cerr << "Failed to run hough transform" << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}


