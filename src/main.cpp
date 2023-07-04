#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <chrono>
#include <iostream>

#include "hough_transform.h"
#include "line_segment.h"

int main(int argc, char* argv[]) {
  const std::string pclFilePath = argv[1];
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(pclFilePath, *cloud) == -1) {
    std::cerr << "Failed to load pcl file" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Number of original cloud: " << cloud->points.size()
            << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  line_fitting::HoughTransform transformer(5, 500, 0.5);

  line_fitting::LineSegments line_segments;
  if (transformer.run(cloud, line_segments)) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time = duration.count();
    std::cout << "Line fitting time: " << time << " ms" << std::endl;
    std::cout << "" << std::endl;
    for (const auto& line : line_segments)
      std::cout << "line function: " << line.endpoints()[0].x() << ","
                << line.endpoints()[0].y() << "," << line.endpoints()[1].x()
                << "," << line.endpoints()[1].y() << std::endl;
  } else {
    std::cerr << "Failed to run hough transform" << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
