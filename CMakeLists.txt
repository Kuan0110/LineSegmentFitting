cmake_minimum_required (VERSION 2.8.4)

project(lane_extraction)
set(CMAKE_BUILD_TYPE Release)

# set(third_party /home/kuan/Hozon/third_party)

# ### OpenCV 
# set(OPENCV_INCLUDE_DIRS ${third_party}/opencv/include/opencv4)
# set(OPENCV_LIBS_DIR ${third_party}/lib)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OPENCV_LIBS_DIR})

# ### Eigen
# set(EIGEN_INCLUDE_DIRS ${third_party}/eigen3/include)

# find_package(Eigen REQUIRED)
# include_directories(${EIGEN_INCLUDE_DIRS})

# # FLANN
# set(FlANN_INCLUDE_DIR /usr/local/include/flann)
# set(FlANN_LIBS_DIR /usr/local/lib)
# include_directories(${FlANN_INCLUDE_DIRs})
# link_directories(${FlANN_LIBS_DIR})

# ### PCL 
# set(PCL_INCLUDE_DIRS ${third_party}/pcl/include/pcl-1.11)
# set(PCL_LIBS_DIR ${third_party}/lib)

# find_package(PCL REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBS_DIR})

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(PCL  REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}
	${PCL_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
	# src/pointcloud.cpp
	# src/douglas_peucker.cpp
	# src/eigen_test.cpp
	src/hough_transform.cpp
	src/line_segment.cpp
	src/polyfit.cpp
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PCL_LIBRARIES}
	stdc++fs
	-lboost_system
	)

# add_executable(pointcloud src/pointcloud.cpp)
# target_link_libraries(pointcloud ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})

# add_executable(image src/image.cpp)
# target_link_libraries(image ${PROJECT_NAME} ${OpenCV_LIBS})

# add_executable(eigen src/eigen_test.cpp)
# target_link_libraries(eigen ${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(hough src/main.cpp)
target_link_libraries(hough ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})

# add_executable(poly src/polyfit.cpp)
# target_link_libraries(poly ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})

