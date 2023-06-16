cmake_minimum_required (VERSION 2.8.4)

project(lane_extraction)
set(CMAKE_BUILD_TYPE Release)

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
	src/hough_transform.cpp
	src/line_segment.cpp
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PCL_LIBRARIES}
	stdc++fs
	-lboost_system
	)

add_executable(hough src/main.cpp)
target_link_libraries(hough ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})


