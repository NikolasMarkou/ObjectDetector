cmake_minimum_required(VERSION 2.8)
project( ObjectDetector )
find_package( OpenCV REQUIRED )
add_executable( ObjectDetector ObjectDetector.cpp )
target_link_libraries( ObjectDetector ${OpenCV_LIBS} )
