cmake_minimum_required(VERSION 3.10)

project(study_opencv)

set(OpenCV_DIR C:/opencv/opencv/build)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS})