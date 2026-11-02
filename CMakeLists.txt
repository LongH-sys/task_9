cmake_minimum_required(VERSION 3.10)
project(kalman_task9)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
add_subdirectory(lib)
add_subdirectory(
    "/home/longh/vino_task/src/YoloVino"
    "${CMAKE_CURRENT_BINARY_DIR}/vino_build")
add_subdirectory(
    "../Hikrobot_task4/lib"
    "${CMAKE_CURRENT_BINARY_DIR}/camera_build"
)
add_subdirectory(app)