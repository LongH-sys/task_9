#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include <yolo_vino.hpp>

bool pnp_solver(
    const cv::Mat& frame,
    cv::Vec3d& tvec,
    YoloVino::Yolov8poseVino& detector
);