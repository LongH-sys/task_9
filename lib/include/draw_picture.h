#pragma once

#include <opencv2/core.hpp>

#include <vector>

void draw_trajectory(
    cv::Mat& frame, 
    const cv::Vec3d& measured, 
    const cv::Vec3d& predicted, 
    const cv::Vec3d& updated, 
    const cv::Mat& cameraMatix, 
    const cv::Mat& distCoeffs, 
    std::vector<cv::Point>& measuredHistory, 
    std::vector<cv::Point>& predictedHistory, 
    std::vector<cv::Point>& updatedHistory
);