#include <opencv2/calib3d.hpp>

#include "pnp_solver.h"

bool pnp_solver(
    const cv::Mat& frame,
    cv::Vec3d& tvec,
    YoloVino::Yolov8poseVino& detector
)
{
    std::vector<cv::Point2f> corners;
    if (frame.empty())
    {
        return false;
    }
    auto results = detector.safe_predict(
        frame, 
        cv::Rect(0, 0, frame.cols, frame.rows)
    );
    if (results.empty())
    {
        return false;
    }
    const auto& armor = results[0];
    if (armor.keypoints.size() != 4)
    {
        return false;
    }
    for (int j = 0; j < 4; j++)
    {
        
        auto& point = armor.keypoints[j];
        corners.push_back(cv::Point2f(point.x, point.y));
    }
    float width = 0.135f;
    float height = 0.055f;
    std::vector<cv::Point3f> objectPoints =                   
    {
        {-width / 2.0f, -height / 2.0f, 0.0f},
        {-width / 2.0f, height / 2.0f, 0.0f},
        {width / 2.0f, -height / 2.0f, 0.0f},
        {width / 2.0f, height / 2.0f, 0.0f}
    };                    
    double fx = 4458.8549687633;
    double fy = 4457.0808035083;
    double cx = 1989.8487562489;
    double cy = 1515.7365238851;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
        fx, 0, cx,
        0, fy, cy,
        0,  0,  1
    );
    double k1 = -0.0803149495;
    double k2 = 0.3602814150;
    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) <<
        k1, k2, 0, 0, 0        
    );
    cv::Vec3d rvec;
    bool success = cv::solvePnP(
        objectPoints,        
        corners,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );
    return success;
}