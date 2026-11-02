#include "draw_picture.h"

#include <opencv2/calib3d.hpp>

#include <opencv2/imgproc.hpp>

void draw_trajectory(
    cv::Mat& canvas, 
    const cv::Vec3d& measured, 
    const cv::Vec3d& predicted, 
    const cv::Vec3d& updated, 
    const cv::Mat& cameraMatrix, 
    const cv::Mat& distCoeffs, 
    std::vector<cv::Point>& measuredHistory, 
    std::vector<cv::Point>& predictedHistory, 
    std::vector<cv::Point>& updatedHistory
)
{
    if (canvas.empty())
    {
        return;
    }

    std::vector<cv::Point3d> points3D = {
        cv::Point3d(measured[0], measured[1], measured[2]),
        cv::Point3d(predicted[0], predicted[1], predicted[2]),
        cv::Point3d(updated[0], updated[1], updated[2])
    };
    std::vector<cv::Point2d> points2D;
    
    cv::projectPoints(
        points3D,
        cv::Vec3d(0, 0, 0),
        cv::Vec3d(0, 0, 0),
        cameraMatrix,
        distCoeffs,
        points2D
    );
    
    measuredHistory.emplace_back(
        cvRound(points2D[0].x),
        cvRound(points2D[0].y)
    );
    predictedHistory.emplace_back(
        cvRound(points2D[1].x),
        cvRound(points2D[1].y)
    );
    updatedHistory.emplace_back(
        cvRound(points2D[2].x),
        cvRound(points2D[2].y)
    );

    while (measuredHistory.size() > 120)
    {
        measuredHistory.erase(measuredHistory.begin());
        predictedHistory.erase(predictedHistory.begin());
        updatedHistory.erase(updatedHistory.begin());
    }

    canvas.setTo(cv::Scalar(255, 255, 255));

    for (int i = 1; i < static_cast<int>(measuredHistory.size()); i++)
    {
        cv::line(
            canvas,
            measuredHistory[i - 1],
            measuredHistory[i],
            cv::Scalar(255, 0, 0),
            2
        );
        cv::line(
            canvas,
            predictedHistory[i - 1],
            predictedHistory[i],
            cv::Scalar(0, 255, 0),
            2
        ); 
        cv::line(
            canvas,
            updatedHistory[i - 1],
            updatedHistory[i],
            cv::Scalar(0, 0, 255),
            2
        );
    }
    
    cv::circle(
            canvas,
            measuredHistory.back(),
            3,
            cv::Scalar(255, 0, 0),
            -1
        );
    cv::circle(
            canvas,
            predictedHistory.back(),
            3,
            cv::Scalar(0, 255, 0),
            -1
        );
    cv::circle(
            canvas,
            updatedHistory.back(),
            3,
            cv::Scalar(0, 0, 255),
            -1
        );
}