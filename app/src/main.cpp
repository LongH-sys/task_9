#include "camera_manage.h"

#include "camera.h"

#include "sdk_manage.h"

#include "pnp_solver.h"

#include "kalman_filter.h"

#include "draw_picture.h"

#include <memory>

#include <utility>

#include <opencv2/highgui.hpp>

#include <chrono>

#include <iostream>

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

int main()
{
    auto logger = std::make_unique<YoloVino::YoloVinoLogger>(
        "/home/longh/vino_task/src/YoloVino/config/yolov8pose_vino_config.yaml");
    logger->print_yaml_info();
    logger->set_info_level(YoloVino::LoggerInfoLevel::warning_info);
    YoloVino::Yolov8poseVino vino(std::move(logger));
    SdkManager sdk;
    if (!sdk.initialize())
    {
        return -1;
    }
    CameraManager cam_manage;
    Camera camera(cam_manage.select_device());
    if (!camera.open())
    {
        if (camera.close())
        {
            sdk.finalize();
        }
        return -1;
    }
    if (!camera.set_trigger_mode(false))
    {
        if (camera.close())
        {
            sdk.finalize();
        }
        return -1;
    }
    if (!camera.start_grabbing())
    {
        if (camera.close())
        {
            sdk.finalize();
        }
        return -1;
    }
    cv::Mat frame;
    std::unique_ptr<KalmanFilter> kalman;
    
    using Clock = std::chrono::steady_clock;
    Clock::time_point lastTime;
    
    std::vector<cv::Point> measuredHistory;
    std::vector<cv::Point> predictedHistory;
    std::vector<cv::Point> updatedHistory;
    
    cv::namedWindow("Task_9", cv::WINDOW_NORMAL);
    std::cout << "连续采集中,请按q停止采集" << std::endl;

    while(true)
    {
        cv::Vec3d tvec;
        if (camera.get_frame(frame) && !frame.empty())
        {    
            auto currentTime = Clock::now();
            bool hasMeasurement = pnp_solver(frame, tvec, vino);
            if (!kalman)
            {
                if (hasMeasurement)
                {
                    kalman = std::make_unique<KalmanFilter>(tvec);
                    lastTime = currentTime;
                }
            }
            else
            {
                double dt = std::chrono::duration<double>(currentTime - lastTime).count();
                lastTime = currentTime;
                Eigen::VectorXd prior = kalman->predict(dt);
                cv::Vec3d predicted(prior(0), prior(1), prior(2));
                if (hasMeasurement)
                {
                    kalman->update(tvec);
                }
                cv::Vec3d updated = kalman->getPosition();
                draw_trajectory(
                    frame,
                    tvec, 
                    predicted, 
                    updated, 
                    cameraMatrix, 
                    distCoeffs, 
                    measuredHistory, 
                    predictedHistory, 
                    updatedHistory,
                    hasMeasurement
                );
            }
            cv::putText(
                frame, 
                "measure", 
                cv::Point(50, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 
                3, 
                cv::Scalar(255, 0, 0), 
                2,
                cv::LINE_AA
            );
            cv::putText(
                frame, 
                "predict", 
                cv::Point(50, 200), 
                cv::FONT_HERSHEY_SIMPLEX, 
                3, 
                cv::Scalar(0, 255, 0), 
                2,
                cv::LINE_AA
            );
            cv::putText(
                frame, 
                "update", 
                cv::Point(50, 300), 
                cv::FONT_HERSHEY_SIMPLEX, 
                3, 
                cv::Scalar(0, 0, 255), 
                2,
                cv::LINE_AA
            ); 
            cv::imshow("Task_9", frame);
        }
        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            break;
        }
        if (cv::getWindowProperty("Task_9", cv::WND_PROP_AUTOSIZE) < 0)
        {
            break;
        }
    }
    cv::destroyAllWindows();
    if (!camera.stop_grabbing())
    {
        if (camera.close())
        {
            sdk.finalize();
        }
        return -1;
    }
    if (!camera.close())
    {
        return -1;
    }
    if (!sdk.finalize())
    {
        return -1;
    }
    return 0;
}