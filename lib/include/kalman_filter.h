#pragma once

#include <Eigen/Core>

#include <opencv2/core.hpp>

class KalmanFilter
{
private:
    Eigen::Matrix<double, 6, 1> X;
    Eigen::Matrix<double, 3, 1> Z;
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 6> Q;
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 6, 6> P;
    Eigen::Matrix<double, 6, 3> K;
    Eigen::Matrix<double, 6, 6> I;
    Eigen::Matrix<double, 3, 6> H;

public:
    explicit KalmanFilter(const cv::Vec3d& tvec);
    KalmanFilter(const KalmanFilter&) = delete;
    KalmanFilter& operator = (const KalmanFilter&) = delete;
    ~KalmanFilter();
    void init(const cv::Vec3d& tvec);
    Eigen::VectorXd predict(double& dt);
    void update(const cv::Vec3d& tvec);
    cv::Vec3d getPosition() const;
};