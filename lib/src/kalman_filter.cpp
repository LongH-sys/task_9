#include "kalman_filter.h"

#include <Eigen/Dense>

KalmanFilter::KalmanFilter(const cv::Vec3d& tvec)
{
    init(tvec);
}

KalmanFilter::~KalmanFilter()
{
}

void KalmanFilter::init(const cv::Vec3d& tvec)
{
    X << tvec[0], tvec[1], tvec[2], 0, 0, 0;
    Z << tvec[0], tvec[1], tvec[2];
    A.setIdentity();
    P.setIdentity();
    P.diagonal() << 0.0009, 0.0009, 0.01, 
                    1.0,    1.0,    1.0;
    R.setZero();
    R.diagonal() << 0.0009, 0.0009, 0.01;
    Q.setZero();
    Q.diagonal() << 0.0001, 0.0001, 0.0001,
                    0.001,  0.01,   0.01;
    K.setZero();
    I.setIdentity();
    H << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0;
}

Eigen::VectorXd KalmanFilter::predict(double& dt)
{
    A(0, 3) = dt;
    A(1, 4) = dt;
    A(2, 5) = dt;
    X = A * X;
    Eigen::MatrixXd A_T = A.transpose();
    P = A * P * A_T + Q;
    return X;
}

void KalmanFilter::update(const cv::Vec3d& tvec)
{
    Z << tvec[0], tvec[1], tvec[2];
    Eigen::Matrix<double, 6, 3> H_T = H.transpose();
    K = P * H_T * (H * P * H_T + R).inverse();
    X = X + K * (Z - H * X);
    P = (I - K * H) * P;
}

cv::Vec3d KalmanFilter::getPosition() const
{
    return cv::Vec3d(X(0), X(1), X(2));
}