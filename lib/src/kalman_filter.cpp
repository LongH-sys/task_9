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
    const double q[3] = {1.0, 1.0, 1.0};
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    for (int i = 0; i < 3; i++)
    {
        Q(i, i) = q[i] * dt3 / 3.0;
        Q(i, i + 3) = q[i] * dt2 / 2.0;
        Q(i + 3, i) = q[i] * dt2 / 2.0;
        Q(i + 3, i + 3) = q[i] * dt;
    }
    Eigen::MatrixXd A_T = A.transpose();
    P = A * P * A_T + Q;
    return X;
}

void KalmanFilter::update(const cv::Vec3d& tvec)
{
    Z << tvec[0], tvec[1], tvec[2];
    Eigen::Matrix<double, 6, 3> H_T = H.transpose();
    Eigen::Matrix3d S = H * P * H_T + R;
    Eigen::Matrix<double, 6, 3>PHt = P * H_T;
    Eigen::Matrix<double, 3, 6>K_T = S.ldlt().solve(PHt.transpose());
    K = K_T.transpose();
    X = X + K * (Z - H * X);
    P = (I - K * H) * P;
}

cv::Vec3d KalmanFilter::getPosition() const
{
    return cv::Vec3d(X(0), X(1), X(2));
}