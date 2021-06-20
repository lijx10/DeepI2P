#ifndef FRUSTUM_REG_3D_HPP
#define FRUSTUM_REG_3D_HPP

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <random>
#include <tuple>

#include <Eigen/Dense>

#include "ceres/ceres.h"
#include "ceres/rotation.h"


// label == 1, that is, the point should be inside the image plane
struct GivenKOutsideImgError3D {
    GivenKOutsideImgError3D(double point_x, double point_y, double point_z,
                          double fx, double fy, double cx, double cy,
                          double H, double W)
        : point_x(point_x)
        , point_y(point_y)
        , point_z(point_z)
        , fx(fx)
        , fy(fy)
        , cx(cx)
        , cy(cy)
        , H(H)
        , W(W){}
    template <typename T>
    bool operator()(const T* const camera,
                    T* residuals) const {
        // camera[0] is the rotation around y axis.
        const T point[3] = {T(point_x), T(point_y), T(point_z)};
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3, 4, 5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // project p into image plane according to K (fx, fy, cx, cy)
        T pixel_x = fx * p[0] / p[2] + cx;
        T pixel_y = fy * p[1] / p[2] + cy;

        // get the cost of (pixel_x, pixel_y) inside or outside image plane
        auto x_dist_to_boundary = T(W*0.5) - ceres::abs(pixel_x-T(W*0.5));
        auto is_x_in = ceres::fmax(x_dist_to_boundary, T(0.0)) / x_dist_to_boundary;

        auto y_dist_to_boundary = T(H*0.5) - ceres::abs(pixel_y-T(H*0.5));
        auto is_y_in = ceres::fmax(y_dist_to_boundary, T(0.0)) / y_dist_to_boundary;

        auto is_front = ceres::fmax(p[2], T(0.0)) / p[2]; // front=1, back=0

        auto xy_dist = x_dist_to_boundary + y_dist_to_boundary;

        residuals[0] = xy_dist * is_front * is_x_in * is_y_in;
        // residuals[1] = ceres::fmax(T(H*0.5) - ceres::abs(pixel_y-T(H*0.5)), T(0.0));

        // std::cout<<"pixel_x: "<<pixel_x<<", pixel_y: "<<pixel_y<<std::endl;
        // std::cout<<"residuals[0]: "<<residuals[0]<<std::endl;

        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(double point_x, double point_y, double point_z,
                                       double fx, double fy, double cx, double cy,
                                       double H, double W) {
        return (new ceres::AutoDiffCostFunction<GivenKOutsideImgError3D, 1, 6>(
                    new GivenKOutsideImgError3D(point_x, point_y, point_z,
                                              fx, fy, cx, cy,
                                              H, W)));
    }
    double point_x;
    double point_y;
    double point_z;
    double fx;
    double fy;
    double cx;
    double cy;
    double H;
    double W;
};


// label == 1, that is, the point should be inside the image plane
struct GivenKInsideImgError3D {
    GivenKInsideImgError3D(double point_x, double point_y, double point_z,
                         double fx, double fy, double cx, double cy,
                         double H, double W)
        : point_x(point_x)
        , point_y(point_y)
        , point_z(point_z)
        , fx(fx)
        , fy(fy)
        , cx(cx)
        , cy(cy)
        , H(H)
        , W(W){}
    template <typename T>
    bool operator()(const T* const camera,
                    T* residuals) const {
        // camera[0] is the rotation around y axis.
        const T point[3] = {T(point_x), T(point_y), T(point_z)};
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3, 4, 5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // project p into image plane according to K (fx, fy, cx, cy)
        T pixel_x = fx * p[0] / p[2] + cx;
        T pixel_y = fy * p[1] / p[2] + cy;
        // get the cost of (pixel_x, pixel_y) inside or outside image plane
        residuals[0] = ceres::fmax(-pixel_x, T(0.0)) + ceres::fmax(pixel_x - T(W), T(0.0));
        residuals[1] = ceres::fmax(-pixel_y, T(0.0)) + ceres::fmax(pixel_y - T(H), T(0.0));

        // points inside image should have positive pz
        residuals[2] = ceres::fmax(-p[2], T(0.0)) * T(100.0);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(double point_x, double point_y, double point_z,
                                       double fx, double fy, double cx, double cy,
                                       double H, double W) {
        return (new ceres::AutoDiffCostFunction<GivenKInsideImgError3D, 3, 6>(
                    new GivenKInsideImgError3D(point_x, point_y, point_z,
                                             fx, fy, cx, cy,
                                             H, W)));
    }
    double point_x;
    double point_y;
    double point_z;
    double fx;
    double fy;
    double cx;
    double cy;
    double H;
    double W;
};



#endif // FRUSTUM_REG_3D_HPP
