#include "registration_2d.hpp"
#include "registration_3d.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"


std::tuple<Eigen::MatrixXd, double, Eigen::VectorXd> solvePGivenK(const Eigen::MatrixXd points,
                                                                  const Eigen::VectorXi labels,
                                                                  const Eigen::Matrix3d K,
                                                                  const double init_y_angle,
                                                                  const Eigen::Vector3d init_T,
                                                                  const double H,
                                                                  const double W,
                                                                  const std::vector<double> t_xyz_lower_bound,
                                                                  const std::vector<double> t_xyz_upper_bound,
                                                                  const int max_iter,
                                                                  const bool is_debug,
                                                                  const bool is_2d){
    double H_1 = H-1;
    double W_1 = W-1;

    size_t translation_param_offset = 1;
    if(is_2d){
        translation_param_offset = 1;
    } else {
        translation_param_offset = 3;
    }

    double* camera_params;
    size_t camera_params_len;
    // initialization
    if(is_2d){
        camera_params_len = 4;
        camera_params = new double[camera_params_len];
        camera_params[0] = init_y_angle;
        for(long i=1;i<4;++i){
            camera_params[i] = init_T(i-1);
        }
    } else {
        camera_params_len = 6;
        camera_params = new double[camera_params_len];
        camera_params[0] = 0;
        camera_params[1] = init_y_angle;
        camera_params[2] = 0;
        for(long i=3;i<6;++i){
            camera_params[i] = init_T(i-3);
        }
    }

    if(is_debug){
        std::cout<<"init camera param:\n";
        for(size_t i=0;i<camera_params_len;++i){
            std::cout<<camera_params[i]<<", ";
        }
        std::cout<<std::endl;

        double* debug_R_array = new double[9];
        double* angle_axis = new double[3];
        if(is_2d){
            angle_axis[0] = 0;
            angle_axis[1] = camera_params[0];
            angle_axis[2] = 0;
        } else {
            angle_axis[0] = camera_params[0];
            angle_axis[1] = camera_params[1];
            angle_axis[2] = camera_params[2];
        }
        ceres::AngleAxisToRotationMatrix(angle_axis, debug_R_array);
        Eigen::Matrix3d debug_R = Eigen::Map<Eigen::Matrix3d>(debug_R_array);
        std::cout<<"Initialized camera_params to R:\n";
        std::cout<<debug_R<<std::endl;
        delete [] angle_axis;
        delete [] debug_R_array;
    }

    ceres::Problem problem;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    std::vector<ceres::ResidualBlockId> residual_block_ids;
    residual_block_ids.reserve(static_cast<size_t>(labels.size()));

    for(long i=0;i<labels.size();++i){
        // for points inside image plane
        if(labels(i)==1){
            ceres::CostFunction* cost_function;
            if(is_2d){
                cost_function =
                        GivenKInsideImgError2D::Create(points(0, i), points(1, i), points(2, i),
                                                     fx, fy, cx, cy,
                                                     H_1, W_1);
            } else {
                cost_function =
                        GivenKInsideImgError3D::Create(points(0, i), points(1, i), points(2, i),
                                                     fx, fy, cx, cy,
                                                     H_1, W_1);
            }
            ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function,
                                                                       new ceres::CauchyLoss(1.0),
                                                                       camera_params);
            residual_block_ids.push_back(block_id);
        } else if(labels(i)==0){
            // for points outside image plane
            ceres::CostFunction* cost_function;
            if(is_2d){
                cost_function =
                        GivenKOutsideImgError2D::Create(points(0, i), points(1, i), points(2, i),
                                                      fx, fy, cx, cy,
                                                      H_1, W_1);
            } else {
                cost_function =
                        GivenKOutsideImgError3D::Create(points(0, i), points(1, i), points(2, i),
                                                      fx, fy, cx, cy,
                                                      H_1, W_1);
            }
            ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function,
                                                                       new ceres::CauchyLoss(1.0),
                                                                       camera_params);
            residual_block_ids.push_back(block_id);
        }
    }

    // set the lower and upper bounds
    for(size_t i=0;i<3;++i){
        problem.SetParameterLowerBound(camera_params,
                                       static_cast<int>(i+translation_param_offset),
                                       t_xyz_lower_bound.at(i));
        problem.SetParameterUpperBound(camera_params,
                                       static_cast<int>(i+translation_param_offset),
                                       t_xyz_upper_bound.at(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iter;
    if(false==is_debug){
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;
    } else {
        options.minimizer_progress_to_stdout = true;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // evaluation to get final cost and residuals
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = residual_block_ids;
    double final_cost;
    std::vector<double> residuals;
    problem.Evaluate(eval_options, &final_cost, &residuals, nullptr, nullptr);
    Eigen::VectorXd residuals_egien = Eigen::Map<Eigen::VectorXd>(residuals.data(),residuals.size());

    if(is_debug){
        std::cout << summary.FullReport() << "\n";
    }

    double* R_array = new double[9];
    double* angle_axis = new double[3];
    if(is_2d){
        angle_axis[0] = 0;
        angle_axis[1] = camera_params[0];
        angle_axis[2] = 0;
    } else {
        angle_axis[0] = camera_params[0];
        angle_axis[1] = camera_params[1];
        angle_axis[2] = camera_params[2];
    }
    ceres::AngleAxisToRotationMatrix<double>(angle_axis, R_array);
    Eigen::Matrix3d R = Eigen::Map<Eigen::Matrix3d>(R_array);
    Eigen::Vector3d T = Eigen::Map<Eigen::Vector3d>(camera_params+translation_param_offset);
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity(4, 4);
    P.topLeftCorner(3,3) = R;
    P(0, 3) = T(0);
    P(1, 3) = T(1);
    P(2, 3) = T(2);

    // memory clean
    delete [] camera_params;
    delete [] R_array;
    delete [] angle_axis;
    return std::make_tuple(P, final_cost, residuals_egien);
}

namespace py = pybind11;

PYBIND11_MODULE(FrustumRegistration, m) {
    m.doc() = "Frustum Registration";

    m.def("solvePGivenK",
          &solvePGivenK,
          py::arg("points"),
          py::arg("labels"),
          py::arg("K"),
          py::arg("init_y_angle"),
          py::arg("init_T"),
          py::arg("H"),
          py::arg("W"),
          py::arg("t_xyz_lower_bound"),
          py::arg("t_xyz_upper_bound"),
          py::arg("max_iter"),
          py::arg("is_debug"),
          py::arg("is_2d"));


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
