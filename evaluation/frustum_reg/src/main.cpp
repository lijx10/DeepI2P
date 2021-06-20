#include <iostream>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <random>

#include "utils.hpp"
#include "keypoints.hpp"


int main(int argc, char** argv){   
    std::cout<<"hello"<<std::endl;

    std::string filename = "/home/jiaxin/PCLKeypoint/data/scan_001_points.dat";

    std::vector<Point3f> points;
    readPoints<Point3f>(filename, points);
    std::cout << "Read " << points.size() << " points." << std::endl;
    if (points.size() == 0){
        std::cerr << "Empty point cloud." << std::endl;
        return -1;
    }

    // random sampling
    uint32_t sample_number = 30000;
    if(sample_number < points.size()){
        std::vector<Point3f> points_sampled;
        std::random_shuffle(points.begin(), points.end());
        points = std::vector<Point3f>(points.begin(), points.begin()+sample_number);
    } else {
        sample_number = points.size();
    }

    // convert to Eigen
    Eigen::MatrixXf points_eigen(sample_number, 3);
    for(size_t i=0;i<points.size();++i){
        points_eigen(i, 0) = points[i].x;
        points_eigen(i, 1) = points[i].y;
        points_eigen(i, 2) = points[i].z;
    }


    // ISS keypoints
//    Eigen::MatrixXf keypoint_iss = keypointIss(points_eigen);
//    printPoints(keypoint_iss);

//    Eigen::MatrixXf keypoint_harris = keypointHarris(points_eigen);
//    printPoints(keypoint_harris);

    Eigen::MatrixXf keypoint_sift = keypointSift(points_eigen);
    printPoints(keypoint_sift);

}
