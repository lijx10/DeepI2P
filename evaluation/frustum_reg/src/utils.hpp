#ifndef EXAMPLES_UTILS_H_
#define EXAMPLES_UTILS_H_

#include <iostream>
#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <Eigen/Dense>


class Point3fLabel{
public:
    Point3fLabel(const float x_, const float y_, const float z_, const int label_)
        : x(x_),
          y(y_),
          z(z_),
          label(label_){
    }

    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    int label{0};
};


class CameraIntrinsic{
public:
    CameraIntrinsic(const float fx_, const float fy_, const float cx_, const float cy_)
        : fx(fx_)
        , fy(fy_)
        , cx(cx_)
        , cy(cy_){

    }

    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};
};


CameraIntrinsic readCameraIntrinsicTxt(const std::string& filename){
    std::ifstream in(filename.c_str());
    std::string line;
    boost::char_separator<char> sep(" ");

    std::vector<float> raw_data;
    raw_data.reserve(9);
    while (!in.eof())
    {
        std::getline(in, line);
        in.peek();

        boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
        std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

        if (tokens.size() != 3) continue;

        for(size_t i=0;i<tokens.size();++i){
            raw_data.push_back(boost::lexical_cast<float>(tokens[i]));
        }
    }
    return CameraIntrinsic(raw_data[0], raw_data[4], raw_data[2], raw_data[5]);
}


template <typename PointT, typename ContainerT>
void readPointsTxt(const std::string& filename, ContainerT& points)
{
    std::ifstream in(filename.c_str());
    std::string line;
    boost::char_separator<char> sep(" ");
    while (!in.eof())
    {
        std::getline(in, line);
        in.peek();

        boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
        std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

        if (tokens.size() != 4) continue;
        float x = boost::lexical_cast<float>(tokens[0]);
        float y = boost::lexical_cast<float>(tokens[1]);
        float z = boost::lexical_cast<float>(tokens[2]);
        int label = boost::lexical_cast<int>(tokens[2]);

        points.emplace_back(x, y, z, label);
    }

    in.close();
}


void printPoints(const Eigen::MatrixXf points){
    std::cout<<"Point number: "<<points.rows()<<std::endl;
    for(size_t i=0;i<points.rows();++i){
        std::cout<<points(i, 0)<<", "<<points(i, 1)<<", "<<points(i, 2)<<std::endl;
    }

}

#endif /* EXAMPLES_UTILS_H_ */
