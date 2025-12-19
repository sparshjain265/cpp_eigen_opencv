#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>

int main()
{
    auto m = Eigen::Matrix3f::Identity();
    std::cout << "Eigen matrix:\n"
              << m << std::endl;

    auto img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Test", img);
    cv::waitKey(0);

    return 0;
}
