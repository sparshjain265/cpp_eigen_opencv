/**
 * MIT License
 *
 * Copyright (c) 2025 Sparsh Jain
 *
 */

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <cpp_eigen_opencv/shared/ndarray.hpp>
#include <cpp_eigen_opencv/shared/geometry.hpp>

int main()
{
    auto m = Eigen::Matrix3f::Identity();
    std::cout << "Eigen matrix:\n"
              << m << std::endl;

    ND::test();
    Geometry::testConvexHull();
    Geometry::testMinAreaRectangle();

    auto img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Test", img);
    while (cv::getWindowProperty("Test", cv::WND_PROP_VISIBLE) > 0)
    {
        cv::waitKey(1000);
    }
    cv::destroyAllWindows();

    return 0;
}
