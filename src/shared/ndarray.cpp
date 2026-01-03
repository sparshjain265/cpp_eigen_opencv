/**
 * MIT License
 *
 * Copyright (c) 2025 Sparsh Jain
 *
 */

#include <iostream>
#include <array>

#include <cpp_eigen_opencv/shared/ndarray.hpp>

namespace ND
{

    void test()
    {
        std::cout << "Running test for NDArray..." << std::endl;

        {
            // Test Shape
            std::cout << "Testing Shape..." << std::endl;
            const Shape<2> shape({3, 4});
            std::cout << "Size: " << shape.size() << std::endl;
            std::cout << "Shape[0]: " << shape[0] << std::endl;
            std::cout << "Shape[1]: " << shape[1] << std::endl;
        }

        {
            // Const Non-Owning NDArray
            const int data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            NDArray<const int, 2> array(data, {3, 4});

            // Uncommenting the following line should result in a compile-time error
            // array(0, 0) = 100;

            std::cout << "Array(0, 0): " << array(0, 0) << std::endl;
        }

        {
            // Non-Owning NDArray
            int data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            NDArray<int, 2> array(data, {3, 4});
            array(0, 0) = 100;
            std::cout << "Array(0, 0): " << array(0, 0) << std::endl;
        }

        {
            // Owning NDArray
            auto array = NDArray<int, 2>::Zeros({3, 4});
            array(0, 0) = 100;
            std::cout << "Array(0, 0): " << array(0, 0) << std::endl;
        }
    }

}
