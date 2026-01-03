/**
 * MIT License
 *
 * Copyright (c) 2025 Sparsh Jain
 *
 */

#include <cpp_eigen_opencv/shared/ndarray.hpp>
#include <cpp_eigen_opencv/shared/geometry.hpp>

namespace Geometry
{
    void testConvexHull(
        const NDArray<double, 2> &points)
    {
        const auto N = points.shape()[0];
        const auto hull = computeConvexHull(points);
        const auto n = hull.shape()[0];

        if (n < 3)
            return; // Trivial hull, no need to test further

        constexpr double eps = 1e-6;
        constexpr auto equal = [](double a, double b)
        {
            return std::abs(a - b) < eps;
        };

        // Hull Points are a subset of input points
        for (size_type i = 0; i < n; ++i)
        {
            bool found = false;
            for (size_type j = 0; j < N; ++j)
            {
                if ((equal(hull(i, 0), points(j, 0))) &&
                    (equal(hull(i, 1), points(j, 1))))
                {
                    found = true;
                    break;
                }
            }

            assert(found && "Hull point not found in input points");
        }

        // Hull points are convex in counter-clockwise order
        for (size_type i = 0; i < n; ++i)
        {
            const auto p0 = NDArray<const double, 1>(&hull(i, 0), {2});
            const auto p1 = NDArray<const double, 1>(&hull((i + 1) % n, 0), {2});
            const auto p2 = NDArray<const double, 1>(&hull((i + 2) % n, 0), {2});

            const auto v1 = p1 - p0;
            const auto v2 = p2 - p1;

            const auto crossProduct = cross(v1, v2);
            assert(crossProduct >= -eps && "Hull points not in counter-clockwise order");
        }

        // All points lie inside or on the hull
        for (size_type i = 0; i < N; ++i)
        {
            const auto p = NDArray<const double, 1>(&points(i, 0), {2});
            bool inside = true;
            for (size_type j = 0; j < n; ++j)
            {
                const auto p0 = NDArray<const double, 1>(&hull(j, 0), {2});
                const auto p1 = NDArray<const double, 1>(&hull((j + 1) % n, 0), {2});
                const auto edge = p1 - p0;
                const auto toPoint = p - p0;
                const auto crossProduct = cross(edge, toPoint);
                if (crossProduct < -eps)
                {
                    inside = false;
                    break;
                }
            }
            assert(inside && "Point not inside hull");
        }
    }

    void testMinAreaRectangle(
        const NDArray<double, 2> &points)
    {
        const auto rectangle = minAreaRectangle(points);
        const auto N = points.shape()[0];

        const double cosA = std::cos(rectangle.angle);
        const double sinA = std::sin(rectangle.angle);

        const auto u = NDArray<double, 1>({cosA, sinA});
        const auto v = NDArray<double, 1>({-sinA, cosA});

        // Check that all points lie within the rectangle
        for (size_type i = 0; i < N; ++i)
        {
            const auto p = NDArray<const double, 1>(&points(i, 0), {2});

            // Translate point to rectangle center
            const auto translated = p - rectangle.center;

            // Rotate point by -angle
            const double xRotated = ND::dot(translated, u);
            const double yRotated = ND::dot(translated, v);

            // Check if point lies within rectangle bounds
            const double halfWidth = rectangle.size[0] * 0.5;
            const double halfHeight = rectangle.size[1] * 0.5;

            constexpr double eps = 1e-6;
            assert((std::abs(xRotated) <= halfWidth + eps) &&
                   (std::abs(yRotated) <= halfHeight + eps) &&
                   "Point lies outside the minimum area rectangle");
        }
    }

}
