#ifndef COMPARE_H
#define COMPARE_H

#include "multi_array_typedefs.h"
#include <Eigen/Dense>
#include <iostream>

inline bool
floating_point_equal(double a, double b, double tolerance)
{
    if (std::abs(a) < tolerance) {
        return (std::abs(a - b) < tolerance);
    } else {
        return (std::abs((a - b) / a) < tolerance);
    }
}

inline double
marray_check_equal(MArray3d const& a, MArray3d const& b, double tolerance)
{
    //    return a.isApprox(b, tolerance);
    //    std::cerr << "marray_check_equal " << a.shape()[0] << ", " <<
    //    a.shape()[1] << ", " << a.shape()[2] << std::endl;
    double max_diff = -1;
    for (long i = 0; i < a.shape()[0]; ++i) {
        for (long j = 0; j < a.shape()[1]; ++j) {
            for (long k = 0; k < a.shape()[2]; ++k) {
                double diff = std::abs(a[i][j][k] - b[i][j][k]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (!floating_point_equal(a[i][j][k], b[i][j][k], tolerance)) {
                    std::cerr << "marray_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a[i][j][k] << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b[i][j][k] << std::endl;
                    std::cerr << "  a-b = " << a[i][j][k] - b[i][j][k]
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

inline bool
marray_check_equal(MArray3dc const& a, MArray3dc const& b, double tolerance)
{
    //    return a.isApprox(b, tolerance);
    //    std::cerr << "marray_check_equal " << a.shape()[0] << ", " <<
    //    a.shape()[1] << ", " << a.shape()[2] << std::endl;
    for (long i = 0; i < a.shape()[0]; ++i) {
        for (long j = 0; j < a.shape()[1]; ++j) {
            for (long k = 0; k < a.shape()[2]; ++k) {
                if ((!floating_point_equal(a[i][j][k].real(), b[i][j][k].real(),
                                           tolerance) ||
                     (!floating_point_equal(a[i][j][k].imag(),
                                            b[i][j][k].imag(), tolerance)))) {
                    std::cerr << "marray_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a[i][j][k] << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b[i][j][k] << std::endl;
                    std::cerr << "  a-b = " << a[i][j][k] - b[i][j][k]
                              << ", tolerance = " << tolerance << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename shape_T, typename array_T>
bool
general_carray_check_equal(shape_T const& shape, array_T const& a,
                           array_T const& b, double tolerance)
{
    for (long i = 0; i < shape[0]; ++i) {
        for (long j = 0; j < shape[1]; ++j) {
            for (long k = 0; k < shape[2]; ++k) {
                if ((!floating_point_equal(a(i, j, k).real(), b(i, j, k).real(),
                                           tolerance) ||
                     (!floating_point_equal(a(i, j, k).imag(),
                                            b(i, j, k).imag(), tolerance)))) {
                    std::cerr << "general_array_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a(i, j, k) << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b(i, j, k) << std::endl;
                    std::cerr << "  a-b = " << a(i, j, k) - b(i, j, k)
                              << ", tolerance = " << tolerance << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename shape_T, typename array_T>
double
general_array_check_equal(shape_T const& shape, array_T const& a,
                          array_T const& b, double tolerance)
{
    double max_diff = -1;
    for (long i = 0; i < shape[0]; ++i) {
        for (long j = 0; j < shape[1]; ++j) {
            for (long k = 0; k < shape[2]; ++k) {
                auto diff = std::abs(a(i,j,k) - b(i,j,k));
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (!floating_point_equal(a(i, j, k), b(i, j, k), tolerance)) {
                    std::cerr << "general_array_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a(i, j, k) << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b(i, j, k) << std::endl;
                    std::cerr << "  a-b = " << a(i, j, k) - b(i, j, k)
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

inline bool
eigen_check_equal(Eigen::Matrix<double, Eigen::Dynamic, 7> const& a,
                  Eigen::Matrix<double, Eigen::Dynamic, 7> const& b,
                  double tolerance)
{
    //    return a.isApprox(b, tolerance);
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        for (Eigen::Index j = 0; j < a.cols(); j++) {
            if (!floating_point_equal(a(i, j), b(i, j), tolerance)) {
                std::cerr << "eigen_check_equal:\n";
                std::cerr << "  a(" << i << "," << j << ") = " << a(i, j)
                          << std::endl;
                std::cerr << "  b(" << i << "," << j << ") = " << b(i, j)
                          << std::endl;
                std::cerr << "  a-b = " << a(i, j) - b(i, j)
                          << ", tolerance = " << tolerance << std::endl;
                return false;
            }
        }
    }
    return true;
}

#endif // COMPARE_H
