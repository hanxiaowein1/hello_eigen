#include "unit_test.h"

#include <iostream>
#include <Eigen/Dense>

TEST(GlobalTest, linear_equation_solver_basic)
{
    Eigen::Matrix3f A;
    Eigen::Vector3f b;
    A << 1,2,3,  4,5,6,  7,8,10;
    b << 3, 3, 4;
    std::cout << "Here is the matrix A:\n" << A << std::endl;
    std::cout << "Here is the vector b:\n" << b << std::endl;
    Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    std::cout << "The solution is:\n" << x << std::endl;
}

TEST(GlobalTest, linear_equation_solver_infinite)
{
    Eigen::Matrix3f A;
    Eigen::Vector3f b;
    //A << 1,2,3,  2,4,6,  7,8,10;
    //A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    A << 0, 0, 0,
        0, 0, 0,
        0, 0, 2;
    b << 0, 0, 2;
    //std::cout << "Here is the matrix A:\n" << A << std::endl;
    //std::cout << "Here is the vector b:\n" << b << std::endl;
    //Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    //std::cout << "The solution is:\n" << x << std::endl;
    Eigen::FullPivLU<Eigen::Matrix3f> a_lu_decomp(A);
    std::cout << a_lu_decomp.rank() << std::endl;
    Eigen::MatrixXf B(A.rows(), A.cols() + 1);
    B << A, b;
    std::cout << B << std::endl;
    Eigen::FullPivLU<Eigen::MatrixXf> b_lu_decomp(B);
    std::cout << b_lu_decomp.rank() << std::endl;
}