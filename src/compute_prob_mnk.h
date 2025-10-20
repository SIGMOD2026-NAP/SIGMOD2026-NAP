#pragma once
#include <iostream>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
static const double PI = 3.14159265358979323846;
static const double SQRT2 = 1.41421356237309504880;
inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

double Phi(double x) {
    return 0.5 * (1 + boost::math::erf(x / std::sqrt(2)));
}

double integrand(double x, int m, int n, double k) {
    double phi_x = std::exp(-x * x / 2) / std::sqrt(2 * PI); // Standard normal PDF
    double Phi_x = Phi(x);
    double Phi_x_over_k = Phi(x / k);
    return (1 - std::pow(Phi_x_over_k, m)) * n * phi_x * std::pow(Phi_x, n - 1);
}

double compute_integral(int m, int n, double k) {
    auto f = [&](double x) { return integrand(x, m, n, k); };

    // Integration limits
    double lower_limit = -10.0;
    double upper_limit = 10.0;

    // Use Gauss-Kronrod quadrature for numerical integration
    double result = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(
        f, lower_limit, upper_limit, 15, 1e-10);

    return result;
}