#pragma once
#include <cmath>
#include <Eigen/Dense>

class MVTarget
{
public:
    // Logarithm of the target density function.
    virtual double log_density(Eigen::VectorXd position) = 0;
    // Gradient of the log target density function.
    virtual Eigen::VectorXd log_density_gradient(Eigen::VectorXd position) = 0;
};

class UVTarget
{
public:
    // Logarithm of the target density function.
    virtual double log_density(double position) = 0;
    // Gradient of the log target density function.
    virtual double log_density_gradient(double position) = 0;
};

class MVN : public MVTarget
{
private:
    double d;
    double log2pi = log(2 * M_PI);
    const Eigen::VectorXd &mean;
    const Eigen::MatrixXd &sigma;
    double log_sigma_determinant;
    Eigen::MatrixXd sigma_inverse;

public:
    MVN(const Eigen::VectorXd &mean, const Eigen::MatrixXd &sigma) : mean{mean}, sigma{sigma}, d{mean.rows()}
    {
        log_sigma_determinant = log(sigma.determinant());
        sigma_inverse = sigma.inverse();
    }

    double log_density(Eigen::VectorXd position)
    {
        return -0.5 * ((d * log2pi) +
                       log_sigma_determinant +
                       ((position - mean).transpose() * sigma_inverse * (position - mean)));
    }

    Eigen::VectorXd log_density_gradient(Eigen::VectorXd position)
    {
        return -sigma_inverse * (position - mean);
    }
};

class Laplace : public UVTarget
{
private:
    // mean
    double u;
    // average absolute deviation
    double b;
    double log_two_b;

public:
    double log_density(double position)
    {
        return -log_two_b - (abs(position - u) / b);
    }
    double log_density_gradient(double position)
    {
        if (position < u)
        {
            return -(1 / b);
        }

        return (1 / b);
    }
    Laplace(double u, double b) : u{u}, b{b}
    {
        log_two_b = log(2 * b);
    }
};

class StandardNormal : public UVTarget
{
public:
    double log_density(double position)
    {
        return (-0.5 * position * position) - log(sqrt(M_PI * 2));
    }
    double log_density_gradient(double position)
    {
        return -position;
    }
};

class UnscaledStandardNormal : public UVTarget
{
public:
    double log_density(double position)
    {
        return -0.5 * position * position;
    }
    double log_density_gradient(double position)
    {
        return -position;
    }
};

// Distributions with finite support; sampler cannot handle them yet.
// class Exponential : public Target
// {
// private:
//     // rate
//     double lambda;
//     double log_lambda;

// public:
//     double log_density(double position)
//     {
//         if (position < 0.0)
//         {
//             return 0.0001;
//         }
//         return log_lambda - lambda * position;
//     }

//     double log_density_gradient(double position)
//     {
//         if (position < 0.0)
//         {
//             return 0.1;
//         }
//         return -lambda;
//     }

//     explicit Exponential(double lambda) : lambda{lambda}
//     {
//         log_lambda = log(lambda);
//     }
// };

// class Weibull : public Target
// {
// private:
//     // shape
//     double k;
//     // scale
//     double lambda;
//     double log_frac_k_lambda;

// public:
//     double log_density(double position)
//     {
//         if (position <= 0)
//         {
//             return 0;
//         }
//         return log_frac_k_lambda +
//                (k - 1) * (log(position) - log(lambda)) -
//                pow((position / lambda), k);
//     }

//     double log_density_gradient(double position)
//     {
//         if (position <= 0)
//         {
//             return 0;
//         }
//         return ((k - 1) / position) - (k * pow(position / lambda, k - 1)) / lambda;
//     }

//     Weibull(double k, double lambda) : k{k}, lambda{lambda}
//     {
//         log_frac_k_lambda = log(k) - log(lambda);
//     }

//     double get_log_frac_k_lambda()
//     {
//         return log_frac_k_lambda;
//     }
// };