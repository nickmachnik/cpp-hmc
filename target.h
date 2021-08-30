#pragma once
#include <cmath>

class Target
{
public:
    // Logarithm of the target density function.
    //
    // TODO: This does not seem like a good way of achieving
    // something like an interface, what is the idiom here?
    virtual double log_density(double position) = 0;
    // Gradient of the log target density function.
    virtual double log_density_gradient(double position) = 0;
};

class Weibull : public Target
{
private:
    // shape
    double k;
    // scale
    double lambda;
    double log_frac_k_lambda;

public:
    double log_density(double position)
    {
        if (position <= 0)
        {
            return 0;
        }
        return log_frac_k_lambda +
               (k - 1) * (log(position) - log(lambda)) -
               pow((position / lambda), k);
    }

    double log_density_gradient(double position)
    {
        if (position <= 0)
        {
            return 0;
        }
        return ((k - 1) / position) - (k * pow(position / lambda, k - 1)) / lambda;
    }

    Weibull(double k, double lambda) : k{k}, lambda{lambda}
    {
        log_frac_k_lambda = log(k) - log(lambda);
    };
};

class StandardNormal : public Target
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

class UnscaledStandardNormal : public Target
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