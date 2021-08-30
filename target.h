#pragma once

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

class StandardNormal : public Target
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