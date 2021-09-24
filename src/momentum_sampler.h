#pragma once
#include <chrono>
#include <random>
#include <Eigen/Dense>

class UVMomentumSampler
{
public:
    virtual double sample() = 0;
    virtual double log_density(double momentum) = 0;
};

class UVStandardNormalSampler : public UVMomentumSampler
{
private:
    std::mt19937 rnd_generator{static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    double log2pi = log(2 * M_PI);

public:
    double sample()
    {
        return standard_normal(rnd_generator);
    }

    double log_density(double momentum)
    {
        return -0.5 * (log2pi + momentum * momentum);
    }
};

class MVMomentumSampler
{
public:
    virtual Eigen::VectorXd sample() = 0;
    virtual double log_density(Eigen::VectorXd momentum) = 0;
    virtual size_t get_dimensions() = 0;
};

class MVStandardNormalSampler : public MVMomentumSampler
{
private:
    std::mt19937 rnd_generator{static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    double log2pi = log(2 * M_PI);
    size_t dimensions;

public:
    Eigen::VectorXd sample()
    {
        Eigen::VectorXd sample(dimensions);
        for (size_t i{0}; i < dimensions; ++i)
        {
            sample(i) = standard_normal(rnd_generator);
        }
        return sample;
    }

    double log_density(Eigen::VectorXd momentum)
    {
        return -0.5 * (dimensions * log2pi + momentum.transpose() * momentum);
    }

    size_t get_dimensions() { return dimensions; };

    MVStandardNormalSampler(size_t dimensions) : dimensions{dimensions} {};
};