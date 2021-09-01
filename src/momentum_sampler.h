#pragma once
#include <chrono>
#include <random>
#include <Eigen/Dense>

template <typename T>
class MomentumSampler
{
public:
    virtual T sample_momentum() = 0;
};

class StandardNormalSampler : public MomentumSampler<double>
{
private:
    std::mt19937 rnd_generator{static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())};
    std::normal_distribution<double> standard_normal{0.0, 1.0};

public:
    double sample_momentum()
    {
        return standard_normal(rnd_generator);
    }

    StandardNormalSampler() {}
};

class MVStandardNormalSampler : public MomentumSampler<Eigen::VectorXd>
{
private:
    std::mt19937 rnd_generator{static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    size_t dimensions;

public:
    Eigen::VectorXd sample_momentum()
    {
        Eigen::VectorXd sample(dimensions);
        for (size_t i{0}; i < dimensions; ++i)
        {
            sample(i) = standard_normal(rnd_generator);
        }
        return sample;
    }

    MVStandardNormalSampler(size_t dimensions) : dimensions{dimensions} {};
};