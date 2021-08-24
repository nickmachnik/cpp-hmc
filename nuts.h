#pragma once
#include <vector>
#include <random>
#include <chrono>

// a position-momentum state
struct state
{
    double position;
    double momentum;

    state(double position, double momentum) : position{position}, momentum{momentum} {};
};

enum direction
{
    left = -1,
    right = 1,
};

// A No-U-Turn Sampler with Dual Averaging.
// Featuring dynamic integration length and automatic step size selection.
class NUTS
{
private:
    const double gamma{0.05};
    const double t_0{10.0};
    const double kappa{0.75};

    std::mt19937 rnd_generator{std::chrono::steady_clock::now().time_since_epoch().count()};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    double step_size{1};
    double mu;
    double sigma;
    double H;
    int warm_up_iterations;

    double sample_momentum();
    double sample_direction();
    double sample_slice_threshold(state w);
    state leapfrog(state w);
    bool step_size_is_reasonable(state w_old, state w_new, double alpha);
    void find_reasonable_step_size(state w);

    double joint_density(state w);
    double target_gradient(double position);
    double log_target_density(double position);
    double log_momentum_density(double momentum);

public:
    NUTS(double initial_position, double sigma, int warm_up_iterations);
    double single_sample(double initial_position);
};
