#include <math.h>
#include <random>
#include <chrono>
#include "nuts.h"

NUTS::NUTS(double initial_position,
           double sigma,
           int burn_in_iterations) : sigma{sigma},
                                     warm_up_iterations{warm_up_iterations} {};

// logarithm of the target density function
double NUTS::log_target_density(double position)
{
    return -0.5 * position * position;
}

// gradient of the target density function
double NUTS::target_gradient(double position)
{
    return -position;
}

// logarithm of the density function of the auxiliary momentum
double NUTS::log_momentum_density(double momentum)
{
    return -0.5 * momentum * momentum;
}

// joint density of position and momentum
double NUTS::joint_density(state w)
{
    return exp(log_momentum_density(w.momentum) + log_target_density(w.position));
}

state NUTS::leapfrog(state w)
{
    w.momentum += 0.5 * step_size * target_gradient(w.position);
    w.position += step_size * w.momentum;
    w.momentum += 0.5 * step_size * target_gradient(w.position);

    return w;
}

bool NUTS::step_size_is_reasonable(state w_old, state w_new, double a)
{
    return pow((joint_density(w_new) / joint_density(w_old)), a) <= pow(2, -a);
}

// see Hoffman and Gelman (2014)
void NUTS::find_reasonable_step_size(state w)
{
    state w_new{leapfrog(w)};
    double a{2 * ((joint_density(w_new) / joint_density(w)) > 0.5) - 1};
    while (!step_size_is_reasonable(w, w_new, a))
    {
        step_size *= pow(2, a);
        w_new = leapfrog(w);
    }
}

double NUTS::sample_slice_threshold(state w)
{
    double upper_bound{
        exp(
            log_target_density(w.position) +
            log_momentum_density(w.momentum))};

    std::uniform_real_distribution<double> u_dist(0.0, upper_bound);

    return u_dist(rnd_generator);
}

double NUTS::sample_direction()
{
    double c{standard_normal(rnd_generator)};

    if (c >= 0)
    {
        return direction::right;
    }

    return direction::left;
}

double NUTS::single_sample(double initial_position)
{
    // r
    double momentum{sample_momentum()};
    // w (theta, r)
    state w{initial_position, momentum};
    // u
    double slice_threshold{sample_slice_threshold(w)};

    // theta^(-)
    double leftmost_position{w.momentum};
    // theta^(+)
    double rightmost_position{w.momentum};

    // s
    bool continue_integration{true};
    while (continue_integration)
    {
        // v
        double v{sample_direction()};

        if (v == direction::left)
        {
        }
        else
        {
        }
    }
}