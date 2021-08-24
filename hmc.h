#pragma once
#include <vector>
#include <random>

class HMC
{
private:
    std::mt19937 rnd_generator;
    std::normal_distribution<double> momentum_sampler{};
    std::normal_distribution<double> normal_zero_sampler{0.0, 1.0};

    // user defined functions
    // potential_energy computes the negative logarithm of the target density.
    double (*potential_energy)(double q);

    // gradient computes the gradient of the negative logarithm of the target density.
    double (*gradient)(double q);

    // private class methods
    std::tuple<double, double> leapfrog(double position, double momentum, int num_steps, double step_size);

    // kinetic_energy computes the negative logarithm of the density of the auxiliary momentum.
    double kinetic_energy(double p);

    double sample_momentum();
    double sample_zero_mean();

public:
    HMC(double momentum_mean,
        double momentum_std,
        double (*gradient)(double q),
        double (*potential_energy)(double q));

    double hamiltonian_monte_carlo(double initial_position, int num_steps, double step_size);
};