#include <random>
#include <iostream>
#include <tuple>
#include "hmc.h"

// gradient computes the gradient of the logarithm of the target density.
double gradient(double q)
{
    return -q;
};

// kinetic_energy computes the negative logarithm of the density of the auxiliary momentum.
double kinetic_energy(double p)
{
    return 0.5 * (p * p);
}

// potential_energy computes the negative logarithm of the target density.
double potential_energy(double q)
{
    return 0.5 * (q * q);
};

std::tuple<double, double> leapfrog(double position, double momentum, int num_steps, double step_size)
{
    momentum -= step_size * 0.5 * gradient(position);
    for (int i{0}; i < num_steps; ++i)
    {
        position += step_size * momentum;
        momentum -= step_size * gradient(position);
    }
    position += step_size * momentum;
    momentum -= step_size * 0.5 * gradient(position);

    return {position, momentum};
}

double hamiltonian_monte_carlo(double initial_position, int num_steps, double step_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(momentum_mean, momentum_std);
    double initial_momentum{distribution(generator)};

    std::cout << "momentum: " << initial_momentum << std::endl;

    auto [final_position, final_momentum] = leapfrog(initial_position, initial_momentum, num_steps, step_size);

    // flipped sign of final momentum: see Betancourt 2018, page 39;
    double final_hamiltonian{
        kinetic_energy(-final_momentum) + potential_energy(final_position)};
    double initial_hamiltonian{
        kinetic_energy(initial_momentum) + potential_energy(initial_position)};

    if (exp(initial_hamiltonian - final_hamiltonian) > distribution(generator))
    {
        return final_position;
    }

    return initial_position;
}