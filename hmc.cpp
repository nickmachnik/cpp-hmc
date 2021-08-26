#include <random>
#include <iostream>
#include <tuple>
#include "hmc.h"

void output_trajectory(double position, double momentum)
{
    std::cout << position << "\t" << momentum << std::endl;
}

HMC::HMC(double momentum_mean,
         double momentum_std,
         double (*gradient)(double q),
         double (*potential_energy)(double q)) : gradient{gradient},
                                                 potential_energy{potential_energy}
{
    std::normal_distribution<double> momentum_sampler{momentum_mean, momentum_std};
};

double HMC::sample_momentum()
{
    return momentum_sampler(rnd_generator);
}

double HMC::sample_zero_mean()
{
    return normal_zero_sampler(rnd_generator);
}

// kinetic_energy computes the negative logarithm of the density of the auxiliary momentum.
double HMC::kinetic_energy(double p)
{
    return 0.5 * (p * p);
}

std::tuple<double, double> HMC::leapfrog(double position, double momentum, int num_steps, double step_size)
{
    for (int i{0}; i < num_steps; ++i)
    {
        momentum -= step_size * 0.5 * gradient(position);
        position += step_size * momentum;
        momentum -= step_size * 0.5 * gradient(position);
        // output_trajectory(position, momentum);
    }

    return {position, momentum};
}

double HMC::hamiltonian_monte_carlo(double initial_position, int num_steps, double step_size)
{
    double initial_momentum{sample_momentum()};

    auto [final_position, final_momentum] = leapfrog(initial_position, initial_momentum, num_steps, step_size);

    // flipped sign of final momentum: see Betancourt 2018, page 39;
    double final_hamiltonian{
        kinetic_energy(-final_momentum) + potential_energy(final_position)};
    double initial_hamiltonian{
        kinetic_energy(initial_momentum) + potential_energy(initial_position)};

    double normal_sample{sample_zero_mean()};

    if (exp(initial_hamiltonian - final_hamiltonian) > normal_sample)
    {
        return final_position;
    }

    return initial_position;
}