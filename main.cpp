#include <vector>
#include <iostream>
#include "hmc.h"

// gradient computes the gradient of the logarithm of the target density.
double gradient(double q)
{
    return -q;
};

// potential_energy computes the negative logarithm of the target density.
double potential_energy(double q)
{
    return 0.5 * (q * q);
};

int main(int argc, char *argv[])
{
    HMC hmc{0.0, 1.0, &gradient, &potential_energy};

    if (argc != 5)
    {
        std::cout
            << "Expected 4 arguments (int sample size, int num_steps, double step_size, double sampled_value), but got "
            << argc - 1 << std::endl;

        return 1;
    }

    int sample_size{strtol(argv[1], nullptr, 10)};
    int num_steps{strtol(argv[2], nullptr, 10)};
    double step_size{strtod(argv[3], nullptr)};
    double sampled_value{strtod(argv[4], nullptr)};

    for (int i{0}; i <= sample_size; ++i)
    {
        sampled_value = hmc.hamiltonian_monte_carlo(sampled_value, num_steps, step_size);
        std::cout << sampled_value << std::endl;
    }

    return 0;
}
