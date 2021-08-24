#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include "hmc.h"
#include "nuts.h"

// gradient computes the gradient of the negative logarithm of the target density.
auto gradient(double q) -> double
{
    return q;
};

// potential_energy computes the negative logarithm of the target density.
auto potential_energy(double q) -> double
{
    return 0.5 * (q * q);
};

auto main(int argc, char *argv[]) -> int
{
    if (argc < 2)
    {
        std::cout << "first argument must be either 'hmc' or 'nuts'" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "hmc") == 0)
    {
        if (argc != 6)
        {
            std::cout
                << "Expected 5 arguments ('hmc', int sample size, int num_steps, double step_size, double sampled_value), but got "
                << argc - 1 << std::endl;

            return 1;
        }

        HMC hmc{0.0, 1.0, &gradient, &potential_energy};
        int sample_size{strtol(argv[2], nullptr, 10)};
        int num_steps{strtol(argv[3], nullptr, 10)};
        double step_size{strtod(argv[4], nullptr)};
        double sampled_value{strtod(argv[5], nullptr)};

        for (int i{0}; i <= sample_size; ++i)
        {
            sampled_value = hmc.hamiltonian_monte_carlo(sampled_value, num_steps, step_size);
        }
    }
    else if (strcmp(argv[1], "nuts") == 0)
    {
        if (argc != 6)
        {
            std::cout
                << "Expected 5 arguments ('nuts', double initial_position, int total_iterations, int warm_up_iterations, double sigma), but got "
                << argc - 1 << std::endl;

            return 1;
        }

        double position{strtod(argv[2], nullptr)};
        size_t total_iterations{strtol(argv[3], nullptr, 10)};
        size_t warm_up_iterations{strtol(argv[4], nullptr, 10)};
        double sigma{strtod(argv[5], nullptr)};

        NUTS nuts{sigma};
        std::vector<double> res{nuts.sample(position, total_iterations, warm_up_iterations)};
    }
    else
    {
        std::cout << "first argument must be either 'hmc' or 'nuts'" << std::endl;
        return 1;
    }

    return 0;
}
