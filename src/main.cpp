#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include "hmc.h"
#include "nuts.h"
#include "mvnuts.h"
#include "momentum_sampler.h"
#include "target.h"

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
        int sample_size{static_cast<int>(strtol(argv[2], nullptr, 10))};
        int num_steps{static_cast<int>(strtol(argv[3], nullptr, 10))};
        double step_size{strtod(argv[4], nullptr)};
        double sampled_value{strtod(argv[5], nullptr)};

        for (int i{0}; i <= sample_size; ++i)
        {
            sampled_value = hmc.hamiltonian_monte_carlo(sampled_value, num_steps, step_size);
            std::cout << sampled_value << "\t" << 0 << "\t" << 0 << "\t"
                      << "True" << std::endl;
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

        double initial_position{strtod(argv[2], nullptr)};
        size_t total_iterations{static_cast<size_t>(strtol(argv[3], nullptr, 10))};
        size_t warm_up_iterations{static_cast<size_t>(strtol(argv[4], nullptr, 10))};
        double aim_acceptance_probability{strtod(argv[5], nullptr)};

        Laplace target{100, 30};
        UVStandardNormalSampler momentum_sampler{};
        double position = initial_position;
        NUTS nuts{aim_acceptance_probability, target, momentum_sampler};

        // Eigen::VectorXd position(2);
        // position << initial_position, initial_position;
        // Eigen::MatrixXd sigma(2, 2);
        // sigma << 10, 7, 7, 5;
        // Eigen::VectorXd mean(2);
        // mean << 100, 100;
        // MVN target{mean, sigma};
        // MVStandardNormalSampler momentum_sampler{2};
        // MVNUTS nuts{aim_acceptance_probability, target, momentum_sampler};

        nuts.sample(position, total_iterations, warm_up_iterations);
    }
    else
    {
        std::cout << "first argument must be either 'hmc' or 'nuts'" << std::endl;
        return 1;
    }

    return 0;
}
