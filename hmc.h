#pragma once
#include <vector>

const double momentum_mean{0.0};
const double momentum_std{1.0};

double hamiltonian_monte_carlo(double initial_position, int num_steps, double step_size);