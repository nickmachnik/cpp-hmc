#include <vector>
#include <iostream>
#include "hmc.h"

int main()
{
    double sampled_value{1.3};
    for (int i{0}; i <= 100; ++i)
    {
        sampled_value = hamiltonian_monte_carlo(sampled_value, 100, 0.0002);
        std::cout << sampled_value << std::endl;
    }

    return 0;
}
