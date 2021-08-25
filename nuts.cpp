#include <math.h>
#include <random>
#include <chrono>
#include "nuts.h"
#include <iostream>
#include <stdlib.h>

NUTS::NUTS(double sigma) : sigma{sigma} {};

// logarithm of the target density function
auto NUTS::log_target_density(double position) -> double
{
    return -0.5 * position * position;
}

// gradient of the target density function
auto NUTS::target_gradient(double position) -> double
{
    return -position;
}

// logarithm of the density function of the auxiliary momentum
auto NUTS::log_momentum_density(double momentum) -> double
{
    return -0.5 * momentum * momentum;
}

// joint density of position and momentum
auto NUTS::joint_density(state w) -> double
{
    return exp(log_momentum_density(w.momentum) + log_target_density(w.position));
}

// joint density of position and momentum
auto NUTS::integration_accuracy_threshold(state w) -> double
{
    return exp(delta_max + log_momentum_density(w.momentum) + log_target_density(w.position));
}

auto NUTS::leapfrog(state w, direction dir = direction::right) -> state
{
    w.momentum += dir * 0.5 * step_size * target_gradient(w.position);
    w.position += dir * step_size * w.momentum;
    w.momentum += dir * 0.5 * step_size * target_gradient(w.position);

    // std::cout << w.position << "\t" << w.momentum << "\t" << step_size << std::endl;

    return w;
}

auto NUTS::step_size_is_reasonable(state w_old, state w_new, double a) -> bool
{
    return pow((joint_density(w_new) / joint_density(w_old)), a) <= pow(2, -a);
}

// see Hoffman and Gelman (2014)
void NUTS::find_reasonable_step_size(double position)
{
    step_size = 1.0;
    state w{position, sample_momentum()};
    state w_new{leapfrog(w)};
    double a{2.0 * ((joint_density(w_new) / joint_density(w)) > 0.5) - 1.0};
    while (!step_size_is_reasonable(w, w_new, a))
    {
        step_size *= pow(2, a);
        w_new = leapfrog(w);
    }
}

auto NUTS::sample_slice_threshold(state w) -> double
{
    double upper_bound{
        exp(
            log_target_density(w.position) +
            log_momentum_density(w.momentum))};

    std::uniform_real_distribution<double> u_dist(0.0, upper_bound);

    return u_dist(rnd_generator);
}

auto NUTS::sample_direction() -> direction
{
    double c{standard_normal(rnd_generator)};

    if (c >= 0)
    {
        return direction::right;
    }

    return direction::left;
}

auto NUTS::sample_momentum() -> double
{
    return standard_normal(rnd_generator);
}

auto NUTS::acceptance_probability(state w_new, state w_old) -> double
{
    double prob{
        exp(
            log_target_density(w_new.position) + log_momentum_density(w_new.momentum) -
            log_target_density(w_old.position) - log_momentum_density(w_old.momentum))};

    if (prob > 1.0)
    {
        return 1.0;
    }

    return prob;
}

auto NUTS::biased_coin_toss(double heads_probability) -> bool
{
    std::bernoulli_distribution dist{heads_probability};

    return dist(rnd_generator);
}

auto NUTS::is_u_turn(state leftmost_w, state rightmost_w) -> bool
{
    double delta_position{rightmost_w.position - leftmost_w.position};
    return ((delta_position * rightmost_w.momentum) < 0) || ((delta_position * leftmost_w.momentum) < 0);
}

auto NUTS::build_tree(const build_tree_params &params) -> build_tree_output
{
    if (params.height == 0)
    {
        build_tree_output output{};
        state w_new{leapfrog(params.initial_tree_w, params.dir)};
        output.leftmost_w = w_new;
        output.rightmost_w = w_new;

        output.n_accepted_states = (params.slice <= joint_density(w_new)) ? 1 : 0;
        output.continue_integration = params.slice < integration_accuracy_threshold(w_new);

        output.acceptance_probability = acceptance_probability(w_new, params.initial_chain_w);
        output.total_states = 1;

        return output;
    }

    build_tree_params sub_tree_params = params;
    sub_tree_params.height -= 1;
    build_tree_output output{build_tree(sub_tree_params)};

    build_tree_output side_tree_output{};
    if (output.continue_integration)
    {
        if (params.dir == direction::left)
        {
            build_tree_params left_tree_params = sub_tree_params;
            left_tree_params.initial_tree_w = output.leftmost_w;
            side_tree_output = build_tree(left_tree_params);
            output.leftmost_w = side_tree_output.leftmost_w;
        }
        else
        {
            build_tree_params right_tree_params = sub_tree_params;
            right_tree_params.initial_tree_w = output.rightmost_w;
            side_tree_output = build_tree(right_tree_params);
            output.rightmost_w = side_tree_output.rightmost_w;
        }
    }

    if (biased_coin_toss(
            (double)side_tree_output.n_accepted_states /
            (double)(output.n_accepted_states + side_tree_output.n_accepted_states)))
    {
        output.sampled_position = side_tree_output.sampled_position;
    }

    output.acceptance_probability += side_tree_output.acceptance_probability;
    output.total_states += side_tree_output.total_states;
    output.n_accepted_states += side_tree_output.n_accepted_states;
    output.continue_integration = side_tree_output.continue_integration &&
                                  !is_u_turn(output.leftmost_w, output.rightmost_w);

    return output;
}

auto NUTS::sample(
    double initial_position,
    size_t total_iterations,
    size_t warm_up_iterations) -> std::vector<double>
{
    // thetas
    std::vector<double> positions{initial_position};
    find_reasonable_step_size(initial_position);
    double mu{log(10 * step_size)};
    double H{0.0};
    int tree_height{0};
    int n_accepted_states{1};
    double alpha{};
    int n_alpha{};
    double step_size_hat{1.0};

    for (size_t m{1}; m < total_iterations; ++m)
    {
        // theta^m = theta^m-1, resample r0
        state initial_w{positions.back(), sample_momentum()};
        // u
        double slice{sample_slice_threshold(initial_w)};
        state leftmost_w = initial_w;
        state rightmost_w = initial_w;
        // s
        bool continue_integration{true};
        build_tree_params new_sub_tree_params{initial_w, slice, direction::right, tree_height, initial_w};

        while (continue_integration)
        {
            // v
            direction v{sample_direction()};

            build_tree_output new_sub_tree{};
            if (v == direction::left)
            {
                new_sub_tree_params.initial_tree_w = leftmost_w;
                new_sub_tree = build_tree(new_sub_tree_params);
                leftmost_w = new_sub_tree.leftmost_w;
            }
            else
            {
                new_sub_tree_params.initial_tree_w = rightmost_w;
                new_sub_tree = build_tree(new_sub_tree_params);
                rightmost_w = new_sub_tree.leftmost_w;
            }
            alpha = new_sub_tree.acceptance_probability;
            n_alpha = new_sub_tree.total_states;

            double transition_probability{
                std::min(1.0, (double)new_sub_tree.n_accepted_states / (double)n_accepted_states)};
            if (new_sub_tree.continue_integration && biased_coin_toss(transition_probability))
            {
                positions.push_back(new_sub_tree.sampled_position);
            }

            n_accepted_states += new_sub_tree.n_accepted_states;
            continue_integration = new_sub_tree.continue_integration &&
                                   !is_u_turn(leftmost_w, rightmost_w);
            ++tree_height;
        }

        std::cout << "iteration: " << m << std::endl;

        if (m <= warm_up_iterations)
        {
            double f = 1.0 / (m + t_0);
            double av_alpha = alpha / double(n_alpha);
            H = (1.0 - f) * H + f * (sigma - av_alpha);
            std::cout << "new H: " << H << std::endl;

            step_size = exp(mu - H * (sqrt(m) / gamma));
            std::cout << "new step size: " << step_size << std::endl;

            double mpk{pow(m, -kappa)};
            step_size_hat = exp((mpk * log(step_size)) + ((1 - mpk) * log(step_size_hat)));
            std::cout << "new step size hat: " << step_size_hat << std::endl;
        }
        else if (m == (warm_up_iterations + 1))
        {
            step_size = step_size_hat;
        }
    }

    return positions;
}