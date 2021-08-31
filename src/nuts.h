#pragma once
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include "target.h"
#include <math.h>
#include <stdlib.h>

// a position-momentum state
struct State
{
    double position;
    double momentum;

    State() : position{0.0}, momentum{0.0} {};
    State(double position, double momentum) : position{position}, momentum{momentum} {};
};

enum Direction
{
    left = -1,
    right = 1,
};

struct BuildTreeParams
{
    State initial_tree_w;
    double slice;
    Direction dir;
    int height;
    State initial_chain_w;

    BuildTreeParams(State initial_tree_w,
                    double slice,
                    Direction dir,
                    int height,
                    State initial_chain_w) : initial_tree_w{initial_tree_w},
                                             slice{slice},
                                             dir{dir},
                                             height{height},
                                             initial_chain_w{initial_chain_w} {};
};

struct BuildTreeOutput
{
    State leftmost_w;
    State rightmost_w;
    // theta'
    double sampled_position{0.0};
    // n'
    int n_accepted_states{0};
    // s'
    bool continue_integration{true};
    // alpha
    double acceptance_probability{0.0};
    // n_alpha
    double total_states{0};

    BuildTreeOutput() = default;

    BuildTreeOutput(State leftmost_w,
                    State rightmost_w,
                    double sampled_position,
                    int n_accepted_states,
                    bool continue_integration,
                    double acceptance_probability,
                    double total_states) : leftmost_w{leftmost_w},
                                           rightmost_w{rightmost_w},
                                           sampled_position{sampled_position},
                                           n_accepted_states{n_accepted_states},
                                           continue_integration{continue_integration},
                                           acceptance_probability{acceptance_probability},
                                           total_states{total_states} {};
};

// A No-U-Turn Sampler with Dual Averaging.
// Featuring dynamic integration length and automatic step size selection.
template <class T>
class NUTS
{
private:
    const double gamma{0.05};
    const double t_0{10.0};
    const double kappa{0.75};
    const double delta_max{1000};

    Target<T> &target;
    std::mt19937 rnd_generator{static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    double step_size{1.0};
    // desired average acceptance rate
    double sigma;

    double sample_momentum();
    double sample_slice_threshold(State w);
    Direction sample_direction();
    State leapfrog(State w, Direction dir = Direction::right);
    bool step_size_is_reasonable(State w_old, State w_new, double alpha);
    void find_reasonable_step_size(double position);

    double joint_density(State w);
    double log_momentum_density(double momentum);
    double integration_accuracy_threshold(State w);
    double acceptance_probability(State w_new, State w_old);
    bool biased_coin_toss(double heads_probability);
    bool is_u_turn(State leftmost_w, State rightmost_w);
    BuildTreeOutput build_tree(const BuildTreeParams &params);

public:
    NUTS(double sigma, Target<T> &target) : sigma{sigma}, target{target} {};
    std::vector<T> sample(T initial_position, size_t total_iterations, size_t warm_up_iterations);
};

// logarithm of the density function of the auxiliary momentum
template <typename T>
auto NUTS<T>::log_momentum_density(double momentum) -> double
{
    return -0.5 * momentum * momentum;
}

// joint density of position and momentum
template <typename T>
auto NUTS<T>::joint_density(State w) -> double
{
    return exp(log_momentum_density(w.momentum) + target.log_density(w.position));
}

// joint density of position and momentum
template <typename T>
auto NUTS<T>::integration_accuracy_threshold(State w) -> double
{
    return exp(delta_max + log_momentum_density(w.momentum) + target.log_density(w.position));
}

template <typename T>
auto NUTS<T>::leapfrog(State w, Direction dir) -> State
{
    w.momentum += dir * 0.5 * step_size * target.log_density_gradient(w.position);
    w.position += dir * step_size * w.momentum;
    w.momentum += dir * 0.5 * step_size * target.log_density_gradient(w.position);

    std::cout << w.position << "\t" << w.momentum << "\t" << step_size << "\t"
              << "false" << std::endl;

    return w;
}

template <typename T>
auto NUTS<T>::step_size_is_reasonable(State w_old, State w_new, double a) -> bool
{
    return pow((joint_density(w_new) / joint_density(w_old)), a) <= pow(2, -a);
}

// see Hoffman and Gelman (2014)
template <typename T>
void NUTS<T>::find_reasonable_step_size(double position)
{
    step_size = 1.0;
    State w{position, sample_momentum()};
    State w_new{leapfrog(w)};
    double a{2.0 * ((joint_density(w_new) / joint_density(w)) > 0.5) - 1.0};
    while (!step_size_is_reasonable(w, w_new, a))
    {
        step_size *= pow(2, a);
        w_new = leapfrog(w);
    }
}

template <typename T>
auto NUTS<T>::sample_slice_threshold(State w) -> double
{
    double upper_bound{
        exp(
            target.log_density(w.position) +
            log_momentum_density(w.momentum))};

    std::uniform_real_distribution<double> u_dist(0.0, upper_bound);

    return u_dist(rnd_generator);
}

template <typename T>
auto NUTS<T>::sample_direction() -> Direction
{
    double c{standard_normal(rnd_generator)};

    if (c >= 0)
    {
        return Direction::right;
    }

    return Direction::left;
}

template <typename T>
auto NUTS<T>::sample_momentum() -> double
{
    return standard_normal(rnd_generator);
}

template <typename T>
auto NUTS<T>::acceptance_probability(State w_new, State w_old) -> double
{
    double prob{
        exp(
            target.log_density(w_new.position) + log_momentum_density(w_new.momentum) -
            target.log_density(w_old.position) - log_momentum_density(w_old.momentum))};

    if (prob > 1.0)
    {
        return 1.0;
    }

    // if (prob < 0.0)
    // {
    //     std::cout << "negative acceptance probability: " << prob << std::endl;
    //     std::exit(1);
    // }

    return prob;
}

template <typename T>
auto NUTS<T>::biased_coin_toss(double heads_probability) -> bool
{
    std::bernoulli_distribution dist{heads_probability};

    return dist(rnd_generator);
}

template <typename T>
auto NUTS<T>::is_u_turn(State leftmost_w, State rightmost_w) -> bool
{
    double delta_position{rightmost_w.position - leftmost_w.position};
    return ((delta_position * rightmost_w.momentum) < 0) || ((delta_position * leftmost_w.momentum) < 0);
}

template <typename T>
auto NUTS<T>::build_tree(const BuildTreeParams &params) -> BuildTreeOutput
{
    if (params.height == 0)
    {
        State w_new{leapfrog(params.initial_tree_w, params.dir)};

        BuildTreeOutput output{
            // leftmost
            w_new,
            // rightmost
            w_new,
            // sampled position
            w_new.position,
            // n_accepted_states
            (params.slice <= joint_density(w_new)) ? 1 : 0,
            // continue integration
            params.slice < integration_accuracy_threshold(w_new),
            // acceptance prob
            acceptance_probability(w_new, params.initial_chain_w),
            // total states
            1};

        return output;
    }

    BuildTreeParams sub_tree_params = params;
    sub_tree_params.height -= 1;
    BuildTreeOutput output{build_tree(sub_tree_params)};

    BuildTreeOutput side_tree_output{};
    if (output.continue_integration)
    {
        if (params.dir == Direction::left)
        {
            BuildTreeParams left_tree_params = sub_tree_params;
            left_tree_params.initial_tree_w = output.leftmost_w;
            side_tree_output = build_tree(left_tree_params);
            output.leftmost_w = side_tree_output.leftmost_w;
        }
        else
        {
            BuildTreeParams right_tree_params = sub_tree_params;
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

template <typename T>
auto NUTS<T>::sample(
    T initial_position,
    size_t total_iterations,
    size_t warm_up_iterations) -> std::vector<T>
{
    // thetas
    std::vector<double> positions(total_iterations, initial_position);
    find_reasonable_step_size(initial_position);
    double mu{log(10 * step_size)};
    double H{0.0};
    double alpha{};
    int n_alpha{};
    double step_size_hat{1.0};
    bool successful_sample{true};

    for (size_t m{1}; m < total_iterations; ++m)
    {
        if (!successful_sample)
        {
            --m;
        }
        successful_sample = false;

        // j
        int tree_height{0};
        // n
        int n_accepted_states{1};
        // theta^m = theta^m-1, resample r0
        State initial_w{positions[m - 1], sample_momentum()};
        // u
        double slice{sample_slice_threshold(initial_w)};
        State leftmost_w = initial_w;
        State rightmost_w = initial_w;
        // s
        bool continue_integration{true};
        BuildTreeParams new_sub_tree_params{initial_w, slice, Direction::right, tree_height, initial_w};

        while (continue_integration)
        {
            // v
            Direction v{sample_direction()};

            BuildTreeOutput new_sub_tree{};
            if (v == Direction::left)
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
                positions[m] = new_sub_tree.sampled_position;
                successful_sample = true;
            }

            n_accepted_states += new_sub_tree.n_accepted_states;
            continue_integration = new_sub_tree.continue_integration &&
                                   !is_u_turn(leftmost_w, rightmost_w);
            ++tree_height;
        }

        if (m <= warm_up_iterations)
        {
            double f = 1.0 / (m + t_0);
            double av_alpha = alpha / double(n_alpha);
            H = (1.0 - f) * H + f * (sigma - av_alpha);
            step_size = exp(mu - H * (sqrt(m) / gamma));
            double mpk{pow(m, -kappa)};
            step_size_hat = exp((mpk * log(step_size)) + ((1 - mpk) * log(step_size_hat)));
        }
        else if (m == (warm_up_iterations + 1))
        {
            step_size = step_size_hat;
        }

        if (m > warm_up_iterations && successful_sample)
        {
            std::cout << positions[m] << "\t" << 0 << "\t" << step_size << "\t"
                      << "true" << std::endl;
        }
    }

    return positions;
}