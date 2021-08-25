#pragma once
#include <chrono>
#include <random>
#include <vector>

// a position-momentum state
struct state
{
    double position;
    double momentum;

    state() : position{0.0}, momentum{0.0} {};
    state(double position, double momentum) : position{position}, momentum{momentum} {};
};

enum direction
{
    left = -1,
    right = 1,
};

struct build_tree_params
{
    state initial_tree_w;
    double slice;
    direction dir;
    int height;
    state initial_chain_w;

    build_tree_params(state initial_tree_w,
                      double slice,
                      direction dir,
                      int height,
                      state initial_chain_w) : initial_tree_w{initial_tree_w},
                                               slice{slice},
                                               dir{dir},
                                               height{height},
                                               initial_chain_w{initial_chain_w} {};
};

struct build_tree_output
{
    state leftmost_w;
    state rightmost_w;
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

    build_tree_output(){};

    build_tree_output(state leftmost_w,
                      state rightmost_w,
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
class NUTS
{
private:
    const double gamma{0.05};
    const double t_0{10.0};
    const double kappa{0.75};
    const double delta_max{1000};

    std::mt19937 rnd_generator{std::chrono::steady_clock::now().time_since_epoch().count()};
    std::normal_distribution<double> standard_normal{0.0, 1.0};
    double step_size{1.0};
    // desired average acceptance rate
    double sigma;

    double sample_momentum();
    double sample_slice_threshold(state w);
    direction sample_direction();
    state leapfrog(state w, direction dir);
    bool step_size_is_reasonable(state w_old, state w_new, double alpha);
    void find_reasonable_step_size(double position);

    double joint_density(state w);
    double target_gradient(double position);
    double log_target_density(double position);
    double log_momentum_density(double momentum);
    double integration_accuracy_threshold(state w);
    double acceptance_probability(state w_new, state w_old);
    bool biased_coin_toss(double heads_probability);
    bool is_u_turn(state leftmost_w, state rightmost_w);
    build_tree_output build_tree(const build_tree_params &params);

public:
    explicit NUTS(double sigma);
    std::vector<double> sample(double initial_position, size_t total_iterations, size_t warm_up_iterations);
};
