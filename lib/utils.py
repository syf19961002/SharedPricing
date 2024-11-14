# This file consists of helper functions and variables.

# external packages
import numpy as np
import pickle
from collections import defaultdict

# data set key
TRAINING_SET_KEY = 1
VALIDATION_SET_KEY = 2
TEST_SET_KEY = 3

# real/synthetic data key
SYN_DATA_KEY = 0
REAL_DATA_KEY = 1

# pricing policy keys
MATCH_KEY = "match"
STATIC_KEY = "static"
ITERATIVE_KEY = "iterative"


def arrive(state, i):
    """Helper function to update state for an arriving rider type i."""
    delta = np.zeros(len(state))
    delta[i] = 1
    return tuple(np.array(state) + delta)


def depart(state, i):
    """Helper function to update state for a departing rider type i."""
    delta = np.zeros(len(state))
    delta[i] = -1
    return tuple(np.array(state) + delta)


def results_saving(
    metrics,  # the final results of metrics under three policies on training/test sets
    eyeball_metrics,  # the detail information for all rider arrivals
    training_time,  # the running time for training
    all_policies_affine_weights,  # the optimal affine weights for all pricing policies
    static_policies_conversions,  # the optimal static conversions for static/iterative pricing policy
    all_policies_sample_sizes,  # the sample sizes for all pricing policies
    static_profit_UB,  # the solution of the fluid formulation for static pricing policy
):
    """Save the solutions and experiment results."""

    data = dict()

    # save the performance metrics
    data["metrics"] = metrics
    data["eyeball_metrics"] = eyeball_metrics

    # save the training time
    data["training_time"] = training_time

    # save the policies' information
    data["all_policies_affine_weights"] = all_policies_affine_weights
    data["static_policies_conversions"] = static_policies_conversions
    data["all_policies_sample_sizes"] = all_policies_sample_sizes
    data["static_profit_UB"] = static_profit_UB

    return data


def import_demand(demand_file):
    """Import demand data from a pickle file."""

    with open(demand_file, "rb") as f:
        data = pickle.load(f)

    rider_types = data["rider_types"]  # list of rider types (origin-destination pairs)
    arrival_rates = data["arrival_rates"]  # arrival rates of rider types
    arrival_events = defaultdict(
        defaultdict
    )  # arrival events of rider types, by training/test set and week index
    arrival_times = defaultdict(
        defaultdict
    )  # arrival times of rider types, by training/test set and week index
    for set_key, set_type in enumerate(["training", "validation", "test"]):
        for week, date in enumerate(data["arrival_types"][set_type]):
            arrival_events[set_key + 1][week] = data["arrival_types"][set_type][date]
            arrival_times[set_key + 1][week] = data["arrival_times"][set_type][date]

    return rider_types, arrival_rates, arrival_events, arrival_times
