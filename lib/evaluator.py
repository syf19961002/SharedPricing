# This file contains the evaluator function that evaluates the performance of a policy

import numpy as np
from gurobipy import *
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# internal classes, functions, and variables
from lib.utils import (
    TRAINING_SET_KEY,
    TEST_SET_KEY,
    MATCH_KEY,
)
from lib.simulation import (
    NO_SAMPLE_KEY,
    calculate_disutility,
)


def evaluator(
    policy_key,
    pricing_policy,
    matching_policy,
    simulator,
    max_iter=15,
    min_disutility_gap=0.01,
    set_type=TRAINING_SET_KEY,
    verbose=False,
    shrinkage_factor=1,
    n_repeats=10,
    momentum_coef=1,
    update_disutility_flag=True,
    fixed_active_disutility=None,
    fixed_passive_disutility=None,
    fixed_disutility=None,
):

    n_rider_types = pricing_policy.instance_data.n_rider_types
    c = pricing_policy.instance_data.c
    sojourn_time = pricing_policy.instance_data.sojourn_time

    # if we need to update the disutility from sharing and detour (starting from 0)
    if update_disutility_flag:

        # starting point of the disutility -- all zeros
        if policy_key == MATCH_KEY:
            active_disutility = np.zeros(n_rider_types)
            passive_disutility = np.zeros(n_rider_types)
        else:
            disutility = np.zeros(n_rider_types)

        # record the metrics of each iteration
        metrics = list()
        eyeball_dfs = list()

        it_cnt = 0
        disutility_gap = 1

        # if it is training data, we don't need to repeat and sum them up
        if set_type == TRAINING_SET_KEY:

            simulator.events_generator(time_limit=simulator.evaluation_time)

            # iteratively update the disutility
            while it_cnt < max_iter and disutility_gap > min_disutility_gap:

                start = time.time()

                # update the disutility in simulator -- not in the policy because the policies were optimized based on their believed disutilities
                if policy_key == MATCH_KEY:
                    active_disutility_prev = active_disutility.copy()
                    passive_disutility_prev = passive_disutility.copy()

                    simulator.instance_data.active_disutility = active_disutility
                    simulator.instance_data.passive_disutility = passive_disutility
                else:
                    disutility_prev = disutility.copy()
                    simulator.instance_data.disutility = disutility

                # simulate
                simulator.simulation(
                    pricing_policy,
                    matching_policy,
                    data_set_key=set_type,
                    sample_key=NO_SAMPLE_KEY,
                    shrinkage_factor=shrinkage_factor,
                )

                end = time.time()
                simulator.metrics["simulation_time"] = end - start

                metrics.append(simulator.metrics)

                # update the disutility
                if policy_key == MATCH_KEY:
                    raw_active_disutility = simulator.metrics[
                        "active_disutility_per_type"
                    ]
                    raw_passive_disutility = simulator.metrics[
                        "passive_disutility_per_type"
                    ]

                    disutility_gap = np.linalg.norm(
                        raw_active_disutility - active_disutility_prev
                    ) + np.linalg.norm(raw_passive_disutility - passive_disutility_prev)

                    if it_cnt == 0:
                        active_disutility = raw_active_disutility
                        passive_disutility = raw_passive_disutility
                    else:
                        active_disutility = (
                            raw_active_disutility * momentum_coef
                            + active_disutility * (1 - momentum_coef)
                        )
                        passive_disutility = (
                            raw_passive_disutility * momentum_coef
                            + passive_disutility * (1 - momentum_coef)
                        )

                else:
                    raw_disutility = simulator.metrics["disutility_per_type"]

                    disutility_gap = np.linalg.norm(raw_disutility - disutility_prev)

                    if it_cnt == 0:
                        disutility = raw_disutility
                    else:
                        disutility = raw_disutility * momentum_coef + disutility * (
                            1 - momentum_coef
                        )

                if verbose:
                    print(
                        f"================= iteration {it_cnt} - {policy_key} - c={c}, t={sojourn_time} ================="
                    )
                    print("Disutility gap:", disutility_gap)
                    print("Profit:", simulator.metrics["profit"])
                    print("Quoted price:", simulator.metrics["ave_quoted_price"])
                    print("Payment:", simulator.metrics["ave_payment"])
                    print("Match rate:", simulator.metrics["match_rate"])
                    print("cost_efficiency:", simulator.metrics["cost_efficiency"])
                    print("throughput:", simulator.metrics["throughput"])
                    print(
                        "Detour rate over matched:",
                        simulator.metrics["ave_detour_rate_over_matched"],
                    )
                    print(
                        "Disutility:", simulator.metrics["ave_disutility_over_requests"]
                    )

                it_cnt += 1

            eyeball_dfs.append(simulator.eyeball_metrics_df)

        # if it is test data, we need to repeat
        # fix a disutility, and repeat the simulation for n_repeats times
        # then take the average of the metrics over the n repetitions to update the disutility
        else:

            while it_cnt < max_iter and disutility_gap > min_disutility_gap:

                # update the disutility
                if policy_key == MATCH_KEY:
                    active_disutility_prev = active_disutility.copy()
                    passive_disutility_prev = passive_disutility.copy()
                    simulator.instance_data.active_disutility = active_disutility
                    simulator.instance_data.passive_disutility = passive_disutility
                else:
                    disutility_prev = disutility.copy()
                    simulator.instance_data.disutility = disutility

                # metrics of n repetitions in this iteration
                metrics_one_iter = list()
                eyeball_dfs_one_iter = list()

                # elements used to calculate the disutility updated in this iteration
                requests_repeats = list()
                matched_requests_repeats = list()
                total_detour_repeats = list()
                active_requests_repeats = list()
                active_matched_requests_repeats = list()
                active_total_detour_repeats = list()
                passive_requests_repeats = list()
                passive_total_detour_repeats = list()

                # fixing the disutility, repeat the simulation for n_repeats times
                for i in range(n_repeats):
                    simulator.events_generator(
                        time_limit=simulator.time_limits[TEST_SET_KEY], seed=i
                    )

                    # simulate
                    simulator.simulation(
                        pricing_policy,
                        matching_policy,
                        data_set_key=set_type,
                        sample_key=NO_SAMPLE_KEY,
                        shrinkage_factor=shrinkage_factor,
                    )

                    metrics_one_iter.append(simulator.metrics)
                    eyeball_dfs_one_iter.append(simulator.eyeball_metrics_df)

                    this_metrics = simulator.metrics
                    requests_repeats.append(this_metrics["requests_per_type"])
                    matched_requests_repeats.append(
                        this_metrics["matched_requests_per_type"]
                    )
                    total_detour_repeats.append(this_metrics["total_detour_per_type"])
                    active_requests_repeats.append(
                        this_metrics["active_requests_per_type"]
                    )
                    active_matched_requests_repeats.append(
                        this_metrics["active_matched_requests_per_type"]
                    )
                    active_total_detour_repeats.append(
                        this_metrics["active_total_detour_per_type"]
                    )
                    passive_requests_repeats.append(
                        this_metrics["passive_requests_per_type"]
                    )
                    passive_total_detour_repeats.append(
                        this_metrics["passive_total_detour_per_type"]
                    )

                # add the metrics of n repetitions to the list
                metrics.append(metrics_one_iter)

                # n dfs combined in one
                eyeball_df = pd.concat([df for df in eyeball_dfs_one_iter], axis=0)
                eyeball_dfs.append(eyeball_df)

                # sum up the n repetitions
                requests = np.sum(requests_repeats, axis=0)
                matched_requests = np.sum(matched_requests_repeats, axis=0)
                total_detour = np.sum(total_detour_repeats, axis=0)
                active_requests = np.sum(active_requests_repeats, axis=0)
                active_matched_requests = np.sum(
                    active_matched_requests_repeats, axis=0
                )
                active_total_detour = np.sum(active_total_detour_repeats, axis=0)
                passive_requests = np.sum(passive_requests_repeats, axis=0)
                passive_total_detour = np.sum(passive_total_detour_repeats, axis=0)

                # update the disutility
                raw_disutility, raw_active_disutility, raw_passive_disutility = (
                    calculate_disutility(
                        pricing_policy.instance_data,
                        shrinkage_factor,
                        requests,
                        matched_requests,
                        total_detour,
                        active_requests,
                        active_matched_requests,
                        active_total_detour,
                        passive_requests,
                        passive_total_detour,
                    )
                )

                if policy_key == MATCH_KEY:
                    disutility_gap = np.linalg.norm(
                        raw_active_disutility - active_disutility_prev
                    ) + np.linalg.norm(raw_passive_disutility - passive_disutility_prev)

                    # use momentum to update the disutility
                    if it_cnt == 0:
                        active_disutility = raw_active_disutility
                        passive_disutility = raw_passive_disutility
                    else:
                        active_disutility = (
                            raw_active_disutility * momentum_coef
                            + active_disutility * (1 - momentum_coef)
                        )
                        passive_disutility = (
                            raw_passive_disutility * momentum_coef
                            + passive_disutility * (1 - momentum_coef)
                        )
                else:
                    disutility_gap = np.linalg.norm(raw_disutility - disutility_prev)

                    # use momentum to update the disutility
                    if it_cnt == 0:
                        disutility = raw_disutility
                    else:
                        disutility = raw_disutility * momentum_coef + disutility * (
                            1 - momentum_coef
                        )

                if verbose:
                    print(
                        f"================= test data: iteration {it_cnt} - {policy_key} - c={c}, t={sojourn_time} ================="
                    )
                    print("Disutility gap:", disutility_gap)
                    print(
                        "Profit:",
                        np.mean(
                            [metrics_one_iter[i]["profit"] for i in range(n_repeats)]
                        ),
                    )
                    print(
                        "Quoted price:",
                        np.mean(
                            [
                                metrics_one_iter[i]["ave_quoted_price"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "Payment:",
                        np.mean(
                            [
                                metrics_one_iter[i]["ave_payment"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "Match rate:",
                        np.mean(
                            [
                                metrics_one_iter[i]["match_rate"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "cost_efficiency:",
                        np.mean(
                            [
                                metrics_one_iter[i]["cost_efficiency"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "throughput:",
                        np.mean(
                            [
                                metrics_one_iter[i]["throughput"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "Detour rate:",
                        np.mean(
                            [
                                metrics_one_iter[i]["ave_detour_rate_over_matched"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )
                    print(
                        "Disutility:",
                        np.mean(
                            [
                                metrics_one_iter[i]["ave_disutility_over_requests"]
                                for i in range(n_repeats)
                            ]
                        ),
                    )

                it_cnt += 1

    # if we fix the disutility from sharing and detour
    else:
        simulator.instance_data.active_disutility = (
            fixed_active_disutility
            if fixed_active_disutility is not None
            else np.zeros(n_rider_types)
        )
        simulator.instance_data.passive_disutility = (
            fixed_passive_disutility
            if fixed_passive_disutility is not None
            else np.zeros(n_rider_types)
        )
        simulator.instance_data.disutility = (
            fixed_disutility
            if fixed_disutility is not None
            else np.zeros(n_rider_types)
        )

        if set_type == TRAINING_SET_KEY:
            simulator.events_generator(time_limit=simulator.evaluation_time)

            # simulate
            simulator.simulation(
                pricing_policy,
                matching_policy,
                data_set_key=set_type,
                sample_key=NO_SAMPLE_KEY,
                shrinkage_factor=shrinkage_factor,
            )

            metrics = simulator.metrics
            eyeball_dfs = simulator.eyeball_metrics_df

            if verbose:
                print(
                    f"================= fixed disutility - {policy_key} - c={c}, t={sojourn_time} ================="
                )
                print("Profit:", metrics["profit"])
                print("Quoted price:", metrics["ave_quoted_price"])
                print("Payment:", metrics["ave_payment"])
                print("Match rate:", metrics["match_rate"])
                print("cost_efficiency:", metrics["cost_efficiency"])
                print("throughput:", metrics["throughput"])
                print("Detour rate:", metrics["ave_detour_rate_over_matched"])
                print("Disutility:", metrics["ave_disutility_over_requests"])

        else:
            # test data-- repeat n times, and take the average
            metrics_repeats = list()
            eyeball_dfs_repeats = list()

            # fixing the disutility, repeat the simulation for n_repeats times
            for i in range(n_repeats):
                simulator.events_generator(
                    time_limit=simulator.time_limits[TEST_SET_KEY], seed=i
                )

                # simulate
                simulator.simulation(
                    pricing_policy,
                    matching_policy,
                    data_set_key=set_type,
                    sample_key=NO_SAMPLE_KEY,
                    shrinkage_factor=shrinkage_factor,
                )

                metrics_repeats.append(simulator.metrics)
                eyeball_dfs_repeats.append(simulator.eyeball_metrics_df)

            metrics = metrics_repeats
            eyeball_dfs = pd.concat([df for df in eyeball_dfs_repeats], axis=0)

            if verbose:
                print(
                    f"================= test data: fixed disutility - {policy_key} - c={c}, t={sojourn_time} ================="
                )
                print(
                    "Profit:",
                    np.mean([metrics_repeats[i]["profit"] for i in range(n_repeats)]),
                )
                print(
                    "Quoted price:",
                    np.mean(
                        [
                            metrics_repeats[i]["ave_quoted_price"]
                            for i in range(n_repeats)
                        ]
                    ),
                )
                print(
                    "Payment:",
                    np.mean(
                        [metrics_repeats[i]["ave_payment"] for i in range(n_repeats)]
                    ),
                )
                print(
                    "Match rate:",
                    np.mean(
                        [metrics_repeats[i]["match_rate"] for i in range(n_repeats)]
                    ),
                )
                print(
                    "cost_efficiency:",
                    np.mean(
                        [
                            metrics_repeats[i]["cost_efficiency"]
                            for i in range(n_repeats)
                        ]
                    ),
                )
                print(
                    "throughput:",
                    np.mean(
                        [metrics_repeats[i]["throughput"] for i in range(n_repeats)]
                    ),
                )
                print(
                    "Detour rate:",
                    np.mean(
                        [
                            metrics_repeats[i]["ave_detour_rate_over_matched"]
                            for i in range(n_repeats)
                        ]
                    ),
                )

    return metrics, eyeball_dfs
