# This file contains the implementation of the static, iterative, and match-based pricing (along with their matching policies).

# external packages
import numpy as np
import copy
from gurobipy import *
from collections import defaultdict
import time
import sys
import os

# change the path to the root directory
sys.path.append(os.getcwd())

# internal functions and variables
from lib.instance_data import InstanceData
from lib.utils import TRAINING_SET_KEY, MATCH_KEY, STATIC_KEY
from lib.simulation import (
    Simulator,
    SAMPLE_KEY,
    NO_SAMPLE_KEY,
)
from lib.evaluator import evaluator

THREADS = 8


class MatchPricingAffineMatchingCombined:

    def initialize(self, instance_data, opt_params):
        self.instance_data = instance_data

        self.EPSILON_CUTTING_PLANE = opt_params.get("epsilon_cutting_plane", 1e-6)
        self.bootstrap_rounds = opt_params.get("bootstrap_rounds", 0)
        self.alpha_bound = opt_params.get(
            "alpha_bound", 1
        )  # if 1, add the bounded constraints for alpha
        self.early_stop = opt_params.get(
            "early_stop", True
        )  # if True, stop the cutting plane algorithm when the profit improvement is small
        self.update_disutil_opt = opt_params.get("update_disutil_opt", True)
        self.shrinkage_factor = opt_params.get("shrinkage_factor", 1)
        self.momentum = opt_params.get("momentum", 0.8)
        self.update_disutil_eval = opt_params.get("update_disutil_eval", True)

        # default conversion rate
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        # max detour rate for each rider type (each row)
        max_detour_percent = np.max(detour_percent_matrix, axis=1)
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        max_disutility = beta0 + beta1 * max_detour_percent
        self.static_conversion = (1 - max_disutility) / 2

    def solve_ALP(self, sampled_states, conversions_init, match_flag):
        """
        Solve the ALP for the given sampled states and initial conversion rates
        """

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_matrix = self.instance_data.request_length_matrix
        request_length_vector = self.instance_data.request_length_vector
        shared_length = self.instance_data.shared_length
        active_disutility = self.instance_data.active_disutility
        passive_disutility = self.instance_data.passive_disutility
        disutility = self.instance_data.disutility
        detour_percent_matrix = (
            self.instance_data.detour_percent_matrix
        )  # detour in percentage

        # set up the model
        m = Model("ALP")
        m.Params.OutputFlag = 0  # no output
        m.Params.method = 2  # barrier
        m.Params.Crossover = 0  # no crossover
        m.Params.Seed = 1  # set random seed
        m.Params.Threads = THREADS
        m.Params.OptimalityTol = 0.01

        # Variables
        # g (profit obejctive in ALP)
        profit_var = m.addVar(obj=1)
        # alpha(i) (affine weights in ALP)
        request_weights_vars = defaultdict()
        for i in range(n_rider_types):
            if self.alpha_bound:
                request_weights_vars[i] = m.addVar(
                    lb=0, ub=request_length_vector[i] * c
                )
            else:
                request_weights_vars[i] = m.addVar(lb=0)
        # Gamma(i, s)
        Gamma_vars = defaultdict(defaultdict)
        for state in sampled_states:
            for i in range(n_rider_types):
                Gamma_vars[state][i] = m.addVar(lb=-GRB.INFINITY)
        # Psi(i, s)
        Psi_vars = defaultdict(defaultdict)
        for state in sampled_states:
            for i in range(n_rider_types):
                Psi_vars[state][i] = m.addVar(lb=-GRB.INFINITY)

        # obj
        m.modelSense = GRB.MINIMIZE

        # constraints
        for state in sampled_states:

            if match_flag:
                # active and passive riders have different disutility expectations
                # NOTE: conversions are changed here: from request-based disutility to match-based disutility

                # constraints 1: g >= expr
                expr = LinExpr()
                for i in range(n_rider_types):
                    # sum Lambda_i * Gamma_(i,s)
                    expr.add(Gamma_vars[state][i], arrival_rates[i])
                    # sum theta * s_i * alpha_i
                    expr.add(request_weights_vars[i], thetas[i] * state[i])
                    # sum (- theta * s_i * c * l_i)
                    expr.addConstant(
                        -thetas[i] * state[i] * c * request_length_vector[i]
                    )

                m.addConstr(profit_var >= expr)

                # constraints 2: Gamma_(i,s) >= value for matching actions
                # Gamma_(i,s) >= lambda_i^M * ((1 - passive_disutility_i - lambda_i^M) * l_i + Psi_(i,s))
                for i in range(n_rider_types):
                    price_match = 1 - passive_disutility[i] - conversions_init[state][i]
                    m.addConstr(
                        Gamma_vars[state][i]
                        >= conversions_init[state][i]
                        * (price_match * request_length_vector[i] + Psi_vars[state][i])
                    )

                # constraints 3: Psi_(i,s) >= value for each waiting request j in each state s
                match_options = np.nonzero(state)[0]
                for i in range(n_rider_types):
                    match_options_i = match_options[
                        (shared_length[i, match_options] > 0)
                        & (
                            detour_percent_matrix[i, match_options]
                            < self.instance_data.max_detour_rate
                        )
                        & (
                            detour_percent_matrix[match_options, i]
                            < self.instance_data.max_detour_rate
                        )
                    ]
                    for j in match_options_i:
                        m.addConstr(
                            Psi_vars[state][i]
                            >= request_weights_vars[j] - c * request_length_matrix[i, j]
                        )

                # constraints 4: Gamma_(i,s) >= value for non-matching actions
                for i in range(n_rider_types):
                    price_wait = 1 - active_disutility[i] - conversions_init[state][i]
                    m.addConstr(
                        Gamma_vars[state][i]
                        >= conversions_init[state][i]
                        * (
                            price_wait * request_length_vector[i]
                            - request_weights_vars[i]
                        )
                    )

            else:
                # in the heuristic, the active and passive riders have the same disutility expectations -- the average disutility overall

                # constraints 1: g >= expr
                expr = LinExpr()
                for i in range(n_rider_types):
                    expr.add(
                        Gamma_vars[state][i],
                        arrival_rates[i] * conversions_init[state][i],
                    )
                    expr.add(request_weights_vars[i], thetas[i] * state[i])
                    expr.addConstant(
                        arrival_rates[i]
                        * conversions_init[state][i]
                        * (1 - disutility[i] - conversions_init[state][i])
                        * request_length_vector[i]
                        - thetas[i] * state[i] * c * request_length_vector[i]
                    )

                m.addConstr(profit_var >= expr)

                # constraints 2: Gamma_(i,s) >= value for matching actions
                match_options = np.nonzero(state)[0]

                for i in range(n_rider_types):
                    match_options_i = match_options[
                        (shared_length[i, match_options] > 0)
                        & (
                            detour_percent_matrix[i, match_options]
                            < self.instance_data.max_detour_rate
                        )
                        & (
                            detour_percent_matrix[match_options, i]
                            < self.instance_data.max_detour_rate
                        )
                    ]
                    for j in match_options:
                        m.addConstr(
                            Gamma_vars[state][i]
                            >= request_weights_vars[j] - c * request_length_matrix[i, j]
                        )

                    # constraints 3: Gamma_(i,s) >= value for non-matching actions
                    m.addConstr(Gamma_vars[state][i] >= -request_weights_vars[i])

        # optimize
        m.optimize()
        # save the model
        self.m = m
        self.profit_var = profit_var
        self.request_weights_vars = request_weights_vars
        self.Gamma_vars = Gamma_vars
        self.Psi_vars = Psi_vars

        # return the weights, Gamma, and profit values
        weights = np.array([request_weights_vars[i].X for i in range(n_rider_types)])
        Gamma = defaultdict()
        for state in sampled_states:
            Gamma[state] = np.array(
                [Gamma_vars[state][i].X for i in range(n_rider_types)]
            )
        Psi = defaultdict()
        for state in sampled_states:
            Psi[state] = np.array([Psi_vars[state][i].X for i in range(n_rider_types)])
        profit = m.objVal

        return weights, profit, Gamma, Psi

    def solve_ALP_with_cutting_plane(
        self, sampled_states, conversions_init, match_flag=True, verbose=False
    ):
        """Solve ALP with cutting plane algorithm, given the sampled states and initial conversions."""

        # solve the initial relaxed LP
        start = time.time()
        weights, profit, Gamma, Psi = self.solve_ALP(
            sampled_states, conversions_init, match_flag
        )
        end = time.time()

        iteration_times = [end - start]  # record the running time of each iteration
        iteration_profits = [profit]  # record the profit of each iteration
        iteration_constraints = [
            self.m.NumConstrs
        ]  # record the number of constraints of each iteration
        iteration_weights = [weights]  # record the weights of each iteration

        cutting_plane_step = 0

        # solve the cutting plane problem, until no violation is found
        while True:
            find_violation, weights, Gamma, Psi, profit, running_time = (
                self.cutting_plane_generation(
                    weights, profit, Gamma, Psi, sampled_states, match_flag
                )
            )

            iteration_times += [running_time]
            iteration_profits += [profit]
            iteration_constraints += [self.m.NumConstrs]
            iteration_weights += [weights]

            if verbose:
                print(
                    "==== Cutting plane iteration {0}: profit = {1} ====".format(
                        cutting_plane_step, profit
                    )
                )
                print(
                    "Profit improvement:", iteration_profits[-1] - iteration_profits[-2]
                )
                print(
                    "Number of new constraints:",
                    iteration_constraints[-1] - iteration_constraints[-2],
                )

            cutting_plane_step += 1

            if not find_violation:
                break

            if self.early_stop and len(iteration_profits) > 1:
                # if iteration_profits[-1], iteration_profits[-2], iteration_profits[-3] are all positive, and the three profits are close enough
                if (
                    iteration_profits[-1] > 0
                    and abs(iteration_profits[-1] - iteration_profits[-2]) < 1e-6
                ):
                    break

        n_iterations = len(iteration_times)
        return (
            weights,
            profit,
            n_iterations,
            iteration_times,
            iteration_profits,
            iteration_constraints,
            iteration_weights,
        )

    def cutting_plane_generation(
        self, weights, profit, Gamma, Psi, sampled_states, match_flag
    ):
        """Find violated cutting planes and add them to the LP."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        request_length_vector = self.instance_data.request_length_vector
        arrival_rates = self.instance_data.arrival_rates
        disutility = self.instance_data.disutility
        active_disutility = self.instance_data.active_disutility
        passive_disutility = self.instance_data.passive_disutility

        # retrieve the model and vars
        m = self.m
        profit_var = self.profit_var
        request_weights_vars = self.request_weights_vars
        Gamma_vars = self.Gamma_vars
        Psi_vars = self.Psi_vars

        # flag for finding violation
        find_violation = 0

        # if use match-based conversion, check the violation for constraints 2 (match) and 4 (non-match).
        if match_flag:

            for state in sampled_states:

                for i in range(n_rider_types):

                    # check constraint 2 -- match
                    # calculate the optimal matching conversion and price for rider i by solving a simple optimization problem
                    opt_match_conversion = (
                        (1 - passive_disutility[i]) * request_length_vector[i]
                        + Psi[state][i]
                    ) / (2 * request_length_vector[i])
                    opt_match_conversion = np.clip(
                        opt_match_conversion, 0, 1 - passive_disutility[i]
                    )
                    opt_match_price = 1 - passive_disutility[i] - opt_match_conversion

                    # calculate the rhs of the constraint
                    match_constraint_rhs = opt_match_conversion * (
                        opt_match_price * request_length_vector[i] + Psi[state][i]
                    )

                    # if violated
                    if (
                        Gamma[state][i]
                        < match_constraint_rhs - self.EPSILON_CUTTING_PLANE
                    ):
                        m.addConstr(
                            Gamma_vars[state][i]
                            >= opt_match_conversion
                            * (
                                opt_match_price * request_length_vector[i]
                                + Psi_vars[state][i]
                            )
                        )
                        find_violation = 1

                    # check constraint 4 -- waiting
                    # calculate the optimal waiting conversion and price for rider i by solving a simple optimization problem
                    opt_wait_conversion = (
                        (1 - active_disutility[i]) * request_length_vector[i]
                        - weights[i]
                    ) / (2 * request_length_vector[i])
                    opt_wait_conversion = np.clip(
                        opt_wait_conversion, 0, 1 - active_disutility[i]
                    )
                    opt_wait_price = 1 - active_disutility[i] - opt_wait_conversion

                    # calculate the rhs of the constraint
                    wait_constraint_rhs = opt_wait_conversion * (
                        opt_wait_price * request_length_vector[i] - weights[i]
                    )

                    # if violated
                    if (
                        Gamma[state][i]
                        < wait_constraint_rhs - self.EPSILON_CUTTING_PLANE
                    ):
                        m.addConstr(
                            Gamma_vars[state][i]
                            >= opt_wait_conversion
                            * (
                                opt_wait_price * request_length_vector[i]
                                - request_weights_vars[i]
                            )
                        )
                        find_violation = 1

        else:
            # active and passive riders have the same disutility expectations

            # compute the optimal conversion in each sampled state
            conversion_opt = defaultdict()
            for state in sampled_states:
                conversion_opt[state] = np.zeros(n_rider_types)
                for i in range(n_rider_types):
                    # use the static pricing policy to find the opt conversions because all riders have the same disutility
                    conversion_opt[state][i] = pricing_policy_in_static_pricing(
                        self.instance_data, weights, state, i
                    )[0]

            # check the violation at each state -- relevant constraint is the first one g >= expr
            for state in sampled_states:

                rhs = 0

                for i in range(n_rider_types):
                    rhs += arrival_rates[i] * conversion_opt[state][i] * (
                        (1 - disutility[i] - conversion_opt[state][i])
                        * request_length_vector[i]
                        + Gamma[state][i]
                    ) + thetas[i] * state[i] * (
                        weights[i] - c * request_length_vector[i]
                    )

                # if violating, then add the constraint
                if rhs > profit + self.EPSILON_CUTTING_PLANE:
                    expr = LinExpr()
                    for i in range(n_rider_types):
                        expr.add(
                            Gamma_vars[state][i],
                            arrival_rates[i] * conversion_opt[state][i],
                        )
                        if state[i] > 0:
                            expr.add(request_weights_vars[i], thetas[i] * state[i])
                        expr.addConstant(
                            arrival_rates[i]
                            * conversion_opt[state][i]
                            * (1 - disutility[i] - conversion_opt[state][i])
                            * request_length_vector[i]
                            - thetas[i] * state[i] * c * request_length_vector[i]
                        )
                    m.addConstr(profit_var >= expr)

                    find_violation = 1

        m.optimize()
        self.m = m
        # print("Cutting plane iteration solved in {0} seconds.".format(m.Runtime))

        # update the weights, Gamma, and profit values
        weights = np.array([request_weights_vars[i].X for i in range(n_rider_types)])
        for state in sampled_states:
            Gamma[state] = np.array(
                [Gamma_vars[state][i].X for i in range(n_rider_types)]
            )
            Psi[state] = np.array([Psi_vars[state][i].X for i in range(n_rider_types)])
        profit = m.objVal

        return find_violation, weights, Gamma, Psi, profit, m.Runtime

    def train_matchbased_pricing_policy(
        self,
        c,
        sojourn_time,
        network,
        rider_types,
        arrival_rates,
        arrival_types,
        arrival_times,
        beta0,
        beta1,
        max_detour_rate,
        policy_opt_params,
        sim_params,
        disutility,
        active_disutility,
        passive_disutility,
        verbose=False,
    ):

        start = time.time()

        instance_data = InstanceData(
            c,
            sojourn_time,
            network,
            rider_types,
            arrival_rates,
            arrival_types,
            arrival_times,
            beta0,
            beta1,
            max_detour_rate,
            disutility,
            active_disutility,
            passive_disutility,
        )

        # initialization
        self.initialize(instance_data, policy_opt_params)

        # create the simulator
        simulator_instance_data = copy.deepcopy(instance_data)
        simulator = Simulator(simulator_instance_data, sim_params, use_real_data=False)

        # bootstrapping for sampling
        bootstrapping_profits = list()  # the profit of each bootstrapping round
        bootstrap_sample_sizes = list()  # the sample size of each bootstrapping round
        weights_list = list()  # the weights optimized at each bootstrapping round
        bootstrap_active_disutilities = list()
        bootstrap_passive_disutilities = list()
        bootstrap_disutilities = list()

        # if there is no affine weights used for warmup, then use the static pricing policy and greedy matching policy for intial sampling
        if not policy_opt_params["warmup"]:

            static_price = 1 - self.instance_data.disutility - self.static_conversion
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, static_price
            )
            greedy_matching_policy = GreedyMatchingPolicy(self.instance_data)

            # states sampling
            simulator.events_generator(time_limit=simulator.time_limit_greedy)
            sampled_states = simulator.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=greedy_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )

            # estimate the disutility under the static pricing policy
            if beta0 > 0 or beta1 > 0:
                simulator.events_generator(time_limit=simulator.evaluation_time)
                metrics, _ = evaluator(
                    STATIC_KEY,
                    static_pricing_policy,
                    greedy_matching_policy,
                    simulator,
                    max_iter=5,
                    verbose=True,
                    shrinkage_factor=self.shrinkage_factor,
                    update_disutility_flag=self.update_disutil_eval,
                )

            # the initial conversion for cutting plane: use the default static conversion
            arbitrary_conversion = self.static_conversion
            affine_weights = None

        else:
            arbitrary_conversion = self.static_conversion

            # estimate the disutility under the warmup weights
            warmup_weights = policy_opt_params["warmup_weights"]
            affine_weights = warmup_weights
            first_policy_instance_data = copy.deepcopy(self.instance_data)
            first_pricing_policy = MatchBasedPricingPolicy(
                first_policy_instance_data, warmup_weights
            )
            first_matching_policy = AffineMatchingPolicy(
                first_policy_instance_data,
                warmup_weights,
                match_flag=first_pricing_policy.match_flag,
            )
            simulator.events_generator(time_limit=simulator.evaluation_time)
            metrics, _ = evaluator(
                MATCH_KEY,
                first_pricing_policy,
                first_matching_policy,
                simulator,
                max_iter=5,
                verbose=verbose,
                shrinkage_factor=self.shrinkage_factor,
                update_disutility_flag=self.update_disutil_eval,
            )

            # update the disutility and evaluate the first policy again
            if self.update_disutil_eval:
                first_policy_instance_data.active_disutility = metrics[-1][
                    "active_disutility_per_type"
                ]
                first_policy_instance_data.passive_disutility = metrics[-1][
                    "passive_disutility_per_type"
                ]
            else:
                first_policy_instance_data.active_disutility = metrics[
                    "active_disutility_per_type"
                ]
                first_policy_instance_data.passive_disutility = metrics[
                    "passive_disutility_per_type"
                ]
            first_pricing_policy = MatchBasedPricingPolicy(
                first_policy_instance_data, warmup_weights
            )
            first_matching_policy = AffineMatchingPolicy(
                first_policy_instance_data,
                warmup_weights,
                match_flag=first_pricing_policy.match_flag,
            )
            # evaluate the warmup policy using the updated disutility
            simulator.events_generator(time_limit=simulator.evaluation_time)
            metrics, _ = evaluator(
                MATCH_KEY,
                first_pricing_policy,
                first_matching_policy,
                simulator,
                max_iter=5,
                verbose=verbose,
                shrinkage_factor=self.shrinkage_factor,
                update_disutility_flag=self.update_disutil_eval,
            )

            # states sampling
            simulator.events_generator(
                time_limit=simulator.time_limits[TRAINING_SET_KEY]
            )
            sampled_states = simulator.simulation(
                pricing_policy=first_pricing_policy,
                matching_policy=first_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )

            if self.update_disutil_opt:
                bootstrapping_profits.append(metrics[-1]["profit"])
            else:
                bootstrapping_profits.append(metrics["profit"])
            bootstrap_active_disutilities.append(
                first_policy_instance_data.active_disutility
            )
            bootstrap_passive_disutilities.append(
                first_policy_instance_data.passive_disutility
            )
            bootstrap_disutilities.append(first_policy_instance_data.disutility)
            weights_list.append(warmup_weights)
            bootstrap_sample_sizes.append(-1)

        # record the optimization process
        # the time for solving the ALP at each bootstrapping round
        ALP_times = list()
        # the number of constraints in the ALP at each bootstrapping round
        ALP_constraints = list()
        # the number of variables in the ALP at each bootstrapping round
        ALP_vars = list()
        # the number of iterations for cutting plane at each bootstrapping round
        cutting_plane_n_iterations = list()
        # the time for each iteration of cutting plane at each bootstrapping round
        cutting_plane_iter_times = list()
        # the profit of each iteration of cutting plane at each bootstrapping round
        cutting_plane_iter_profits = list()
        # the number of constraints of the LP in each iteration of cutting plane at each bootstrapping round
        cutting_plane_iter_constraints = list()
        # the weights solved in each iteration of cutting plane at each bootstrapping round
        cutting_plane_iter_weights = list()

        for iteration in range(self.bootstrap_rounds):
            if verbose:
                print("===== Bootstrapping round: {0} =====".format(iteration))

            if self.update_disutil_opt:
                if self.update_disutil_eval:
                    updated_active_disutility = metrics[-1][
                        "active_disutility_per_type"
                    ]
                    updated_passive_disutility = metrics[-1][
                        "passive_disutility_per_type"
                    ]
                    updated_disutility = metrics[-1]["disutility_per_type"]
                else:
                    updated_active_disutility = metrics["active_disutility_per_type"]
                    updated_passive_disutility = metrics["passive_disutility_per_type"]
                    updated_disutility = metrics["disutility_per_type"]

                bootstrap_active_disutilities.append(updated_active_disutility)
                bootstrap_passive_disutilities.append(updated_passive_disutility)
                bootstrap_disutilities.append(updated_disutility)
                self.instance_data.active_disutility = updated_active_disutility
                self.instance_data.passive_disutility = updated_passive_disutility
                self.instance_data.disutility = updated_disutility

            bootstrap_sample_sizes.append(len(sampled_states))

            # solve the ALP with cutting plane
            # initialize the conversions

            if affine_weights is None:
                conversions_init = conversions_init_for_sampled_states(
                    sampled_states,
                    match_flag=False,
                    static_conversion=arbitrary_conversion,
                )
            else:
                # use the updated weights to calculate the initial conversions at each sampled state
                conversions_init = conversions_init_for_sampled_states(
                    sampled_states,
                    match_flag=True,
                    affine_weights=affine_weights,
                    instance_data=self.instance_data,
                )

            alp_start = time.time()
            (
                affine_weights,
                ALP_obj,
                cp_n_iterations,
                cp_iteration_times,
                cp_iteration_profits,
                cp_iteration_constraints,
                iteration_weights,
            ) = self.solve_ALP_with_cutting_plane(sampled_states, conversions_init)
            alp_end = time.time()
            # print("Time for solving ALP:", np.round(end - start, 1))
            ALP_times.append(alp_end - alp_start)
            ALP_constraints.append(self.m.NumConstrs)
            ALP_vars.append(self.m.NumVars)
            cutting_plane_n_iterations.append(cp_n_iterations)
            cutting_plane_iter_times.append(cp_iteration_times)
            cutting_plane_iter_profits.append(cp_iteration_profits)
            cutting_plane_iter_constraints.append(cp_iteration_constraints)
            cutting_plane_iter_weights.append(iteration_weights)

            weights_list.append(affine_weights)

            # decide the pricing and matching policies
            match_based_pricing_policy = MatchBasedPricingPolicy(
                self.instance_data, affine_weights
            )
            affine_matching_policy = AffineMatchingPolicy(
                self.instance_data,
                affine_weights,
                match_flag=match_based_pricing_policy.match_flag,
            )

            # resampling
            simulator.events_generator(
                time_limit=simulator.time_limits[TRAINING_SET_KEY]
            )
            sampled_states = simulator.simulation(
                pricing_policy=match_based_pricing_policy,
                matching_policy=affine_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )
            if verbose:
                print("Sample size:", len(sampled_states))

            # evaluate
            simulator.events_generator(time_limit=simulator.evaluation_time)
            metrics, _ = evaluator(
                MATCH_KEY,
                match_based_pricing_policy,
                affine_matching_policy,
                simulator,
                max_iter=5,
                verbose=verbose,
                shrinkage_factor=self.shrinkage_factor,
                update_disutility_flag=self.update_disutil_eval,
            )

            if self.update_disutil_eval:
                bootstrapping_profits.append(metrics[-1]["profit"])
            else:
                bootstrapping_profits.append(metrics["profit"])

        if verbose:
            print("Booststrapping profits (in match-based):", bootstrapping_profits)

        # select the weights with the highest profit
        opt_affine_weights_idx = np.argmax(bootstrapping_profits)
        opt_affine_weights = weights_list[opt_affine_weights_idx]
        # affine_sample_size = bootstrap_sample_sizes[opt_affine_weights_idx]

        # get the belief of the disutility used in the optimal policy
        if self.update_disutil_opt:
            # if the disutility is updated over the process, then use the corresponding belief of disutility used in the optimal policy
            opt_active_disutility = bootstrap_active_disutilities[
                opt_affine_weights_idx
            ]
            opt_passive_disutility = bootstrap_passive_disutilities[
                opt_affine_weights_idx
            ]
        else:
            # if the disutility is fixed from the beginning, then use the fixed disutility as the belief of disutility
            opt_active_disutility = self.instance_data.active_disutility
            opt_passive_disutility = self.instance_data.passive_disutility

        end = time.time()
        training_time = end - start

        return (
            opt_affine_weights,
            opt_active_disutility,
            opt_passive_disutility,
            training_time,
        )


class StaticPricingAffineMatchingCombined:

    def initialize(self, instance_data, opt_params):
        self.instance_data = instance_data

        # algorithm parameters
        self.eps_fluid = opt_params.get("eps_fluid", 1e-6)
        self.bootstrap_rounds = opt_params.get("bootstrap_rounds", 0)
        self.DISUTILITY_GAP = opt_params.get("disutility_gap", 1e-3)
        self.alpha_bound = opt_params.get(
            "alpha_bound", 1
        )  # if 1, add the bounded constraints for alpha
        self.update_disutil_opt = opt_params.get("update_disutil_opt", 1)
        self.shrinkage_factor = opt_params.get("shrinkage_factor", 1)
        self.momentum = opt_params.get("momentum", 0.8)
        self.update_disutil_eval = opt_params.get("update_disutil_eval", True)

        # default conversion rate
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        # max detour for each rider type
        max_detour_percent = np.max(detour_percent_matrix, axis=1)
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        max_disutility = beta0 + beta1 * max_detour_percent
        self.static_conversion = (1 - max_disutility) / 2

    def solve_ALP(self, sampled_states, conversions_static, verbose=False):
        """
        Solve the ALP for the given sampled states and fixed conversion rates
        """

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_matrix = self.instance_data.request_length_matrix
        request_length_vector = self.instance_data.request_length_vector
        disutility = self.instance_data.disutility
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        shared_length = self.instance_data.shared_length

        # set up the model
        m = Model("ALP")
        m.Params.OutputFlag = 0  # no output
        m.Params.method = 2  # barrier
        m.Params.Crossover = 0  # no crossover
        m.Params.Seed = 1  # set random seed
        m.Params.Threads = THREADS
        m.Params.OptimalityTol = 0.01
        # m.Params.BarHomogeneous = 1

        # Variables
        # g (profit obejctive in ALP)
        profit_var = m.addVar(obj=1)
        # alpha(i) (affine weights in ALP)
        request_weights_vars = defaultdict()
        for i in range(n_rider_types):
            if self.alpha_bound:
                request_weights_vars[i] = m.addVar(
                    lb=0, ub=request_length_vector[i] * c
                )
            else:
                request_weights_vars[i] = m.addVar(lb=0)
        # Gamma(i, s)
        Gamma_vars = defaultdict(defaultdict)
        for state in sampled_states:
            for i in range(n_rider_types):
                Gamma_vars[state][i] = m.addVar(lb=-GRB.INFINITY)

        # obj
        m.modelSense = GRB.MINIMIZE

        # constraints
        for state in sampled_states:
            # g >= expr
            expr = LinExpr()
            for i in range(n_rider_types):
                expr.add(
                    Gamma_vars[state][i],
                    arrival_rates[i] * conversions_static[i],
                )
                expr.add(request_weights_vars[i], thetas[i] * state[i])
                expr.addConstant(
                    arrival_rates[i]
                    * conversions_static[i]
                    * (1 - disutility[i] - conversions_static[i])
                    * request_length_matrix[i, i]
                    - thetas[i] * state[i] * c * request_length_matrix[i, i]
                )
            m.addConstr(profit_var >= expr)

            # Gamma constraints (the Gamma term has the different sign with the one in paper)
            match_options = np.nonzero(state)[0]

            for i in range(n_rider_types):
                # only consider positive shared length and low detour
                match_options_i = match_options[
                    (shared_length[i, match_options] > 0)
                    & (
                        detour_percent_matrix[i, match_options]
                        < self.instance_data.max_detour_rate
                    )
                    & (
                        detour_percent_matrix[match_options, i]
                        < self.instance_data.max_detour_rate
                    )
                ]
                # Gamma >= V(s-e_j) - c l_ij
                for j in match_options_i:
                    m.addConstr(
                        Gamma_vars[state][i]
                        >= request_weights_vars[j] - c * request_length_matrix[i, j]
                    )

                # Gamma >= V(s+e_i)
                if state[i] == 0:
                    m.addConstr(Gamma_vars[state][i] >= -request_weights_vars[i])

        m.optimize()
        if verbose:
            print("ALP solved in {0} seconds.".format(m.Runtime))

        # save the model
        self.m = m
        self.profit_var = profit_var
        self.request_weights_vars = request_weights_vars
        self.Gamma_vars = Gamma_vars

        # return the weights, Gamma, and profit values
        weights = np.array([request_weights_vars[i].X for i in range(n_rider_types)])
        Gamma = defaultdict()
        for state in sampled_states:
            Gamma[state] = np.array(
                [Gamma_vars[state][i].X for i in range(n_rider_types)]
            )
        profit = m.objVal

        return weights, Gamma, profit

    def solve_fluid(self, conversion, verbose=False):
        """Solve the fluid formulation with fixed static pricing."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        shared_length = self.instance_data.shared_length
        request_length_matrix = self.instance_data.request_length_matrix
        request_length_vector = self.instance_data.request_length_vector
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        max_detour_rate = self.instance_data.max_detour_rate

        m = Model("FluidLP")
        m.Params.OutputFlag = 0
        m.Params.method = 2
        m.Params.Crossover = 0
        m.Params.Threads = THREADS
        # m.Params.OptimalityTol = 0.001

        # variables
        # x[i, j] is the average rate of dispatches between type i active riders with type j passive riders
        x = defaultdict()
        # y[i] is the average rate of solo dispatch of type i riders
        y = defaultdict()

        for i in range(n_rider_types):
            y[i] = m.addVar()

            match_options = np.arange(n_rider_types)
            # only consider positive shared_length and low detour matching options
            match_options = match_options[
                (shared_length[i, match_options] > 0)
                & (detour_percent_matrix[i, match_options] <= max_detour_rate)
                & (detour_percent_matrix[match_options, i] <= max_detour_rate)
            ]

            for j in range(n_rider_types):
                if j in match_options:
                    x[i, j] = m.addVar()
                else:
                    # if j is not in the match options, then bound the variable to 0
                    x[i, j] = m.addVar(ub=0)

        # obj
        obj = QuadExpr()
        for i in range(n_rider_types):
            obj.add(
                arrival_rates[i]
                * request_length_vector[i]
                * conversion[i]
                * (1 - beta0 - conversion[i])
                + (beta0 - c) * request_length_vector[i] * y[i]
            )

            for j in range(n_rider_types):
                obj.add(
                    -(
                        beta1
                        * detour_percent_matrix[i, j]
                        * (request_length_vector[i] + request_length_vector[j])
                        + c * request_length_matrix[i, j]
                    )
                    * x[i, j]
                )
        m.setObjective(obj, GRB.MAXIMIZE)

        # constraints
        for i in range(n_rider_types):
            expr = LinExpr()
            for j in range(n_rider_types):
                expr.add(x[i, j] + x[j, i])

            m.addConstr(
                expr + y[i] - arrival_rates[i] * conversion[i] == 0,
                name="gamma_{0}".format(i),
            )

        for i in range(n_rider_types):
            for j in range(n_rider_types):
                m.addConstr(
                    thetas[i] * x[i, j] <= arrival_rates[j] * conversion[j] * y[i],
                    name="eta_{0}_{1}".format(i, j),
                )

        m.optimize()
        if verbose:
            print("obj:", m.objVal)
        return m, x, y

    def solve_fluid_cost(self, conversion):
        """Solve the fluid formulation with fixed static pricing; the obj is only cost."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        shared_length = self.instance_data.shared_length
        request_length_matrix = self.instance_data.request_length_matrix
        request_length_vector = self.instance_data.request_length_vector
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        max_detour_rate = self.instance_data.max_detour_rate

        m = Model("FluidLP")
        m.Params.OutputFlag = 0
        m.Params.method = 2
        m.Params.Crossover = 0
        m.Params.Threads = THREADS
        # m.Params.OptimalityTol = 1e-8

        # variables
        # x[i, j] is the average rate of dispatches between type i active riders with type j passive riders
        x = defaultdict()
        # y[i] is the average rate of solo dispatch of type i riders
        y = defaultdict()

        for i in range(n_rider_types):
            y[i] = m.addVar()

            match_options = np.arange(n_rider_types)
            # only consider positive shared_length and low detour matching options
            match_options = match_options[
                (shared_length[i, match_options] > 0)
                & (detour_percent_matrix[i, match_options] <= max_detour_rate)
                & (detour_percent_matrix[match_options, i] <= max_detour_rate)
            ]

            for j in range(n_rider_types):
                if j in match_options:
                    x[i, j] = m.addVar()
                else:
                    x[i, j] = m.addVar(ub=0)

        # obj
        obj = QuadExpr()
        for i in range(n_rider_types):
            obj.add((beta0 - c) * request_length_vector[i] * y[i])

            for j in range(n_rider_types):
                obj.add(
                    -(
                        beta1
                        * detour_percent_matrix[i, j]
                        * (request_length_vector[i] + request_length_vector[j])
                        + c * request_length_matrix[i, j]
                    )
                    * x[i, j]
                )
        m.setObjective(obj, GRB.MAXIMIZE)

        # constraints
        for i in range(n_rider_types):
            expr = LinExpr()
            for j in range(n_rider_types):
                expr.add(x[i, j] + x[j, i])

            m.addConstr(
                expr + y[i] - arrival_rates[i] * conversion[i] == 0,
                name="gamma_{0}".format(i),
            )

        for i in range(n_rider_types):
            for j in range(n_rider_types):
                m.addConstr(
                    thetas[i] * x[i, j] <= arrival_rates[j] * conversion[j] * y[i],
                    name="eta_{0}_{1}".format(i, j),
                )

        m.optimize()
        return m, x, y

    def gradient_by_hand(self, i, conversion, previous_obj, epsilon=1e-6):
        """Calculate the cost function's gradient with respect to the ith conversion rate by hand"""

        # add a small epsilon to the ith conversion rate
        new_conversion = conversion.copy()
        new_conversion[i] += epsilon
        # resolve the fluid LP with the new conversion rate
        m, _, _ = self.solve_fluid_cost(new_conversion)
        obj = m.ObjVal

        # calculate the gradient
        gradient_of_i = (obj - previous_obj) / epsilon

        return gradient_of_i

    def static_pricing_mm(self, starting_point=None, verbose=False):
        """MM algorithm for solving the static pricing problem."""

        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_vector = self.instance_data.request_length_vector
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        request_length_vector = self.instance_data.request_length_vector
        detour_percent_matrix = self.instance_data.detour_percent_matrix

        # initialize the step count, conversion, subgradient, and obj
        if starting_point is None:
            conversion = np.ones(n_rider_types) * 0.5
        else:
            conversion = np.ones(n_rider_types) * starting_point
        static_lp_obj = -1e5

        opt_obj = -1e5
        opt_conversion = np.zeros(n_rider_types)
        opt_objs = list()

        step_count = 0

        while True:
            if verbose:
                print("step_count:", step_count)

            # conversion_prev += [conversion]
            obj_prev = static_lp_obj

            # solve the static LP
            m, x, y = self.solve_fluid(conversion, verbose=verbose)
            static_lp_obj = m.ObjVal

            if static_lp_obj > opt_obj:
                opt_obj = static_lp_obj
                opt_conversion = conversion
                opt_objs += [opt_obj]

            if static_lp_obj < obj_prev + self.eps_fluid:
                break

            # get the dual variables of the fluid formulation (gamma and eta)
            gamma = np.zeros(n_rider_types)
            eta = np.zeros((n_rider_types, n_rider_types))
            for i in range(n_rider_types):
                gamma[i] = m.getConstrByName("gamma_{0}".format(i)).Pi
                # if conversion[i] == 0:
                #     print(f'Gamma {i} = {gamma[i]}')
                for j in range(n_rider_types):
                    eta[i, j] = m.getConstrByName("eta_{0}_{1}".format(i, j)).Pi

            y_values = np.zeros(n_rider_types)
            for i in range(n_rider_types):
                y_values[i] = y[i].x

            x_values = np.zeros((n_rider_types, n_rider_types))
            for i in range(n_rider_types):
                for j in range(n_rider_types):
                    x_values[i, j] = x[i, j].x

            # calculate the new direction (with the momentum term)
            gradient = arrival_rates * gamma + np.sum(
                np.outer(arrival_rates, y_values) * eta.T, axis=1
            )

            # correct the gradient for the zero conversion
            # calculate the cost function value before changing the conversion
            m, x, y = self.solve_fluid_cost(conversion)
            previous_obj = m.ObjVal

            # for those rider types with zero conversion, calculate the gradient by hand
            for i in range(n_rider_types):
                if conversion[i] == 0:
                    gradient_by_dual = gradient[i]
                    gradient[i] = self.gradient_by_hand(i, conversion, previous_obj)
                    if verbose:
                        print(
                            f"Rider {i}: conversion = {conversion[i]}, gradient by dual = {gradient_by_dual}, gradient by hand = {gradient[i]}"
                        )

            # update the conversion with the corrected gradient
            conversion = (
                (1 - beta0) * arrival_rates * request_length_vector + gradient
            ) / (2 * arrival_rates * request_length_vector)
            conversion = np.clip(conversion, 0, 1)

            step_count += 1

        # if verbose:
        #     print("opt obj:", opt_obj)

        # match rate and detour rate over all
        # match_rate := 1 - y_values / (arrival_rates * conversion)
        match_rate = np.zeros(n_rider_types)
        match_rate[opt_conversion > 0] = 1 - y_values[opt_conversion > 0] / (
            arrival_rates[opt_conversion > 0] * opt_conversion[opt_conversion > 0]
        )
        unconditional_detour = np.zeros(n_rider_types)
        unconditional_detour[opt_conversion > 0] = (
            np.sum(
                x_values[opt_conversion > 0, :]
                * detour_percent_matrix[opt_conversion > 0, :],
                axis=1,
            )
            + np.sum(
                x_values[:, opt_conversion > 0]
                * detour_percent_matrix[opt_conversion > 0, :].T,
                axis=0,
            )
        ) / (arrival_rates[opt_conversion > 0] * opt_conversion[opt_conversion > 0])

        disutility = beta0 * match_rate + beta1 * unconditional_detour
        detour_rate = unconditional_detour / match_rate
        opt_static_prices = 1 - disutility - opt_conversion

        # match rate and detour for active riders
        # active_match_rate_i := sum_j x[i, j] / (sum_j x[i, j]  + y_i)
        # active_detour_i := sum_j x[i, j] * detour_{ij} / (sum_j x[i, j])
        active_match_rate = np.zeros(n_rider_types)
        active_match_rate[opt_conversion > 0] = np.sum(
            x_values[opt_conversion > 0, :], axis=1
        ) / (
            np.sum(x_values[opt_conversion > 0, :], axis=1)
            + y_values[opt_conversion > 0]
        )
        active_detour_rate = np.zeros(n_rider_types)
        active_detour_rate[opt_conversion > 0] = np.sum(
            x_values[opt_conversion > 0, :]
            * detour_percent_matrix[opt_conversion > 0, :],
            axis=1,
        ) / np.sum(x_values[opt_conversion > 0, :], axis=1)
        active_disutility = active_match_rate * (beta0 + beta1 * active_detour_rate)

        # detour for passive riders
        # passive_detour_rate_i := sum_j x[j, i] * detour_{ij} / (sum_j x[j, i])
        passive_detour_rate = np.zeros(n_rider_types)
        passive_detour_rate[opt_conversion > 0] = np.sum(
            x_values[:, opt_conversion > 0]
            * detour_percent_matrix[opt_conversion > 0, :].T,
            axis=0,
        ) / np.sum(x_values[:, opt_conversion > 0], axis=0)
        passive_disutility = beta0 + beta1 * passive_detour_rate

        results = dict()
        results["opt_static_obj"] = opt_obj
        results["opt_objs"] = opt_objs
        results["opt_static_conversion"] = opt_conversion
        results["opt_static_prices"] = opt_static_prices
        results["x_values"] = x_values
        results["y_values"] = y_values
        results["opt_gamma"] = gamma
        results["opt_eta"] = eta
        results["match_rate"] = match_rate
        results["detour_rate"] = detour_rate
        results["disutility"] = disutility
        results["active_match_rate"] = active_match_rate
        results["active_detour_rate"] = active_detour_rate
        results["active_disutility"] = active_disutility
        results["passive_detour_rate"] = passive_detour_rate
        results["passive_disutility"] = passive_disutility

        return results

    def optimize_static_matching(
        self,
        sampled_states,
        static_prices,
        simulator,
        verbose=False,
    ):
        """Optimize the matching policies with bootstrapping, with fixed static pricing."""

        n_rider_types = self.instance_data.n_rider_types

        static_conversion = 1 - static_prices - self.instance_data.disutility
        static_conversion = np.clip(
            static_conversion, 0, 1 - self.instance_data.disutility
        )

        # bootstrapping
        bootstrapping_sample_sizes = list()
        bootstrapping_profits = list()

        weights_init = np.zeros(n_rider_types)
        this_weights = weights_init
        weights_list = (
            list()
        )  # record all candidate weights for each bootstrapping round\

        bootstrapping_disutilities = list()
        disutility = self.instance_data.disutility

        # update the weights iteratively until convergence
        # if verbose:
        #     print("---------- Fixing static conversion, optimizing matching ----------")
        for it in range(self.bootstrap_rounds):
            if verbose:
                print(
                    f"===== Bootstrapping round (in optimizing static matching): {it} ====="
                )
                print("Sample size:", len(sampled_states))
            weights_prev = this_weights
            bootstrapping_sample_sizes.append(len(sampled_states))

            if self.update_disutil_opt:
                bootstrapping_disutilities.append(disutility)

            start = time.time()
            this_weights, _, _ = self.solve_ALP(sampled_states, static_conversion)
            end = time.time()
            if verbose:
                print("Time for solving ALP:", np.round(end - start, 1))

            weights_list += [this_weights]

            if self.bootstrap_rounds > 1:
                # decide the pricing and matching policies
                policy_instance_data = copy.deepcopy(self.instance_data)
                if self.update_disutil_opt:
                    policy_instance_data.disutility = disutility

                static_pricing_policy = StaticPricingPolicy(
                    policy_instance_data, static_prices
                )
                affine_matching_policy = AffineMatchingPolicy(
                    policy_instance_data,
                    this_weights,
                    match_flag=static_pricing_policy.match_flag,
                )

                start = time.time()

                simulator.events_generator(
                    time_limit=simulator.time_limits[TRAINING_SET_KEY]
                )  # reset the synthetic training set
                sampled_states = simulator.simulation(
                    pricing_policy=static_pricing_policy,
                    matching_policy=affine_matching_policy,
                    data_set_key=TRAINING_SET_KEY,
                    sample_key=SAMPLE_KEY,
                )
                end = time.time()
                if verbose:
                    print("Time for resampling:", np.round(end - start, 1))

                simulator.events_generator(time_limit=simulator.evaluation_time)
                metrics, _ = evaluator(
                    STATIC_KEY,
                    static_pricing_policy,
                    affine_matching_policy,
                    simulator,
                    max_iter=5,
                    verbose=False,
                    update_disutility_flag=self.update_disutil_eval,
                )

                if self.update_disutil_eval:
                    bootstrapping_profits.append(metrics[-1]["profit"])
                    disutility = metrics[-1]["disutility_per_type"]
                else:
                    bootstrapping_profits.append(metrics["profit"])
                    disutility = metrics["disutility_per_type"]

            # check convergence
            if np.all(np.isclose(this_weights, weights_prev, atol=0.01, rtol=0.05)):
                break

        if self.bootstrap_rounds > 1:
            if verbose:
                print(
                    "Booststrapping profits (in optimize_static_matching):",
                    bootstrapping_profits,
                )
            # select the weights with the highest profit
            opt_static_weights_idx = np.argmax(bootstrapping_profits)
            opt_static_weights = weights_list[opt_static_weights_idx]
            static_sample_size = bootstrapping_sample_sizes[opt_static_weights_idx]
            if self.update_disutil_opt:
                opt_disutility = bootstrapping_disutilities[opt_static_weights_idx]
            else:
                opt_disutility = disutility
        else:
            opt_static_weights = this_weights
            static_sample_size = len(sampled_states)
            opt_disutility = disutility

        return (
            opt_static_weights,
            static_sample_size,
            weights_list,
            bootstrapping_sample_sizes,
            bootstrapping_profits,
            opt_disutility,
        )

    def train_matching(
        self,
        simulator,
        static_prices,
        init_disutility,
        max_it=1,  # maximum number of iterations for updating disutility -- =1 as we fix the disutility
        verbose=False,
        warmup_start=False,
        warmup_weights=None,
    ):
        """Fixing the static rices, iteratively update the matching policy by updating the disutility; the initial disutility is given from the fluid LP."""

        if not warmup_start:
            # first sampling through a greedy matching policy
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, static_prices
            )
            greedy_matching_policy = GreedyMatchingPolicy(self.instance_data)
            simulator.events_generator(time_limit=simulator.time_limit_greedy)
            sampled_states = simulator.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=greedy_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )
        else:
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, static_prices
            )
            affine_matching_policy = AffineMatchingPolicy(
                self.instance_data,
                warmup_weights,
                match_flag=static_pricing_policy.match_flag,
            )
            simulator.events_generator(
                time_limit=simulator.time_limits[TRAINING_SET_KEY]
            )
            sampled_states = simulator.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=affine_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )

        it_cnt = 0
        disutility_gap = 1
        disutility = init_disutility
        bootstrapping_weights = list()
        bootstrapping_profits = list()
        bootstrapping_sample_sizes = list()

        while disutility_gap > self.DISUTILITY_GAP and it_cnt < max_it:
            start = time.time()

            disutility_prev = copy.deepcopy(disutility)

            # optimize the matching policy with fixed static pricing
            (
                opt_static_weights,
                static_sample_size,
                weights_list,
                sample_sizes,
                profits,
                opt_disutility,
            ) = self.optimize_static_matching(
                sampled_states, static_prices, simulator, verbose
            )
            bootstrapping_weights.append(weights_list)
            bootstrapping_profits.append(profits)
            bootstrapping_sample_sizes.append(sample_sizes)

            # policies
            policy_instance_data = copy.deepcopy(self.instance_data)
            policy_instance_data.disutility = opt_disutility
            static_pricing_policy = StaticPricingPolicy(
                policy_instance_data, static_prices
            )
            affine_matching_policy = AffineMatchingPolicy(
                policy_instance_data,
                opt_static_weights,
                match_flag=static_pricing_policy.match_flag,
            )

            # resimulation under the new matching policy
            simulator.events_generator(time_limit=simulator.evaluation_time)

            # evaluate using the formal process
            metrics, _ = evaluator(
                STATIC_KEY,
                static_pricing_policy,
                affine_matching_policy,
                simulator,
                max_iter=5,
                verbose=False,
                shrinkage_factor=self.shrinkage_factor,
                update_disutility_flag=self.update_disutil_eval,
            )
            if self.update_disutil_eval:
                disutility = metrics[-1]["disutility_per_type"]
            else:
                disutility = metrics["disutility_per_type"]

            disutility_gap = np.linalg.norm(disutility - disutility_prev)

            end = time.time()

            result = dict()
            result["metrics"] = (
                metrics  # a list -- we should take only the last converged one
            )
            result["disutility_gap"] = disutility_gap
            result["iter_time"] = end - start
            result["static_iter_cnt"] = it_cnt
            result["static_sample_size"] = static_sample_size
            result["opt_static_weights"] = opt_static_weights  # matching policy updated
            result["static_prices"] = static_prices  # fixed pricing policy
            result["bootstrapping_weights"] = bootstrapping_weights
            result["bootstrapping_sample_sizes"] = bootstrapping_sample_sizes
            result["bootstrapping_profits"] = bootstrapping_profits
            # if verbose:
            #     this_metrics = metrics[-1]
            #     print("=== step ", it_cnt, " ===")
            #     print("Profit: ", this_metrics["profit"])
            #     print("ave_quoted_price: ", this_metrics["ave_quoted_price"])
            #     print("Bootstrapping sample size: ", bootstrapping_sample_sizes)
            #     print("Bootstrapping profits: ", bootstrapping_profits)

            it_cnt += 1

        return opt_static_weights, result, opt_disutility

    def train_static_pricing_policy(
        self,
        c,
        sojourn_time,
        network,
        rider_types,
        arrival_rates,
        arrival_types,
        arrival_times,
        beta0,
        beta1,
        max_detour_rate,
        policy_opt_params,
        sim_params,
        verbose=False,
    ):

        start = time.time()

        instance_data = InstanceData(
            c,
            sojourn_time,
            network,
            rider_types,
            arrival_rates,
            arrival_types,
            arrival_times,
            beta0,
            beta1,
            max_detour_rate,
        )

        # initialization
        self.initialize(instance_data, policy_opt_params)

        # solve the fluid model for static prices and disutility
        fluid_result = self.static_pricing_mm(verbose=verbose)
        fluid_disutility = fluid_result["disutility"]
        fluid_active_disutility = fluid_result["active_disutility"]
        fluid_passive_disutility = fluid_result["passive_disutility"]
        fluid_static_prices = fluid_result["opt_static_prices"]
        fluid_profit = fluid_result["opt_static_obj"]
        self.instance_data.disutility = fluid_disutility

        # fixing prices and disutility, training the matching policys
        simulator_instance_data = copy.deepcopy(instance_data)
        simulator = Simulator(simulator_instance_data, sim_params, use_real_data=False)
        opt_static_weights, _, opt_disutility = self.train_matching(
            simulator,
            fluid_static_prices,
            fluid_disutility,
            verbose=verbose,
            warmup_start=policy_opt_params["warmup"],
            warmup_weights=policy_opt_params["warmup_weights"],
        )

        end = time.time()
        training_time = end - start

        return (
            fluid_static_prices,
            fluid_disutility,
            fluid_active_disutility,
            fluid_passive_disutility,
            fluid_profit,
            opt_static_weights,
            opt_disutility,
            training_time,
        )


class IterativePricingAffineMatchingCombined(StaticPricingAffineMatchingCombined):
    """
    This class contains the policy that combines the iterative pricing
    (static pricing) and affine matching policies.
    """

    def initialize(self, instance_data, opt_params):
        self.instance_data = instance_data

        # algorithm parameters
        self.eps_fluid = opt_params.get("eps_fluid", 1e-6)
        self.bootstrap_rounds = opt_params.get("bootstrap_rounds", 0)
        self.DISUTILITY_GAP = opt_params.get("disutility_gap", 1e-3)
        self.alpha_bound = opt_params.get(
            "alpha_bound", 1
        )  # if 1, add the bounded constraints for alpha
        self.update_disutil_opt = opt_params.get("update_disutil_opt", 1)
        self.shrinkage_factor = opt_params.get("shrinkage_factor", 1)
        self.momentum = opt_params.get("momentum", 0.8)
        self.update_disutil_eval = opt_params.get("update_disutil_eval", True)
        # iteration times upper bound
        self.iter_max_iter = opt_params.get("iter_max_iter", 10)

        # default conversion rate
        detour_percent_matrix = self.instance_data.detour_percent_matrix
        # max detour for each rider type
        max_detour_percent = np.max(detour_percent_matrix, axis=1)
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1
        max_disutility = beta0 + beta1 * max_detour_percent
        self.static_conversion = (1 - max_disutility) / 2

    def iterative_pricing(
        self,
        sampled_states,
        simulator,
        price_0=None,  # initial price
        # file_name=None,
        verbose=False,
    ):
        """Iteratively update the conversions, match rate, detour, disutility together."""

        disutility = self.instance_data.disutility

        # initial conversion
        if price_0 is None:
            price_0 = 1 - disutility - self.static_conversion

        # initialize the current conversion
        this_price = price_0
        # record the sample size of each iteration
        iterative_sample_sizes = [len(sampled_states)]
        # record the profit of each iteration
        iterative_profits = []

        for it in range(self.iter_max_iter):
            if verbose:
                print("========== Iteration {0} ==========".format(it))

            # record the conversion of the previous iteration
            price_prev = copy.deepcopy(this_price)
            disutility_prev = disutility.copy()

            # fixing the static conversion, update the affine weights via bootstrapping
            opt_weights, iterative_sample_size, _, _, _, _ = (
                self.optimize_static_matching(
                    sampled_states, this_price, simulator, verbose
                )
            )

            # decide the pricing and matching policies
            static_pricing_policy = StaticPricingPolicy(self.instance_data, this_price)
            affine_match_policy = AffineMatchingPolicy(
                self.instance_data,
                opt_weights,
                match_flag=static_pricing_policy.match_flag,
            )

            # update price and disutility via simulation over a longer time under the evaluation process
            metrics, _ = evaluator(
                STATIC_KEY,
                static_pricing_policy,
                affine_match_policy,
                simulator,
                max_iter=6,
                set_type=TRAINING_SET_KEY,
                shrinkage_factor=self.shrinkage_factor,
                verbose=False,
                momentum_coef=self.momentum,
                update_disutility_flag=self.update_disutil_eval,
            )

            # update disutility, cost, profit
            if self.update_disutil_eval:
                disutility = metrics[-1]["disutility_per_type"]
                ave_cost_per_type = metrics[-1]["ave_cost_per_type"]
                iterative_profits.append(metrics[-1]["profit"])
            else:
                disutility = metrics["disutility_per_type"]
                ave_cost_per_type = metrics["ave_cost_per_type"]
                iterative_profits.append(metrics["profit"])
            self.instance_data.disutility = disutility
            # update the pricing
            this_price = np.clip(
                (1 - disutility + ave_cost_per_type) / 2, 0, 1 - disutility
            )

            # the disutility_prev is the belief of disutility in the previous iteration
            self.disutility_prev = disutility_prev.copy()

            if verbose:
                print("=== Iteration {0} ===".format(it))
                print("profit:", iterative_profits[-1])

            # check convergence
            if np.all(np.isclose(this_price, price_prev, atol=0.001, rtol=0.01)):
                break

            # check if the conversion hits 0 -- price hits 1 - disutility
            elif np.all(np.isclose(this_price, 1 - disutility, atol=0.001, rtol=0.01)):
                break

            # resampling the states, under the updated disutility
            simulator.instance_data.disutility = disutility
            simulator.events_generator(
                time_limit=simulator.time_limits[TRAINING_SET_KEY]
            )
            sampled_states = simulator.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=affine_match_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )
            iterative_sample_sizes.append(len(sampled_states))

        sample_size = iterative_sample_sizes[-1]

        return (
            opt_weights,
            sample_size,
            this_price,
            iterative_profits[-1],
            disutility_prev,
        )

    def train_iterative_pricing_policy(
        self,
        c,
        sojourn_time,
        network,
        rider_types,
        arrival_rates,
        arrival_types,
        arrival_times,
        beta0,
        beta1,
        max_detour_rate,
        policy_opt_params,
        sim_params,
        verbose=False,
    ):

        start = time.time()

        # start at zero disutility
        initial_disutility = np.zeros(len(rider_types))
        instance_data = InstanceData(
            c,
            sojourn_time,
            network,
            rider_types,
            arrival_rates,
            arrival_types,
            arrival_times,
            beta0,
            beta1,
            max_detour_rate,
            disutility=initial_disutility,
        )

        # initialization
        self.initialize(instance_data, policy_opt_params)

        # create a simulator
        simulator_instance_data = copy.deepcopy(instance_data)
        simulator = Simulator(simulator_instance_data, sim_params, use_real_data=False)

        # first iteration under an arbitrary initial static price and greedy matching
        static_price = 1 - initial_disutility - self.static_conversion
        static_pricing_policy = StaticPricingPolicy(self.instance_data, static_price)
        greedy_match_policy = GreedyMatchingPolicy(self.instance_data)
        # sampling via simulation (on a short time)
        simulator.events_generator(time_limit=simulator.time_limit_greedy)
        sampled_states = simulator.simulation(
            pricing_policy=static_pricing_policy,
            matching_policy=greedy_match_policy,
            data_set_key=TRAINING_SET_KEY,
            sample_key=SAMPLE_KEY,
        )
        # resimulate on a longer time to estimate the cost, disutility
        simulator.events_generator(time_limit=simulator.evaluation_time)
        simulator.simulation(
            pricing_policy=static_pricing_policy,
            matching_policy=greedy_match_policy,
            data_set_key=TRAINING_SET_KEY,
            sample_key=NO_SAMPLE_KEY,
            shrinkage_factor=self.shrinkage_factor,
        )
        # estimate the cost
        ave_cost_per_type = simulator.metrics["ave_cost_per_type"]
        # update the initial conversion from the cost
        initial_conversion = np.maximum(
            np.minimum((1 - initial_disutility - ave_cost_per_type) / 2, 1), 0
        )
        # initial price
        initial_price = np.clip(
            1 - initial_disutility - initial_conversion, 0, 1 - initial_disutility
        )
        # estimate the disutility
        self.instance_data.disutility = simulator.metrics["disutility_per_type"]

        # train the iterative pricing
        (
            opt_iterative_weights,
            iterative_sample_size,
            opt_iterative_price,
            _,
            iterative_opt_disutility,
        ) = self.iterative_pricing(
            sampled_states, simulator, price_0=initial_price, verbose=verbose
        )

        end = time.time()
        training_time = end - start

        return (
            opt_iterative_price,
            opt_iterative_weights,
            iterative_opt_disutility,
            training_time,
        )


def conversions_init_for_sampled_states(
    sampled_states,
    match_flag,
    static_conversion=None,
    affine_weights=None,
    instance_data=None,
):
    """Initialize the static conversions for all sampled states."""

    conversions_init = dict()

    if not match_flag:
        # use the same static conversion for all states
        for state in sampled_states:
            conversions_init[state] = static_conversion
    else:
        # use the warmup weights to calculate the initial conversions at each sampled state
        for state in sampled_states:
            conversions_init[state] = np.zeros(instance_data.n_rider_types)
            for i in range(instance_data.n_rider_types):
                conversion = matchbased_pricing_and_matching_policy(
                    instance_data, affine_weights, state, i
                )[1]
                conversion = np.clip(conversion, 0, 1 - instance_data.disutility[i])
                conversions_init[state][i] = conversion

    return conversions_init


""" Match-based pricing and matching policies related functions """


def matching_for_match_in_matchbased_pricing(instance_data, weights, state, i):
    """Calculate Psi -- the matching decision for incoming request i at state s, when it has to be matched with a waiting request if there is any."""

    # given the value functions, return the optimal waiting request to match with incoming request i at state s (condiiton on the incoming rider i is matched upon arrival)
    # by choosing the max alpha_j - c * l_ij
    c = instance_data.c
    detour_percent_matrix = instance_data.detour_percent_matrix
    request_length_matrix = instance_data.request_length_matrix
    shared_length = instance_data.shared_length
    detour_percent_matrix = instance_data.detour_percent_matrix

    match_options = np.nonzero(state)[0]
    # only consider positive shared_length and low detour matching options
    match_options = match_options[
        (shared_length[i, match_options] > 0)
        & (detour_percent_matrix[i, match_options] < instance_data.max_detour_rate)
        & (detour_percent_matrix[match_options, i] < instance_data.max_detour_rate)
    ]

    if len(match_options) > 0:
        # compute the optimal matching request for incoming request i at state s
        matching_values = (
            weights[match_options] - c * request_length_matrix[i, match_options]
        )

        # find the optimal matching action
        matched_request = match_options[np.argmax(matching_values)]
        max_Psi = np.max(matching_values)

        return matched_request, max_Psi

    else:
        # if no waiting requests
        return None, -1e10


def pricing_for_match_in_matchbased_pricing(instance_data, weights, state, i):
    """Calculate optimal conversion, price, and Gamma value"""

    request_length_vector = instance_data.request_length_vector
    passive_disutility = instance_data.passive_disutility

    # get the optimal matching request and value of Psi
    matched_request, max_Psi = matching_for_match_in_matchbased_pricing(
        instance_data, weights, state, i
    )

    # calculate the optimal conversion
    opt_conversion = (
        (1 - passive_disutility[i]) * request_length_vector[i] + max_Psi
    ) / (2 * request_length_vector[i])
    opt_conversion = np.clip(opt_conversion, 0, 1 - passive_disutility[i])
    opt_price = 1 - passive_disutility[i] - opt_conversion

    matching_value = opt_conversion * (opt_price * request_length_vector[i] + max_Psi)

    return matched_request, matching_value, opt_conversion, opt_price


def pricing_for_waiting_in_matchbased_pricing(instance_data, weights, i):
    """The optimal conversion if action is waiting for incoming request at state s"""

    request_length_vector = instance_data.request_length_vector
    active_disutility = instance_data.active_disutility

    # compute the optimal conversion for waiting actions - opt_conversion is a number
    opt_conversion = (
        (1 - active_disutility[i]) * request_length_vector[i] - weights[i]
    ) / (2 * request_length_vector[i])

    # bound the conversion
    opt_conversion = np.clip(opt_conversion, 0, 1 - active_disutility[i])
    opt_price = 1 - active_disutility[i] - opt_conversion

    # waiting_value
    waiting_value = opt_conversion * (opt_price * request_length_vector[i] - weights[i])

    return waiting_value, opt_conversion, opt_price


def matchbased_pricing_and_matching_policy(instance_data, weights, state, i):
    """The optimal matching and pricing decision for incoming request i at state s, given the affine weights"""

    matched_request, max_matching_value, opt_conversion_match, opt_price_match = (
        pricing_for_match_in_matchbased_pricing(instance_data, weights, state, i)
    )

    waiting_value, opt_conversion_wait, opt_price_wait = (
        pricing_for_waiting_in_matchbased_pricing(instance_data, weights, i)
    )

    # compare matching and waiting
    if matched_request is not None and max_matching_value >= waiting_value:
        return matched_request, opt_conversion_match, opt_price_match

    else:
        return None, opt_conversion_wait, opt_price_wait


def matching_policy_in_static_pricing(instance_data, weights, state, i):
    """Calculate Gamma in static pricing, and the optimal matching request or letting the rider wait"""

    # only need to compare the value of Gamma -- irrelevant to conversion and price
    c = instance_data.c
    request_length_matrix = instance_data.request_length_matrix
    shared_length = instance_data.shared_length
    detour_percent_matrix = instance_data.detour_percent_matrix

    # waiting value
    waiting_value = -weights[i]

    # matching value
    match_options = np.nonzero(state)[0]
    # only consider positive shared_length and low detour matching options
    match_options = match_options[
        (shared_length[i, match_options] > 0)
        & (detour_percent_matrix[i, match_options] < instance_data.max_detour_rate)
        & (detour_percent_matrix[match_options, i] < instance_data.max_detour_rate)
    ]

    if len(match_options) > 0:
        # compute the optimal matching request for incoming request i at state s
        matching_values = (
            weights[match_options] - c * request_length_matrix[i, match_options]
        )

        # find the optimal matching request and max matching value
        matched_request = match_options[np.argmax(matching_values)]
        max_matching_value = np.max(matching_values)

        # compare with the waiting value
        if max_matching_value > waiting_value:
            return matched_request, max_matching_value
        else:
            return None, waiting_value
    else:
        return None, waiting_value


def pricing_policy_in_static_pricing(instance_data, weights, state, i):
    """The pricing policy in static pricing"""

    request_length_vector = instance_data.request_length_vector
    disutility = instance_data.disutility

    # optimal matching decision
    _, Gamme_value = matching_policy_in_static_pricing(instance_data, weights, state, i)

    # calculate the optimal conversion
    opt_conversion = ((1 - disutility[i]) * request_length_vector[i] + Gamme_value) / (
        2 * request_length_vector[i]
    )

    # bound the conversion
    opt_conversion = np.clip(opt_conversion, 0, 1 - disutility[i])
    opt_price = 1 - disutility[i] - opt_conversion

    return opt_conversion, opt_price


class PricingPolicy(object):
    """
    This class is a general pricing policy class containing the instance data.
    We need to inherit this class to implement a specific pricing policy.
    """

    def __init__(self, instance_data):
        self.instance_data = instance_data


class MatchingPolicy(object):
    """
    This class is a general matching policy class containing the instance data.
    We nee to inherit this class to implement a specific matching policy.
    """

    def __init__(self, instance_data):
        self.instance_data = instance_data


class MatchBasedPricingPolicy(PricingPolicy):
    """
    This class is the match-based pricing policy.
    """

    def __init__(self, instance_data, weights):
        super().__init__(instance_data)

        # affine weights for match-based pricing
        self.weights = weights
        self.match_flag = True  # = True for match-based pricing -- riders have match-based disutilities and conversions

    def pricing_function(self, request, state):
        """
        The pricing function takes the request type and the state as the input,
        and returns the optimal price for the given request in the given state,
        and a conversion rate based on the belief of disutility.
        """

        _, optimal_conversion, optimal_price = matchbased_pricing_and_matching_policy(
            self.instance_data, self.weights, state, request
        )

        # this optimal_conversion is the expected conversion rate under the belief of disutility of the policy, not the real conversion
        return optimal_conversion, optimal_price


class AffineMatchingPolicy(MatchingPolicy):
    """
    Affine matching policy -- the matching policy is based on the implementation of ALP given a set of affine weights.
    """

    def __init__(self, instance_data, weights, match_flag):
        super().__init__(instance_data)

        # affine weights for affine matching
        self.weights = weights
        self.match_flag = match_flag  # if in match-based pricing, then match_flag = True; if static pricing, match_flag = False

    def matching_function(self, request, state):
        """
        The matching function takes the request type and the state as the input,
        and returns the match option with the given request in the form of a
        single-element list; if there is no match option, then return an empty list.
        The optimal match option is a waiting requst with the highest value-to-go.
        """

        if self.match_flag:
            # matching policy in match-based pricing -- depending on both the value functions and the corresponding match-based optimal prices
            matching_decision = matchbased_pricing_and_matching_policy(
                self.instance_data, self.weights, state, request
            )[0]
        else:
            # matching policy in static pricing -- only depending on the value functions
            matching_decision = matching_policy_in_static_pricing(
                self.instance_data, self.weights, state, request
            )[0]

        return matching_decision


class StaticPricingPolicy(PricingPolicy):
    """
    Static pricing policy -- each rider is quoted a price regardless of the state.
    """

    def __init__(self, instance_data, static_price):
        super().__init__(instance_data)

        self.static_price = static_price
        self.match_flag = False

    def pricing_function(self, request, state):
        """
        The pricing function takes the request type and the state as the input,
        and returns the optimal price for the given request in the given
        state, and a conversion rate based on the belief of disutility.
        Although the state is not used in the static pricing policy, it
        is still required as the input for format consistency.
        """

        optimal_price = self.static_price[request]

        # the optimal_conversion is the expected conversion rate under the belief of disutility of the policy, not the real conversion in simulation
        optimal_conversion = np.clip(
            1 - self.instance_data.disutility[request] - optimal_price,
            0,
            1 - self.instance_data.disutility[request],
        )

        return optimal_conversion, optimal_price


class GreedyMatchingPolicy(MatchingPolicy):
    """
    This class is the greedy matching policy.
    """

    def __init__(self, instance_data):
        super().__init__(instance_data)

    def matching_function(self, request, state):
        """
        The matching function takes the request type and the state as the input,
        and returns the match option with the longest shared length with the given
        request (if there are multiple, then choose the one with the longest trip
        length), in the form of the rider type index; if there is no match option,
        then return None.
        """

        # get the shared length and request length of the instance
        shared_length = self.instance_data.shared_length
        request_length_vector = self.instance_data.request_length_vector

        # potential match options (nonzero rider types) in the current state
        match_options = np.nonzero(state)[0]
        # shared lengths between i and the match options
        shared_lengths = shared_length[request, match_options]

        # if there is a match option, then choose the one with the longest shared length with i
        # (if there are multiple, then choose the one with the longest trip length)
        if len(match_options) > 0 and np.max(shared_lengths) > 0:
            all_j = match_options[
                np.argwhere(shared_lengths == np.max(shared_lengths)).flatten()
            ]
            trip_lengths = request_length_vector[all_j]
            j = all_j[np.argmax(trip_lengths)]
            return j
        else:
            return None
