# This file contains all policy-related classes.

# external packages
import numpy as np
import copy
from gurobipy import *
from collections import defaultdict
import time
import sys
import os

# internal functions and variables
from lib.utils import arrive, depart, TRAINING_SET_KEY
from lib.simulation import SAMPLE_KEY  # key of doing sampling in a simulation

sys.path.append(os.getcwd() + "/..")


class ALP(object):
    """This class contains the approximate linear program (ALP) model
    and the method to solve it."""

    def __init__(self, instance_data):
        # read instance data
        self.instance_data = instance_data

        # save the LP model info (model and variables), for future quick retrieval
        self.LP_mdl_list = list()  # LP models
        self.avg_profit_var_list = list()  # profit variables
        self.request_weights_vars_list = list()  # affine weights variables
        self.Gamma_vars_list = list()  # Gamma variables

    def solve_ALP(self, sampled_states, conversions_init):
        """Constructs the ALP model under static pricing and sampled states, and solves it."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_matrix = self.instance_data.request_length_matrix

        # construct the ALP model
        m = Model("ApproximateLinearProgram")
        m.Params.OutputFlag = 0  # no output
        m.Params.method = 2  # barrier
        m.Params.Crossover = 0  # no crossover
        m.Params.Seed = 1  # set random seed
        m.Params.Threads = 8

        # Variables
        # g (profit obejctive in ALP)
        avg_profit_var = m.addVar(obj=1, name="profit")

        # alpha (affine weights in ALP)
        request_weights_vars = defaultdict()
        for i in range(n_rider_types):
            request_weights_vars[i] = m.addVar(lb=0)

        # gamma (Gamma in ALP)
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
                    arrival_rates[i] * conversions_init[state][i],
                )
                expr.add(request_weights_vars[i], thetas[i] * state[i])
                expr.addConstant(
                    arrival_rates[i]
                    * conversions_init[state][i]
                    * (1 - conversions_init[state][i])
                    * request_length_matrix[i, i]
                    - thetas[i]
                    * state[i]
                    * c
                    * request_length_matrix[i, i]
                )
            m.addConstr(avg_profit_var >= expr, name="profit_cons")

            # Gamma constraints (the Gamma term has the different sign with the one in paper)
            match_options = np.nonzero(state)[0]

            for i in range(n_rider_types):
                # Gamma >= V(s-e_j) - c l_ij
                for j in match_options:
                    m.addConstr(
                        Gamma_vars[state][i]
                        >= request_weights_vars[j]
                        - c * request_length_matrix[i, j]
                    )

                # Gamma >= V(s+e_i)
                if state[i] == 0:
                    m.addConstr(Gamma_vars[state][i] >= -request_weights_vars[i])

        m.optimize()

        # save the LP model info
        self.LP_mdl_list.append(m)  # LP model
        self.avg_profit_var_list.append(avg_profit_var)  # profit variable
        self.request_weights_vars_list.append(
            request_weights_vars
        )  # affine weights
        self.Gamma_vars_list.append(Gamma_vars)  # Gamma variables

        # return the weights, Gamma, and profit values
        weights = np.array([request_weights_vars[i].X for i in range(n_rider_types)])
        Gamma = defaultdict()
        for state in sampled_states:
            Gamma[state] = np.array(
                [Gamma_vars[state][i].X for i in range(n_rider_types)]
            )
        profit = m.objVal

        return weights, Gamma, profit


class MatchPricingAffineMatchingCombined(ALP):
    """
    This class contains the policy that combines the match-based pricing
    and affine matching policies.
    """

    def __init__(self, instance_data, alg_params):
        super().__init__(instance_data)

        self.new_rows_num_list = list()
        self.EPSILON_CUTTING_PLANE = alg_params["EPSILON_CUTTING_PLANE"]
        self.bootstrapping_rounds = alg_params["bootstrapping_rounds"]
        self.static_conversion = (
            np.ones(self.instance_data.n_rider_types) * alg_params["static_conversion_value"]
        )  # used as the default static pricing

    def solve_ALP_with_cutting_plane(self, sampled_states, conversions_init):
        """Solve ALP with cutting plane algorithm, given the sampled states and initial conversions."""

        # solve the initial relaxed LP
        start = time.time()
        weights, Gamma, profit = self.solve_ALP(sampled_states, conversions_init)
        end = time.time()

        self.new_rows_num_list.append(
            []
        )  # record the number of new rows added in this iteration
        iteration_times = [end - start]  # record the running time of each iteration
        iteration_profits = [profit]  # record the profit of each iteration

        # solve the cutting plane problem, until no violation is found
        while True:
            (
                find_violation,
                weights,
                Gamma,
                profit,
                running_time,
            ) = self.cutting_plane_generation(weights, Gamma, profit, sampled_states)
            iteration_times += [running_time]
            iteration_profits += [profit]
            if not find_violation:
                break

        n_iterations = len(iteration_times)

        return weights, profit, n_iterations, iteration_times, iteration_profits

    def cutting_plane_generation(self, weights, Gamma, profit, sampled_states):
        """Generate violated cutting planes and add them to the LP."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_matrix = self.instance_data.request_length_matrix

        # retrieve the LP model
        LP_relaxed_mdl = self.LP_mdl_list[-1]

        # compute the optimal conversion in each sampled state
        conversion_opt = defaultdict()
        for state in sampled_states:
            conversion_opt[state] = np.zeros(n_rider_types)
            for i in range(n_rider_types):
                conversion, _ = affine_decisions_computer(
                    weights, state, i, self.instance_data
                )
                conversion_opt[state][i] = conversion

        # retrieve the variables
        avg_profit_var = self.avg_profit_var_list[-1]
        request_weights_vars = self.request_weights_vars_list[-1]
        Gamma_vars = self.Gamma_vars_list[-1]

        # compute reduced cost
        exprs = list()  # list to store the rhs of all violated constraints (g >= rhs)

        for state in sampled_states:
            rhs = 0

            for i in range(n_rider_types):
                rhs += arrival_rates[i] * conversion_opt[state][i] * (
                    (1 - conversion_opt[state][i]) * request_length_matrix[i, i]
                    + Gamma[state][i]
                ) + thetas[i] * state[i] * (
                    weights[i] - c * request_length_matrix[i, i]
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
                        * (1 - conversion_opt[state][i])
                        * request_length_matrix[i, i]
                        - thetas[i]
                        * state[i]
                        * c
                        * request_length_matrix[i, i]
                    )
                exprs.append(expr)

        # add all violating constraints to the LP
        LP_relaxed_mdl.addConstrs(
            (avg_profit_var >= exprs[i] for i in range(len(exprs))), name="profit_cons"
        )

        find_violation = 0  # indicator
        if len(exprs) > 1:
            find_violation = 1

        # count new rows number
        new_rows_num = len(exprs)
        self.new_rows_num_list[-1].append(new_rows_num)

        LP_relaxed_mdl.Params.method = 2
        LP_relaxed_mdl.Params.Crossover = 0
        # LP_relaxed_mdl.Params.OutputFlag = 0

        start = time.time()
        LP_relaxed_mdl.optimize()
        end = time.time()
        running_time = end - start
        self.LP_mdl_list[-1] = LP_relaxed_mdl

        # retrieve the weights, Gamma, and profit values
        weights = np.array([request_weights_vars[i].X for i in range(n_rider_types)])
        for state in sampled_states:
            Gamma[state] = np.array(
                [Gamma_vars[state][i].X for i in range(n_rider_types)]
            )
        profit = LP_relaxed_mdl.objVal

        return find_violation, weights, Gamma, profit, running_time

    def training_match(
        self,
        simulator_instance,
        opt_static_obj=0,
        opt_static_conversion=None,
        opt_static_weights=None,
    ):
        """
        Trains the combined match-based pricing and affine matching policy.
        If opt_static_obj > 0, then use the optimal static conversion and weights as
        the baseline policy; otherwise, use the greedy matching as the baseline policy.
        """

        # get the parameters
        set_type = TRAINING_SET_KEY

        # record the running time
        start_time = time.time()

        # initial states sampling under a baseline policy
        # if opt_static_obj <= 0, then use the static pricing + affine matching
        if opt_static_obj > 0:
            # decide the pricing and matching policies
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, opt_static_conversion
            )
            affine_matching_policy = AffineMatchingPolicy(
                self.instance_data, opt_static_weights
            )
            # states sampling
            sampled_states = simulator_instance.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=affine_matching_policy,
                data_set_key=set_type,
                sample_key=SAMPLE_KEY,
            )

            # the initial conversion for cutting plane
            arbitrary_conversion = opt_static_conversion

        # if opt_static_obj <= 0, then use static pricing + greedy matching
        else:
            # decide the pricing and matching policies
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, self.static_conversion
            )
            greedy_matching_policy = GreedyMatchingPolicy(self.instance_data)
            # generate the synthetic training set for greedy matching
            simulator_instance.events_generator(
                time_limit=simulator_instance.time_limit_greedy
            )
            # states sampling
            sampled_states = simulator_instance.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=greedy_matching_policy,
                data_set_key=set_type,
                sample_key=SAMPLE_KEY,
            )

            # the initial conversion for cutting plane
            arbitrary_conversion = self.static_conversion

        # bootstrapping to find the optimal affine weights
        bootstrapping_profits = list()  # record the profit of each bootstrapping round
        bootstrap_sample_sizes = (
            list()
        )  # record the sample size of each bootstrapping round
        simulator_instance.events_generator(
            time_limit=simulator_instance.time_limits[set_type]
        )  # reset the synthetic training set
        weights_list = list()  # reset the weights list of bootstrapping

        for iteration in range(self.bootstrapping_rounds):
            bootstrap_sample_sizes.append(len(sampled_states))

            # cutting plane method to solve the optimal affine weights under the current sampled states
            conversions_init = conversions_init_for_sampled_states(
                sampled_states, arbitrary_conversion
            )
            (
                weights,
                _,
                _,
                _,
                _,
            ) = self.solve_ALP_with_cutting_plane(sampled_states, conversions_init)

            # record the results
            weights_list.append(weights)

            # decide the pricing and matching policies
            match_based_pricing_policy = MatchBasedPricingPolicy(
                self.instance_data, weights
            )
            affine_matching_policy = AffineMatchingPolicy(self.instance_data, weights)
            # resample the states under the current optimal affine weights
            sampled_states = simulator_instance.simulation(
                pricing_policy=match_based_pricing_policy,
                matching_policy=affine_matching_policy,
                data_set_key=set_type,
                sample_key=SAMPLE_KEY,
            )
            # record the profit of the current bootstrapping round
            bootstrapping_profits.append(simulator_instance.metrics_list[-1]["profit"])

        # select the weights with the highest profit
        opt_affine_weights_idx = np.argmax(bootstrapping_profits)
        opt_affine_weights = weights_list[opt_affine_weights_idx]
        affine_sample_size = bootstrap_sample_sizes[opt_affine_weights_idx]

        end_time = time.time()
        training_time = end_time - start_time

        return opt_affine_weights, affine_sample_size, training_time


class StaticPricingAffineMatchingCombined(ALP):
    """
    This class contains the policy that combines the static pricing and
    affine matching policies.
    """

    def __init__(self, instance_data, alg_params):
        super().__init__(instance_data)

        self.EPSILON_SUBGRADIENT = alg_params["EPSILON_SUBGRADIENT"]
        self.bootstrapping_rounds = alg_params["bootstrapping_rounds"]

    def solve_fluid_formulation(self, conversion):
        """Solve the fluid formulation with fixed static pricing."""

        # instance parameters
        c = self.instance_data.c
        thetas = self.instance_data.thetas
        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_matrix = self.instance_data.request_length_matrix

        m = Model("LP")
        m.Params.OutputFlag = 0
        m.Params.method = 2
        m.Params.Crossover = 0
        m.Params.Threads = 8

        # variables
        # x[i, j] is the average rate of dispatches between type i active riders
        # with type j passive riders
        x = defaultdict()
        # x_a[i] is the average rate of solo dispatch of type i riders
        x_a = defaultdict()

        for i in range(n_rider_types):
            x_a[i] = m.addVar()

            for j in range(n_rider_types):
                x[i, j] = m.addVar()

        # obj
        obj = QuadExpr()
        for i in range(n_rider_types):
            obj.add(
                arrival_rates[i]
                * request_length_matrix[i, i]
                * conversion[i]
                * (1 - conversion[i])
                - c * request_length_matrix[i, i] * x_a[i]
            )

            for j in range(n_rider_types):
                obj.add(-c * request_length_matrix[i, j] * x[i, j])
        m.setObjective(obj, GRB.MAXIMIZE)

        # constraints
        for i in range(n_rider_types):
            expr = LinExpr()
            for j in range(n_rider_types):
                expr.add(x[i, j] + x[j, i])

            m.addConstr(
                expr + x_a[i] - arrival_rates[i] * conversion[i] == 0,
                name="gamma_{0}".format(i),
            )

        for i in range(n_rider_types):
            for j in range(n_rider_types):
                m.addConstr(
                    thetas[i] * x[i, j] <= arrival_rates[j] * conversion[j] * x_a[i],
                    name="eta_{0}_{1}".format(i, j),
                )

        m.optimize()
        return m, x, x_a

    def static_pricing_subgradient(
        self, step_size, momentum_coef=0.9, starting_point=1
    ):
        """Solve for the optimal static pricing of the fluid model with
        subgradient method."""

        n_rider_types = self.instance_data.n_rider_types
        arrival_rates = self.instance_data.arrival_rates
        request_length_vector = self.instance_data.request_length_vector

        # best conversion so far
        current_opt_obj = -1e+5
        current_opt_conversion = np.zeros(n_rider_types)
        current_opt_subgradient = np.zeros(n_rider_types)

        # initialize the step count, conversion, subgradient, and obj
        step_count = 0
        conversion = np.ones(n_rider_types) * starting_point
        subgradient = np.zeros(n_rider_types)
        static_lp_obj = 0
        conversion_prev = list()
        opt_obj_prev = list()

        while True:
            # print('step_count:', step_count)
            # print('current step size:', step_size)

            conversion_prev += [conversion]
            opt_obj_prev += [static_lp_obj]
            subgradient_prev = subgradient

            # solve the static LP
            m, x, x_a = self.solve_fluid_formulation(
                conversion
            )  # x is the match flow value, x_a is the abandoning flow value
            static_lp_obj = m.ObjVal

            # get the dual variables of the fluid formulation (gamma and eta)
            gamma = np.zeros(n_rider_types)
            eta = np.zeros((n_rider_types, n_rider_types))
            for i in range(n_rider_types):
                gamma[i] = m.getConstrByName("gamma_{0}".format(i)).Pi
                for j in range(n_rider_types):
                    eta[i, j] = m.getConstrByName("eta_{0}_{1}".format(i, j)).Pi

            x_a_array = np.zeros(n_rider_types)
            for i in range(n_rider_types):
                x_a_array[i] = x_a[i].x

            # calculate the new direction (with the momentum term)
            subgradient = (
                arrival_rates * request_length_vector * (1 - 2 * conversion)
                + arrival_rates * gamma
                + np.sum(np.outer(arrival_rates, x_a_array) * eta.T, axis=1)
                + momentum_coef * subgradient_prev
            )

            if static_lp_obj > current_opt_obj:
                current_opt_obj = static_lp_obj
                current_opt_conversion = conversion
                current_opt_subgradient = subgradient

            # print('static_lp_obj:', static_lp_obj)
            # print('optimal obj so far:', current_opt_obj)

            # control the conversion to be within [0, 1]
            conversion = np.maximum(
                np.minimum(conversion_prev[-1] + subgradient * step_size, 1), 0
            )

            step_count += 1

            # termination rule:
            # if the current objective is close to all 5 objs previous to it, then terminate
            if len(opt_obj_prev) > 6 and np.all(
                np.isclose(
                    [opt_obj_prev[-1]] * 5,
                    opt_obj_prev[-6:-1],
                    atol=self.EPSILON_SUBGRADIENT,
                    rtol=5 * self.EPSILON_SUBGRADIENT,
                )
            ):
                # print('Converged')
                # print('Converged obj = ', current_opt_obj)
                break

            # if the step count is too large and still not converging, then terminate
            if step_count == 5000:
                # print('Stopped at step {0}.'.format(step_count))
                # print('Converged obj = ', current_opt_obj)
                break

            # if oscillate 4 times, then reduce the step size, unless the step size is too small
            if len(opt_obj_prev) > 5 and step_size > 1e-10:
                if (
                    static_lp_obj < opt_obj_prev[-1]
                    and opt_obj_prev[-1] > opt_obj_prev[-2]
                    and opt_obj_prev[-2] < opt_obj_prev[-3]
                    and opt_obj_prev[-3] > opt_obj_prev[-4]
                ):
                    step_size = step_size * 0.8
                    # print('Oscillation. New step size=', step_size)

            # regularly diminish the step size (every 500 steps)
            # until the step size is sufficiently small
            if step_size > 1e-10:
                for n in range(30):
                    if step_count == 500 * n:
                        step_size = step_size * 0.8

        # print('opt_static_obj:', current_opt_obj)

        return current_opt_conversion, current_opt_obj, current_opt_subgradient

    def optimize_static_matching(
        self, simulator_instance, sampled_states, static_conversion
    ):
        """Optimize the matching policies with bootstrapping, with fixed static pricing."""

        n_rider_types = self.instance_data.n_rider_types

        bootstrapping_sample_sizes = list()
        bootstrapping_profits = list()

        weights_init = np.zeros(n_rider_types)
        this_weights = weights_init
        weights_list = (
            list()
        )  # record all candidate weights generated from bootstrapping

        # update the weights iteratively until convergence
        for it in range(self.bootstrapping_rounds):
            weights_prev = this_weights
            bootstrapping_sample_sizes.append(len(sampled_states))
            conversions_init = conversions_init_for_sampled_states(
                sampled_states, static_conversion
            )
            this_weights, _, _ = self.solve_ALP(sampled_states, conversions_init)
            weights_list += [this_weights]

            # decide the pricing and matching policies
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, static_conversion
            )
            affine_matching_policy = AffineMatchingPolicy(
                self.instance_data, this_weights
            )

            sampled_states = simulator_instance.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=affine_matching_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )

            bootstrapping_profits.append(simulator_instance.metrics_list[-1]["profit"])

            # check convergence
            if np.all(np.isclose(this_weights, weights_prev, atol=0.0001, rtol=0.0005)):
                break

        # print('Static_iterative_profits:', bootstrapping_profits)

        return weights_list, bootstrapping_sample_sizes, bootstrapping_profits

    def training_static(self, simulator_instance):
        """Trains the combined static pricing and affine matching policies."""

        # get the parameters
        c = self.instance_data.c
        n_rider_types = self.instance_data.n_rider_types
        set_type = TRAINING_SET_KEY

        # record the running time
        start_time = time.time()

        # Use subgradient method to solve a fluid mathematical program to get the
        # optimal static conversion: if c < 1, use momentum method to speed up
        # (default coefficient = 0.9); if c >= 1, use vanilla subgradient method
        if c < 1:
            momentum_coef = 0.9
        else:
            momentum_coef = 0
        opt_static_conversion, opt_static_obj, _ = self.static_pricing_subgradient(
            step_size=1 / n_rider_types, momentum_coef=momentum_coef
        )  # default initial step size = 1/N
        opt_static_obj = (
            opt_static_obj * 60 / 1609
        )  # transfer the unit to min and mile (originally in second and meter)

        # if the optimal static objective converges to a non-positive value,
        # then the conversion and affine weights are set as 0
        if opt_static_obj <= 0:
            opt_static_conversion = np.zeros(n_rider_types)
            opt_static_weights = np.zeros(n_rider_types)
            static_sample_size = 0

        # otherwise, we fix the static conversion and use boostrapping
        # to find the optimal affine weights as matching policy
        else:
            # initial constraint sampling under a baseline policy
            # (static pricing and greedy matching)
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, opt_static_conversion
            )
            greedy_matching_policy = GreedyMatchingPolicy(self.instance_data)

            sampled_states = simulator_instance.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=greedy_matching_policy,
                data_set_key=set_type,
                sample_key=SAMPLE_KEY,
            )

            # update the matching policy via boostrapping
            (
                static_weights_list,
                static_sample_sizes,
                static_iterative_profits,
            ) = self.optimize_static_matching(
                simulator_instance, sampled_states, opt_static_conversion
            )

            # select the weights with the highest profit
            opt_static_weights_idx = np.argmax(static_iterative_profits)
            opt_static_weights = static_weights_list[opt_static_weights_idx]
            static_sample_size = static_sample_sizes[opt_static_weights_idx]

        # record the running time
        end_time = time.time()
        training_time = end_time - start_time

        return (
            opt_static_conversion,
            opt_static_weights,
            opt_static_obj,
            static_sample_size,
            training_time,
        )


class IterativePricingAffineMatchingCombined(StaticPricingAffineMatchingCombined):
    """
    This class contains the policy that combines the iterative pricing
    (static pricing) and affine matching policies.
    """

    def __init__(self, instance_data, alg_params):
        super().__init__(instance_data, alg_params)

        self.static_conversion = (
            np.ones(self.instance_data.n_rider_types) * alg_params["static_conversion_value"]
        )  # used as the default static pricing

    def iterative_pricing(
        self,
        sampled_states,
        simulator_instance,
        conversion_0=None,  # initial conversion
        max_it=5,  # maximum number of iterations
    ):
        """Iteratively update the conversions and affine weights."""

        n_rider_types = self.instance_data.n_rider_types

        # initial conversion
        if conversion_0 is None:
            conversion_0 = self.static_conversion

        this_conversion = copy.deepcopy(
            conversion_0
        )  # initialize the current conversion
        iterative_sample_sizes = [
            len(sampled_states)
        ]  # record the sample size of each iteration
        iterative_profits = []  # record the profit of each iteration

        for it in range(max_it):
            # record the conversion of the previous iteration
            conversion_prev = copy.deepcopy(this_conversion)

            # fixing the static conversion, update the affine weights via bootstrapping
            (
                weights_list,
                _,
                bootstrapping_profits,
            ) = self.optimize_static_matching(
                simulator_instance, sampled_states, this_conversion
            )

            # select the weights with the highest training profit
            opt_weights_idx = np.argmax(bootstrapping_profits)
            opt_weights = weights_list[opt_weights_idx]

            # decide the pricing and matching policies
            static_pricing_policy = StaticPricingPolicy(
                self.instance_data, this_conversion
            )
            affine_match_policy = AffineMatchingPolicy(self.instance_data, opt_weights)

            # resample, update the cost and pricing
            sampled_states = simulator_instance.simulation(
                pricing_policy=static_pricing_policy,
                matching_policy=affine_match_policy,
                data_set_key=TRAINING_SET_KEY,
                sample_key=SAMPLE_KEY,
            )

            iterative_sample_sizes.append(len(sampled_states))
            ave_cost_per_type = simulator_instance.metrics_list[-1][
                "ave_cost_per_type"
            ]  # update the cost
            iterative_profits.append(simulator_instance.metrics_list[-1]["profit"])
            this_conversion = np.maximum(
                np.minimum((1 - ave_cost_per_type) / 2, 1), 0
            )  # update the pricing

            # check convergence
            if np.all(
                np.isclose(this_conversion, conversion_prev, atol=0.0001, rtol=0.0005)
            ):
                break

            # check if the conversion hits 0
            elif np.all(
                np.isclose(
                    this_conversion, np.zeros(n_rider_types), atol=0.0001, rtol=0.0005
                )
            ):
                break

        sample_size = iterative_sample_sizes[-1]

        return opt_weights, sample_size, this_conversion, iterative_profits[-1]

    def training_iterative(self, simulator_instance):
        """Trains the combined iterative pricing and affine matching policies."""

        # record the running time
        start_time = time.time()

        # initial sampling under a baseline policy (static pricing and greedy matching)
        static_pricing_policy = StaticPricingPolicy(
            self.instance_data, self.static_conversion
        )
        greedy_match_policy = GreedyMatchingPolicy(self.instance_data)
        sampled_states = simulator_instance.simulation(
            pricing_policy=static_pricing_policy,
            matching_policy=greedy_match_policy,
            data_set_key=TRAINING_SET_KEY,
            sample_key=SAMPLE_KEY,
        )

        # calculate the initial conversion from the cost
        request_ave_cost = simulator_instance.metrics_list[-1]["ave_cost_per_type"]
        initial_conversion = np.maximum(np.minimum((1 - request_ave_cost) / 2, 1), 0)

        # run the iterative pricing algorithm to find the optimal conversion and affine weights
        (
            iterative_weights,
            iterative_sample_size,
            iterative_conversion,
            _,
        ) = self.iterative_pricing(
            sampled_states, simulator_instance, conversion_0=initial_conversion
        )

        # record the running time
        end_time = time.time()
        training_time = end_time - start_time

        return (
            iterative_conversion,
            iterative_weights,
            iterative_sample_size,
            training_time,
        )


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
    This class is the match-based pricing policy. It receives the instance data
    and affine weights in the constructor.
    """

    def __init__(self, instance_data, weights):
        super().__init__(instance_data)

        # affine weights for match-based pricing
        self.weights = weights

    def pricing_function(self, request, state):
        """
        The pricing function takes the request type and the state as the input,
        and returns the optimal conversion for the given request in the given state.
        """

        optimal_conversion, _ = affine_decisions_computer(
            self.weights, state, request, self.instance_data
        )

        return optimal_conversion


class StaticPricingPolicy(PricingPolicy):
    """
    This class is the static pricing policy. It receives the instance data and
    the static conversion in the constructor.
    """

    def __init__(self, instance_data, static_conversion):
        super().__init__(instance_data)

        self.static_conversion = static_conversion

    def pricing_function(self, request, state):
        """
        The pricing function takes the request type and the state as the input,
        and returns the optimal conversion for the given request in the given
        state. Although the state is not used in the static pricing policy, it
        is still required as the input for format consistency.
        """

        return self.static_conversion[request]


class GreedyMatchingPolicy(MatchingPolicy):
    """
    This class is the greedy matching policy. It receive the instance data in
    the constructor.
    """

    def __init__(self, instance_data):
        super().__init__(instance_data)

    def matching_function(self, request, state):
        """
        The matching function takes the request type and the state as the input,
        and returns the match option with the longest shared length with the given
        request (if there are multiple, then choose the one with the longest trip
        length), in the form of a single-element list; if there is no match option,
        then return an empty list.
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
            return [j]
        else:
            return []


class AffineMatchingPolicy(MatchingPolicy):
    """
    This class is the affine matching policy.
    It receive the instance data and affine weights in the constructor.
    """

    def __init__(self, instance_data, weights):
        super().__init__(instance_data)

        # affine weights for affine matching
        self.weights = weights

    def matching_function(self, request, state):
        """
        The matching function takes the request type and the state as the input,
        and returns the match option with the given request in the form of a
        single-element list; if there is no match option, then return an empty list.
        The optimal match option is a waiting requst with the highest value-to-go.
        """

        _, match_option = affine_decisions_computer(
            self.weights, state, request, self.instance_data
        )

        return match_option


# helper function
def conversions_init_for_sampled_states(sampled_states, arbitrary_conversion):
    """Initialize the static conversions for all sampled states."""

    conversions_init = dict()
    for state in sampled_states:
        conversions_init[state] = arbitrary_conversion

    return conversions_init


# helper function for matching and pricing based on affine weights
def affine_decisions_computer(weights, state, i, instance_data):
    """
    Calculates the affine conversion and match option for a type-i request in
    a given state, based on the affine weights.
    """

    c = instance_data.c
    request_length_matrix = instance_data.request_length_matrix
    shared_length = instance_data.shared_length
    satisfy_monge = instance_data.satisfy_monge
    n_rider_types = instance_data.n_rider_types

    # potential match options (nonzero rider types) for type i in the current state
    match_options = np.nonzero(state)[0]
    # not consider the match options that have no shared length with i
    match_options = np.delete(
        match_options, np.where(shared_length[i, match_options] == 0)
    )

    # if type i satisfies the Monge property, then it can match with itself immediately
    # when there is a type i request waiting; otherwise, it can only match with itself
    # when there are two type i requests waiting in the system
    if satisfy_monge[i]:
        self_match = 1  # self match immediately
    else:
        self_match = 2  # self match only when there are two

    # compute all values-to-go
    V_state = np.sum(-weights * state)  # V(s)

    # self match
    if state[i] >= self_match:
        match_option = [i]
        state_after_match = depart(state, i)
        gamma_term = (
            np.sum(-weights * state_after_match) - c * request_length_matrix[i, i]
        )

    # no self match
    else:
        V_wait = np.sum(-weights * arrive(state, i))  # V(s + e_i)

        # compute V(s - e_j) - c * l_ij
        states_after_match = np.array([]).reshape(0, n_rider_types)
        for j in match_options:
            states_after_match = np.vstack(
                (states_after_match, np.array(depart(state, j)))
            )

        V_list = (
            np.sum(-weights * states_after_match, axis=1)
            - c * request_length_matrix[i, match_options]
        )  # V(s - e_j) - c * l_ij

        # if the value of matching is higher, then match; otherwise, wait
        if len(V_list) > 0 and np.max(V_list) > V_wait + 1e-7:
            match_option = [match_options[np.argmax(V_list)]]
            gamma_term = np.max(V_list)
        else:
            match_option = []
            gamma_term = V_wait

    # calculate the optimal conversion with the closed form solution
    conversion = np.minimum(
        np.maximum(
            (request_length_matrix[i, i] - V_state + gamma_term)
            / (2 * request_length_matrix[i, i]),
            0,
        ),
        1,
    )

    return conversion, match_option
