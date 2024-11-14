# This file contains the Simulator class to conduct the simulation given pricing and matching policies.

# external packages
import numpy as np
import pandas as pd
from collections import defaultdict
import math

# internal functions and variables
from lib.utils import arrive, depart, TRAINING_SET_KEY, VALIDATION_SET_KEY, TEST_SET_KEY
from lib.instance_data import ABAB, ABBA, BABA, BAAB

# events flag
OBSERVER = 0
ARRIVE = 1
DEPART = 2

# match flags
MATCHED = 1
UNMATCHED = 0
ACTIVE = 1  # arrive earlier
PASSIVE = 2  # arrive later

# key of evaluation on the synthetic training set
EVALUATION_KEY = 1
NOT_EVALUATION_KEY = 0

# key of doing sampling in a simulation or not
SAMPLE_KEY = 1
NO_SAMPLE_KEY = 0

# key of recording durations of states in a simulation
DURATION_KEY = 1
NO_DURATION_KEY = 0


class Simulator(object):
    def __init__(self, instance_data, parameters, use_real_data):
        # read instance data
        self.instance_data = instance_data
        n_rider_types = instance_data.n_rider_types
        arrival_rates = instance_data.arrival_rates
        thetas = instance_data.thetas
        arrival_types = instance_data.arrival_types
        arrival_times = instance_data.arrival_times
        self.observer_rate_portion = parameters.get("observer_rate_portion", 0)

        # simulation parameters
        observer_rate = (sum(arrival_rates) + sum(thetas)) * self.observer_rate_portion
        self.observer_rate = np.array([observer_rate])
        self.time_limits = parameters.get("time_limits")
        self.evaluation_time = parameters.get("evaluation_time")
        self.time_limit_greedy = parameters.get("time_limit_greedy")

        # arrival/departure events definition
        self.outside_event_choices = np.array(
            [(ARRIVE, i) for i in np.arange(n_rider_types)] + [(OBSERVER, 0)]
        )
        self.depart_choices = np.array([(DEPART, i) for i in np.arange(n_rider_types)])

        # data sets
        self.set_types = [TRAINING_SET_KEY, VALIDATION_SET_KEY, TEST_SET_KEY]
        self.data_sets_division = parameters["data_sets_division"][use_real_data]
        self.data_set = {
            TRAINING_SET_KEY: self.data_sets_division[0],
            VALIDATION_SET_KEY: self.data_sets_division[1],
            TEST_SET_KEY: self.data_sets_division[2],
        }

        # if use_real_data = 1, then create `self.arrival_events` and `self.arrival_times` from
        # real data; otherwise, generate synthetic data with the `events_generator` method later
        self.use_real_data = use_real_data
        if use_real_data:
            self.arrival_events = defaultdict(defaultdict)
            for set_type in self.set_types:
                for week in range(self.data_set[set_type]):
                    self.arrival_events[set_type][week] = list(
                        (ARRIVE, i) for i in arrival_types[set_type][week]
                    )
            self.arrival_times = arrival_times

        # arrivals and conversions
        self.outside_arrival_time = defaultdict(defaultdict)
        self.outside_arrival_events = defaultdict(defaultdict)
        self.cdf_inverses = defaultdict(defaultdict)
        self.convert_events = defaultdict(defaultdict)
        self.observer_arrival_time = defaultdict(defaultdict)

    def events_generator(self, time_limit, seed=0):
        """
        Generates events through the given simulation time limit,
        including arrivals, departures, conversions, and synthetic observers.
        The arrivals can be real or synthetic.
        """

        arrival_rates = self.instance_data.arrival_rates

        np.random.seed(seed)

        # save the simulation time limit of the current data set
        self.current_time_limit = time_limit

        # generate events for each data set and week
        for set_type in self.set_types:
            for week in range(self.data_set[set_type]):
                # Get real arrivals; generate synthetic observers
                if self.use_real_data:
                    # Get real arrival events and times
                    eyeball_events_times = list(
                        zip(
                            self.arrival_times[set_type][week],
                            self.arrival_events[set_type][week],
                        )
                    )

                    # Generate observer arrivals.
                    # To speed up the np.exponential() call, we generate 10x more than mean,
                    # then truncate past the time limit
                    observer_interarrival_times = np.random.exponential(
                        scale=1 / self.observer_rate,
                        size=int(time_limit * self.observer_rate * 10),
                    )
                    observer_arrival_time = np.cumsum(observer_interarrival_times)
                    if observer_arrival_time[-1] < time_limit:
                        raise Exception(
                            "The observer number is not enough for the time horizon."
                        )
                    observer_arrival_time = observer_arrival_time[
                        observer_arrival_time < time_limit
                    ]  # truncate events past time limit
                    self.observer_arrival_time[set_type][week] = observer_arrival_time
                    # print('observer number:', len(self.observer_arrival_time[set_type][week]))

                    # Get all arrival event times, sorted by time
                    observer_arrival_events_times = [
                        (t, (OBSERVER, 0)) for t in observer_arrival_time
                    ]
                    outside_events_times = (
                        eyeball_events_times + observer_arrival_events_times
                    )
                    outside_events_times.sort()

                    # Now we save all arrival events and times separately:
                    # The `outside_events_times` is a list of tuples
                    # `(arrival_time, (ARRIVE, type-i request))` or `(arrival_time, (OBSERVER, 0))`,
                    # denoting the arrival events and times of all eyeballs and observers.
                    # The following lines of codes unpack the list `outside_events_times`
                    # into two lists of arrival times and events.
                    # For example, `outside_events_times` = [(1, (ARRIVE, 3), (2, (ARRIVE, 10)), (4, (OBSERVER, 0)), ...].
                    # the first list has all arrival times, such as [1, 2, 4, ...];
                    # the second list includes tuples representing the arrival events,
                    # such as [(ARRIVE, 3), (ARRIVE, 10), (OBSERVER, 0), ...].
                    self.outside_arrival_time[set_type][week] = np.array(
                        list(zip(*outside_events_times))[0]
                    )
                    self.outside_arrival_events[set_type][week] = np.array(
                        list(zip(*outside_events_times))[1]
                    )

                # Generate synthetic arrivals and observers
                else:
                    # Generate all arrival event times.
                    # To speed up the np.exponential() call, we generate 1.5x more than mean,
                    # then truncate past the time limit.
                    total_outside_arrival_rate = (
                        np.sum(arrival_rates) + self.observer_rate
                    )
                    outside_interarrival_times = np.random.exponential(
                        scale=1 / total_outside_arrival_rate,
                        size=int(time_limit * total_outside_arrival_rate * 1.5),
                    )
                    outside_arrival_time = np.cumsum(outside_interarrival_times)
                    if outside_arrival_time[-1] < time_limit:
                        raise Exception(
                            "The outside arrivals number is not enough for the time horizon."
                        )
                    self.outside_arrival_time[set_type][week] = outside_arrival_time[
                        outside_arrival_time < time_limit
                    ]

                    # Poisson split all events into eyeballs versus observers
                    eyeball_prob = arrival_rates / total_outside_arrival_rate
                    observer_prob = self.observer_rate / total_outside_arrival_rate
                    self.outside_arrival_events[set_type][week] = (
                        self.outside_event_choices[
                            np.random.choice(
                                len(self.outside_event_choices),
                                size=len(self.outside_arrival_time[set_type][week]),
                                p=np.concatenate((eyeball_prob, observer_prob)),
                            )
                        ]
                    )

                # Regardless of real or synthetic data, generate departures and conversions using inversion sampling
                self.cdf_inverses[set_type][week] = list(
                    np.random.rand(2 * len(self.outside_arrival_time[set_type][week]))
                )
                self.convert_events[set_type][week] = list(
                    np.random.rand(len(self.outside_arrival_time[set_type][week]))
                )

    def simulation(
        self,
        pricing_policy,
        matching_policy,
        data_set_key,
        sample_key=NO_SAMPLE_KEY,  # =1 if states are sampled in the simulation, =0 if not
        duration_key=NO_DURATION_KEY,  # =1 if the duration of each state that occurs is recorded, =0 if not
        seed=0,  # random seed
        shrinkage_factor=1,  # factor c in the shrinkage rate = N/(N+c)
    ):
        """Conducts the simulation."""

        # check the InstanceData objects in the policy classes and Simulator are consistent
        # assert self.instance_data.equals(pricing_policy.instance_data)
        # assert self.instance_data.equals(matching_policy.instance_data)

        # retrieve the instance data
        n_rider_types = self.instance_data.n_rider_types
        thetas = self.instance_data.thetas
        c = self.instance_data.c
        beta0 = self.instance_data.beta0
        beta1 = self.instance_data.beta1

        request_length_matrix = self.instance_data.request_length_matrix
        request_length_vector = self.instance_data.request_length_vector
        A_solo_length = self.instance_data.A_solo_length
        A_solo_length_1 = self.instance_data.A_solo_length_1
        B_solo_length = self.instance_data.B_solo_length
        B_solo_length_1 = self.instance_data.B_solo_length_1
        shared_length = self.instance_data.shared_length
        trip_choice = self.instance_data.trip_choice

        # set random seed
        np.random.seed(seed)

        # data set
        set_type = data_set_key
        # number of weeks in the data set (if it is the synthetic data, then there is only one week)
        n_weeks = self.data_set[data_set_key]
        time_limit = self.current_time_limit

        # initialized performance metrics
        profit = 0
        cost = 0
        revenue = 0
        total_eyeballs = 0
        throughput = 0
        matched_riders = 0
        arrivals_per_type = np.zeros(n_rider_types)
        total_cost_per_type = np.zeros(n_rider_types)
        payments = list()  # list of all payments

        # values to calculate the match rate and detour over all requests
        requests_per_type = np.zeros(n_rider_types)  # throughput of each type
        matched_requests_per_type = np.zeros(
            n_rider_types
        )  # for match rate of each type
        total_detour_per_type = np.zeros(
            n_rider_types
        )  # for the expected detour of each type

        # for match rate and detour over active riders (who see two prices and wait)
        active_requests_per_type = np.zeros(n_rider_types)
        active_matched_requests_per_type = np.zeros(n_rider_types)
        active_total_detour_per_type = np.zeros(n_rider_types)

        # for match rate and detour over passive riders (who see one price and gets matched immediately)
        passive_requests_per_type = np.zeros(n_rider_types)
        passive_total_detour_per_type = np.zeros(n_rider_types)

        eyeball_metrics = {
            "rider_type": list(),  # rider type id
            "price": list(),  # quoted price
            "quoted_disutility": list(),  # quoted disutility
            "realized_disutility": list(),  # realized disutility
            "convert": list(),  # =1 if it converts
            "wait": list(),  # =1 if it waits in the system (not immdiately dispatched)
            "matched": list(),  # =1 if it is matched with another request
            "cost": list(),
            "matched_with": list(),  # rider type id
            "active/passive": list(),  # active: 1; passive: 2; solo: 0; not convert: np.nan
            "week": list(),  # week number of the arrival time
            "arrival_time": list(),  # arrival time (in seconds)
            "dispatch_time": list(),  # dispatch time (in seconds)
            "wait_dist_before_pickup": list(),  # in meters; after dispatching, before pickup
            "detour": list(),  # in meters
            "detour_rate": list(),  # in rate
        }

        if sample_key:
            # if do sampling, initialize the list of observed states
            observed_states = []

        if duration_key:
            # if record the durations of each state, then initialize the duration of all states that occur in the simulation
            states_durations = defaultdict(float)

        # initialize the list of undispatched (waiting) requests, their quoted prices, and arrival times
        undispatched_riders = list()
        undispatched_riders_price = list()
        undispatched_riders_arrival_time = list()
        undispatched_riders_quoted_disutility = list()

        # max same-type waiting requests
        max_same_type_waiting = 0

        # iterate over weeks in the data set
        for week in range(n_weeks):
            # initialized time and state
            accumulated_time = 0
            initial_state = tuple(np.zeros(n_rider_types))
            state = initial_state

            # retrieve the events and cdf_inverses for this week
            outside_arrival_time_copy = self.outside_arrival_time[set_type][week].copy()
            outside_arrival_events_copy = self.outside_arrival_events[set_type][
                week
            ].copy()
            cdf_inverses = self.cdf_inverses[set_type][week].copy()
            convert_events = self.convert_events[set_type][week].copy()

            # iterate over events, until time limit is reached
            while accumulated_time < time_limit:

                # update the max state space observed
                max_same_type_waiting = max(max(state), max_same_type_waiting)

                # record the time of the last state
                accumulated_time_prev = accumulated_time

                # -------------------------------------------------------------------------------
                # Decide whether next event is an eyeball/observer arrival or a request departure
                # -------------------------------------------------------------------------------

                # get the time of the next eyeball arrival, if there are still outside arrivals left
                if len(outside_arrival_time_copy) > 0:
                    next_outside_arrival_time = outside_arrival_time_copy[0]
                else:
                    break

                # get departure rate for the current state
                total_depart_rates = np.sum(state * thetas)
                # if the departure rate is positive, then the next event can be a departure
                if total_depart_rates > 0:
                    # compute the probability of each departure
                    depart_prob = state * thetas / total_depart_rates
                    # randomly choose the next request to depart
                    event = self.depart_choices[
                        np.random.choice(n_rider_types, p=depart_prob)
                    ]
                    # inversion sampling for interarrival times of departures
                    if not len(cdf_inverses):
                        raise Exception(
                            "The list cdf_inverses has run out of elements."
                        )
                    depart_interarrival_time = (
                        -math.log(1 - cdf_inverses.pop()) / total_depart_rates
                    )
                    next_depart_time = accumulated_time + depart_interarrival_time
                    # if the depart time is before the next arrival, then the next event is a departure;
                    # otherwise, the next event is an outside arrival
                    if next_outside_arrival_time > next_depart_time:
                        accumulated_time = next_depart_time
                    else:
                        accumulated_time = next_outside_arrival_time
                        event = outside_arrival_events_copy[0]
                        # remove the first element from the list
                        outside_arrival_events_copy = outside_arrival_events_copy[1:]
                        outside_arrival_time_copy = outside_arrival_time_copy[1:]

                # if the departure rate is 0, then the next event must be an outside arrival
                else:
                    accumulated_time = next_outside_arrival_time
                    event = outside_arrival_events_copy[0]
                    outside_arrival_events_copy = outside_arrival_events_copy[1:]
                    outside_arrival_time_copy = outside_arrival_time_copy[1:]

                if duration_key:
                    # record the duration of the previous state and add it to the duration of the state
                    states_durations[state] += accumulated_time - accumulated_time_prev

                # convert the state to a tuple
                event = tuple(event)

                # -----------------------------------------------------------------
                # Actions for observer, rider departure, and eyeball arrival events
                # -----------------------------------------------------------------

                # observer arrival
                if event == (OBSERVER, 0):
                    if sample_key:
                        observed_states.append(state)  # record the observed state
                    continue  # no need to update the state

                for i in np.arange(n_rider_types):
                    # departure of a type-i request
                    if event == (DEPART, i):
                        # change state
                        state = depart(state, i)

                        # performance metrics when the type-i request is dispatched solo
                        throughput += 1
                        requests_per_type[i] += 1
                        total_eyeballs += 1
                        arrivals_per_type[i] += 1
                        price_i = undispatched_riders_price[
                            undispatched_riders.index(i)
                        ]
                        profit += price_i * request_length_matrix[i, i]
                        revenue += price_i * request_length_matrix[i, i]
                        solo_dispatch_cost = c * request_length_matrix[i, i]
                        profit -= solo_dispatch_cost
                        cost += solo_dispatch_cost
                        total_cost_per_type[i] += solo_dispatch_cost
                        payments += [price_i]

                        active_requests_per_type[i] += 1

                        # riders metrics of i
                        eyeball_metrics["rider_type"] += [i]
                        eyeball_metrics["convert"] += [1]
                        eyeball_metrics["wait"] += [1]
                        eyeball_metrics["price"] += [price_i]
                        eyeball_metrics["quoted_disutility"] += [
                            undispatched_riders_quoted_disutility[
                                undispatched_riders.index(i)
                            ]
                        ]
                        eyeball_metrics["realized_disutility"] += [
                            price_i
                        ]  # solo ride -- the realized disutility is the price
                        eyeball_metrics["matched"] += [UNMATCHED]
                        eyeball_metrics["cost"] += [solo_dispatch_cost]
                        eyeball_metrics["matched_with"] += [np.nan]
                        eyeball_metrics["active/passive"] += [ACTIVE]
                        eyeball_metrics["week"] += [week]
                        eyeball_metrics["arrival_time"] += [
                            undispatched_riders_arrival_time[
                                undispatched_riders.index(i)
                            ]
                        ]
                        eyeball_metrics["dispatch_time"] += [accumulated_time]
                        eyeball_metrics["wait_dist_before_pickup"] += [0]
                        eyeball_metrics["detour"] += [0]
                        eyeball_metrics["detour_rate"] += [0]

                        # Dispatch i, remove i from the list of undispatched riders
                        del undispatched_riders_price[undispatched_riders.index(i)]
                        del undispatched_riders_quoted_disutility[
                            undispatched_riders.index(i)
                        ]
                        del undispatched_riders_arrival_time[
                            undispatched_riders.index(i)
                        ]
                        del undispatched_riders[undispatched_riders.index(i)]

                    # arrival of a type-i request
                    if event == (ARRIVE, i):
                        # pricing policy
                        price = pricing_policy.pricing_function(i, state)[1]

                        # matching policy
                        match_option = matching_policy.matching_function(i, state)

                        # calculate the conversion rate
                        # the conversion depends on the disutility
                        # if the disutility is based on the matching outcome (e.g., match-based pricing)
                        if pricing_policy.match_flag:
                            if match_option is None:
                                # if wait, then the conversion is based on the active disutility
                                disutility = self.instance_data.active_disutility[i]
                            else:
                                # if match, then the conversion is based on the passive disutility
                                disutility = self.instance_data.passive_disutility[i]
                        # if the disutility is not based on the matching outcome (e.g., static pricing)
                        else:
                            disutility = self.instance_data.disutility[i]

                        # calculate the conversion rate
                        conversion = max(1 - disutility - price, 0)
                        # the quoted disutility is the sum of the expected disutility and the quoted price
                        quoted_disutility = disutility + price

                        # whether the type-i request converts
                        if not len(convert_events):
                            raise Exception(
                                "The list convert_events has run out of elements."
                            )
                        # if the conversion is lower than the random number in [0,1),
                        # i.e., quoted price higher than the willingness-to-pay,
                        # then the type-i request does not convert
                        if convert_events.pop() > conversion:
                            total_eyeballs += 1
                            arrivals_per_type[i] += 1
                            eyeball_metrics["rider_type"] += [i]
                            eyeball_metrics["convert"] += [0]
                            eyeball_metrics["wait"] += [np.nan]
                            eyeball_metrics["price"] += [price]
                            eyeball_metrics["quoted_disutility"] += [quoted_disutility]
                            eyeball_metrics["realized_disutility"] += [np.nan]
                            eyeball_metrics["matched"] += [np.nan]
                            eyeball_metrics["cost"] += [np.nan]
                            eyeball_metrics["matched_with"] += [np.nan]
                            if match_option is None:
                                eyeball_metrics["active/passive"] += [ACTIVE]
                            else:
                                eyeball_metrics["active/passive"] += [PASSIVE]
                            eyeball_metrics["week"] += [week]
                            eyeball_metrics["arrival_time"] += [accumulated_time]
                            eyeball_metrics["dispatch_time"] += [np.nan]
                            eyeball_metrics["wait_dist_before_pickup"] += [np.nan]
                            eyeball_metrics["detour"] += [np.nan]
                            eyeball_metrics["detour_rate"] += [np.nan]

                        # the type-i request converts
                        else:
                            # if there is no match option, then the type-i request waits in the system
                            if match_option is None:
                                # change state
                                state = arrive(state, i)
                                # add i to the list of undispatched riders
                                undispatched_riders += [i]
                                undispatched_riders_price += [price]
                                undispatched_riders_quoted_disutility += [
                                    quoted_disutility
                                ]
                                undispatched_riders_arrival_time += [accumulated_time]

                            # if there is a match option, then the type-i request matches with the match option, and i is the passive rider
                            else:
                                j = match_option
                                # update the state
                                state = depart(state, j)

                                # system level metrics
                                throughput += 2
                                requests_per_type[i] += 1
                                requests_per_type[j] += 1
                                total_eyeballs += 2
                                arrivals_per_type[i] += 1
                                arrivals_per_type[j] += 1

                                price_i = price
                                price_j = undispatched_riders_price[
                                    undispatched_riders.index(j)
                                ]
                                profit += (
                                    price_i * request_length_vector[i]
                                    + price_j * request_length_vector[j]
                                )
                                revenue += (
                                    price_i * request_length_vector[i]
                                    + price_j * request_length_vector[j]
                                )
                                match_cost = c * request_length_matrix[i, j]
                                profit -= match_cost
                                cost += match_cost
                                matched_riders += 2
                                payments += [price_i, price_j]
                                matched_requests_per_type[i] += 1
                                matched_requests_per_type[j] += 1

                                # cost allocation
                                A_cost = c * (
                                    A_solo_length[i, j] + 0.5 * shared_length[i, j]
                                )
                                B_cost = c * (
                                    B_solo_length[i, j] + 0.5 * shared_length[i, j]
                                )
                                total_cost_per_type[i] += A_cost
                                total_cost_per_type[j] += B_cost

                                # for the detour of each type
                                detour_i = (
                                    shared_length[i, j]
                                    + A_solo_length[i, j]
                                    - request_length_vector[i]
                                )
                                detour_j = (
                                    shared_length[i, j]
                                    + B_solo_length[i, j]
                                    - request_length_vector[j]
                                )
                                total_detour_per_type[i] += detour_i
                                total_detour_per_type[j] += detour_j

                                # j is active, i is passive
                                active_requests_per_type[j] += 1
                                active_matched_requests_per_type[j] += 1
                                active_total_detour_per_type[j] += detour_j
                                passive_requests_per_type[i] += 1
                                passive_total_detour_per_type[i] += detour_i

                                quoted_disutility_i = quoted_disutility
                                quoted_disutility_j = (
                                    undispatched_riders_quoted_disutility[
                                        undispatched_riders.index(j)
                                    ]
                                )

                                # riders metrics of i, j
                                eyeball_metrics["rider_type"] += [i, j]
                                eyeball_metrics["convert"] += [1, 1]
                                eyeball_metrics["wait"] += [0, 1]
                                eyeball_metrics["price"] += [price_i, price_j]
                                eyeball_metrics["quoted_disutility"] += [
                                    quoted_disutility_i,
                                    quoted_disutility_j,
                                ]
                                eyeball_metrics["realized_disutility"] += [
                                    price_i
                                    + beta0
                                    + beta1 * detour_i / request_length_vector[i],
                                    price_j
                                    + beta0
                                    + beta1 * detour_j / request_length_vector[j],
                                ]
                                eyeball_metrics["matched"] += [MATCHED, MATCHED]
                                eyeball_metrics["cost"] += [A_cost, B_cost]
                                eyeball_metrics["matched_with"] += [j, i]
                                eyeball_metrics["active/passive"] += [PASSIVE, ACTIVE]
                                eyeball_metrics["week"] += [week, week]
                                eyeball_metrics["arrival_time"] += [
                                    accumulated_time,
                                    undispatched_riders_arrival_time[
                                        undispatched_riders.index(j)
                                    ],
                                ]
                                eyeball_metrics["dispatch_time"] += [
                                    accumulated_time,
                                    accumulated_time,
                                ]
                                if trip_choice[i, j] == ABAB:
                                    eyeball_metrics["wait_dist_before_pickup"] += [
                                        0,
                                        A_solo_length[i, j],
                                    ]
                                elif trip_choice[i, j] == ABBA:
                                    eyeball_metrics["wait_dist_before_pickup"] += [
                                        0,
                                        A_solo_length_1[i, j],
                                    ]
                                elif trip_choice[i, j] == BABA:
                                    eyeball_metrics["wait_dist_before_pickup"] += [
                                        B_solo_length[i, j],
                                        0,
                                    ]
                                elif trip_choice[i, j] == BAAB:
                                    eyeball_metrics["wait_dist_before_pickup"] += [
                                        B_solo_length_1[i, j],
                                        0,
                                    ]
                                else:
                                    raise Exception("Invalid trip choice.")

                                eyeball_metrics["detour"] += [detour_i, detour_j]
                                eyeball_metrics["detour_rate"] += [
                                    detour_i / request_length_vector[i],
                                    detour_j / request_length_vector[j],
                                ]

                                # Dispatch j
                                del undispatched_riders_price[
                                    undispatched_riders.index(j)
                                ]
                                del undispatched_riders_quoted_disutility[
                                    undispatched_riders.index(j)
                                ]
                                del undispatched_riders_arrival_time[
                                    undispatched_riders.index(j)
                                ]
                                del undispatched_riders[undispatched_riders.index(j)]

        # compute other metrics
        if throughput > 0:

            # match rate
            match_rate = matched_riders / throughput

            # cost efficiency
            solo_cost = np.sum(requests_per_type * c * request_length_vector)
            match_efficiency = 1 - cost / solo_cost

            # average payment
            ave_payment = np.mean(payments)
        else:
            match_rate = np.nan
            solo_cost = 0
            match_efficiency = np.nan
            ave_payment = np.nan
        conversion_rate = throughput / total_eyeballs
        ave_quoted_price = np.mean(eyeball_metrics["price"])
        ave_quoted_disutility = np.mean(eyeball_metrics["quoted_disutility"])
        ave_quoted_price_per_type = np.zeros(n_rider_types)
        ave_conversion_per_type = np.zeros(n_rider_types)
        for i in range(n_rider_types):
            ave_quoted_price_per_type[i] = np.mean(
                [
                    price
                    for j, price in enumerate(eyeball_metrics["price"])
                    if i == eyeball_metrics["rider_type"][j]
                ]
            )
            ave_conversion_per_type[i] = np.mean(
                [
                    convert
                    for j, convert in enumerate(eyeball_metrics["convert"])
                    if i == eyeball_metrics["rider_type"][j]
                ]
            )

        # compute the average cost per type
        ave_cost_per_type = np.zeros(n_rider_types)
        for i in range(n_rider_types):
            if requests_per_type[i] == 0:
                ave_cost_per_type[i] = c
            else:
                ave_cost_per_type[i] = total_cost_per_type[i] / (
                    requests_per_type[i] * request_length_vector[i]
                )

        # overall detour and disutility
        ave_detour_over_matched = np.sum(total_detour_per_type) / matched_riders
        ave_detour_over_all_requests = np.sum(total_detour_per_type) / throughput
        ave_detour_rate_over_matched = np.sum(total_detour_per_type) / np.sum(
            matched_requests_per_type * request_length_vector
        )
        ave_detour_rate_over_all_requests = np.sum(total_detour_per_type) / np.sum(
            requests_per_type * request_length_vector
        )
        # sharing and detour disutility
        ave_disutility_over_requests = (
            beta0 * matched_riders
            + beta1 * np.sum(total_detour_per_type / request_length_vector)
        ) / throughput
        # realized disutility (over all requests): sharing and detour disutility + payment
        ave_realized_disutility = ave_disutility_over_requests + ave_payment

        # calculate the disutility vectors for each rider type
        disutility_per_type, active_disutility_per_type, passive_disutility_per_type = (
            calculate_disutility(
                self.instance_data,
                shrinkage_factor,
                requests_per_type,
                matched_requests_per_type,
                total_detour_per_type,
                active_requests_per_type,
                active_matched_requests_per_type,
                active_total_detour_per_type,
                passive_requests_per_type,
                passive_total_detour_per_type,
            )
        )

        # save performance metrics
        metrics = dict()
        metrics["profit"] = profit / time_limit
        metrics["ave_quoted_price"] = ave_quoted_price
        metrics["ave_quoted_disutility"] = ave_quoted_disutility
        metrics["ave_realized_disutility"] = ave_realized_disutility
        metrics["ave_payment"] = ave_payment
        metrics["match_rate"] = match_rate
        metrics["cost_efficiency"] = match_efficiency
        metrics["throughput"] = throughput / time_limit
        metrics["total_arrivals"] = total_eyeballs / time_limit
        metrics["revenue"] = revenue / time_limit
        metrics["cost"] = cost / time_limit
        metrics["conversion_rate"] = conversion_rate
        metrics["ave_detour_over_matched"] = (
            ave_detour_over_matched  # in meters, over matched requests
        )
        metrics["ave_detour_over_all_requests"] = (
            ave_detour_over_all_requests  # in meters, over all requests
        )
        metrics["ave_detour_rate_over_matched"] = (
            ave_detour_rate_over_matched  # in rate, over matched requests
        )
        metrics["ave_detour_rate_over_all_requests"] = (
            ave_detour_rate_over_all_requests  # in rate, over all requests
        )
        metrics["ave_disutility_over_requests"] = ave_disutility_over_requests
        if throughput > 0:
            metrics["ave_revenue_per_rider"] = revenue / throughput
            metrics["ave_cost_per_rider"] = cost / throughput
        else:
            metrics["ave_revenue_per_rider"] = np.nan
            metrics["ave_cost_per_rider"] = np.nan
        metrics["ave_quoted_price_per_type"] = ave_quoted_price_per_type
        metrics["ave_conversion_per_type"] = ave_conversion_per_type
        metrics["ave_cost_per_type"] = ave_cost_per_type
        metrics["arrival_rate_per_type"] = arrivals_per_type / time_limit

        metrics["requests_per_type"] = requests_per_type
        metrics["matched_requests_per_type"] = matched_requests_per_type

        # disutility per type
        metrics["disutility_per_type"] = disutility_per_type
        metrics["active_disutility_per_type"] = active_disutility_per_type
        metrics["passive_disutility_per_type"] = passive_disutility_per_type

        # elements used to calculate the disutility vectors
        metrics["requests_per_type"] = requests_per_type
        metrics["matched_requests_per_type"] = matched_requests_per_type
        metrics["total_detour_per_type"] = total_detour_per_type
        metrics["active_requests_per_type"] = active_requests_per_type
        metrics["active_matched_requests_per_type"] = active_matched_requests_per_type
        metrics["active_total_detour_per_type"] = active_total_detour_per_type
        metrics["passive_requests_per_type"] = passive_requests_per_type
        metrics["passive_total_detour_per_type"] = passive_total_detour_per_type

        # at most how many requests of the same type are waiting in the system
        metrics["max_same_type_waiting"] = max_same_type_waiting

        self.metrics = metrics

        # eyeball level metrics (in a dataframe)
        eyeball_metrics_df = pd.DataFrame.from_dict(eyeball_metrics)
        eyeball_metrics_df["wait_time_in_system"] = (
            eyeball_metrics_df["dispatch_time"] - eyeball_metrics_df["arrival_time"]
        )
        eyeball_metrics_df["cost_efficiency"] = 1 - eyeball_metrics_df["cost"] / (
            c
            * eyeball_metrics_df["rider_type"].apply(lambda x: request_length_vector[x])
        )
        self.eyeball_metrics_df = eyeball_metrics_df

        if sample_key:
            # delete the repeated states in the list of sampled states
            sampled_states = [*set(observed_states)]

            if duration_key:
                return sampled_states, states_durations
            else:
                return sampled_states

        else:
            if duration_key:
                return states_durations


def calculate_disutility(
    instance_data,
    shrinkage_factor,
    requests_per_type,
    matched_requests_per_type,
    total_detour_per_type,
    active_requests_per_type,
    active_matched_requests_per_type,
    active_total_detour_per_type,
    passive_requests_per_type,
    passive_total_detour_per_type,
):
    """Calculate the three disutility vectors for each rider type (disutility per type, active disutility per type, and passive disutility per type)."""

    n_rider_types = instance_data.n_rider_types
    beta0, beta1 = instance_data.beta0, instance_data.beta1
    request_length_vector = instance_data.request_length_vector

    """ Match rate and detour, over all requests -- updated for static/iterative pricing """
    # match rate and detour, over all (both active and passive) requests
    average_match_rate_per_type = np.zeros(n_rider_types)  # conditional on requests
    average_detour_per_type_over_matched = np.zeros(
        n_rider_types
    )  # in meters, conditional on matched requests

    # shrinkage rates, for match rate and detour, respectively
    shrinkage_rates_mr = requests_per_type / (
        requests_per_type + shrinkage_factor
    )  # the coefficient of keeping the true match rate
    shrinkage_rates_d = matched_requests_per_type / (
        matched_requests_per_type + shrinkage_factor
    )  # the coefficient of keeping the true detour

    for i in range(n_rider_types):
        # 1. match rate
        # this is the part that the average match rate of at the origin - 0
        # this is the part of the average match rate of the type -- only counted when there is any request of the type
        if requests_per_type[i] > 0:
            average_match_rate_per_type[i] += (
                shrinkage_rates_mr[i]
                * matched_requests_per_type[i]
                / requests_per_type[i]
            )

        # 2. detour
        # this is the part that the average detour of at the origin - 0
        # this is the part of the average detour of the type -- only counted when there is any matched request of the type
        if matched_requests_per_type[i] > 0:
            average_detour_per_type_over_matched[i] = (
                shrinkage_rates_d[i]
                * total_detour_per_type[i]
                / matched_requests_per_type[i]
            )

    # disutility per type
    disutility_per_type = average_match_rate_per_type * (
        beta0 + beta1 * average_detour_per_type_over_matched / request_length_vector
    )

    """ Match rate and detour rate, over active riders -- updated for match-based pricing"""
    average_active_match_rate_per_type = np.zeros(
        n_rider_types
    )  # conditional on active requests
    average_active_detour_per_type = np.zeros(
        n_rider_types
    )  # conditional on active and matched requests

    # shrinakge rates, for active match rate and detour, respectively
    shrinkage_rates_mr_active = active_requests_per_type / (
        active_requests_per_type + shrinkage_factor
    )  # the coefficient of keeping the true match rate
    shrinkage_rates_d_active = active_matched_requests_per_type / (
        active_matched_requests_per_type + shrinkage_factor
    )  # the coefficient of keeping the true detour

    for i in range(n_rider_types):

        # 1. active match rate
        if active_requests_per_type[i] > 0:
            average_active_match_rate_per_type[i] += (
                shrinkage_rates_mr_active[i]
                * active_matched_requests_per_type[i]
                / active_requests_per_type[i]
            )

        # 2. active detour
        if active_matched_requests_per_type[i] > 0:
            average_active_detour_per_type[i] += (
                shrinkage_rates_d_active[i]
                * active_total_detour_per_type[i]
                / active_matched_requests_per_type[i]
            )

    active_disutility_per_type = average_active_match_rate_per_type * (
        beta0 + beta1 * average_active_detour_per_type / request_length_vector
    )

    """ Detour rate, over passive riders -- updated for match-based pricing"""
    average_passive_detour_per_type = np.zeros(n_rider_types)

    # shrinakge rates, for passive detour
    shrinkage_rates_d_passive = passive_requests_per_type / (
        passive_requests_per_type + shrinkage_factor
    )

    for i in range(n_rider_types):
        if passive_requests_per_type[i] > 0:
            average_passive_detour_per_type[i] = (
                shrinkage_rates_d_passive[i]
                * passive_total_detour_per_type[i]
                / passive_requests_per_type[i]
            )

    passive_disutility_per_type = (
        beta0 + beta1 * average_passive_detour_per_type / request_length_vector
    )

    return disutility_per_type, active_disutility_per_type, passive_disutility_per_type
