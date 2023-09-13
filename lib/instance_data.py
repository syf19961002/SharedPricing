# This file contains the class `InstanceData` which designed for for saving the instance data,
# along with a helper function for calculating the request lengths and other relevant data.

import numpy as np
import itertools

# choice matrix of the pick-up and drop-off order for a shared trip between rider types i and j
# (type i: A to A; type j: B to B)
AABB = 0
ABAB = 1
ABBA = 2
BABA = 3
BAAB = 4


class InstanceData(object):
    """This class contains the instance data"""

    def __init__(
        self,
        c,
        sojourn_time,
        time_window,
        network,
        rider_types,
        arrival_rates,
        arrival_types,
        arrival_times,
    ):
        self.network = network
        self.rider_types = rider_types
        self.n_rider_types = len(rider_types)
        self.arrival_rates = arrival_rates
        self.c = c
        self.sojourn_time = sojourn_time
        self.thetas = np.ones(self.n_rider_types) / sojourn_time
        self.time_window = time_window
        self.arrival_types = arrival_types
        self.arrival_times = arrival_times

        # trip calculation:
        # 1) request_length_matrix: matrix of the total trip length for each rider type pair
        # 2) shared_length: matrix of the length of shared part of the trip
        # 3) A_solo_length: matrix of the length of type i's solo part in a shared trip between i, j
        # 4) B_solo_length: matrix of the length of type j's solo part in a shared trip between i, j
        # 5) trip_choice: matrix of pick-up and drop-off order for each rider type pair, denoted by AABB, etc.
        # 6) request_length_vector: vector of the trip length for each rider type
        (
            self.request_length_matrix,
            self.shared_length,
            self.A_solo_length,
            self.B_solo_length,
            self.trip_choice,
        ) = populate_total_trip_cost(network, rider_types)
        self.request_length_vector = np.diag(self.request_length_matrix)

        # rider_types i' satisfaction of monge condition:
        # d(i, i) + d(r, s) <= d(i, r) + d(i, s) forall r, s
        self.satisfy_monge = np.ones(self.n_rider_types)
        for i in range(self.n_rider_types):
            for r in range(self.n_rider_types):
                for s in range(self.n_rider_types):
                    if (
                        self.request_length_matrix[i, i]
                        + self.request_length_matrix[r, s]
                        > self.request_length_matrix[i, r]
                        + self.request_length_matrix[i, s]
                    ):
                        self.satisfy_monge[i] = 0
                        continue

    def equals(self, another_instance_data):
        """Check whether two `InstanceData` objects have all the same properties."""

        if (
            self.network is not another_instance_data.network
            or self.c != another_instance_data.c
            or self.sojourn_time != another_instance_data.sojourn_time
            or self.rider_types != another_instance_data.rider_types
            or self.arrival_rates != another_instance_data.arrival_rates
            or self.arrival_types != another_instance_data.arrival_types
            or self.arrival_times != another_instance_data.arrival_times
        ):
            return False
        else:
            return True


def populate_total_trip_cost(network, rider_types):
    """First calculate the shortest time matrices between all possible pairs of nodes
    (for any rider type pair i, j, we have node pairs including OiDi, OiOj, DiDj, OiDj).
    Then with function `match_efficiency_single`, we calculate the request_length_matrix,
    shared cost, solo cost, and the best pick-up and drop-off order for all rider type pairs.
    """

    n_rider_types = len(rider_types)
    origins = np.array([req[0] for req in rider_types]).reshape(n_rider_types, 1)
    destinations = np.array([req[1] for req in rider_types]).reshape(n_rider_types, 1)
    O0D0 = np.repeat(
        network.shortest_time_matrix[origins, destinations][:, np.newaxis, :],
        n_rider_types,
        axis=1,
    )  # i, j -> OiDi
    O0O1 = network.shortest_time_matrix[
        origins[:, np.newaxis, :], origins[np.newaxis, :, :]
    ]  # i, j -> OiOj
    D0D1 = network.shortest_time_matrix[
        destinations[:, np.newaxis, :], destinations[np.newaxis, :, :]
    ]  # i, j -> DiDj
    O0D1 = network.shortest_time_matrix[
        origins[:, np.newaxis, :], destinations[np.newaxis, :, :]
    ]  # i, j -> OiDj

    return match_efficiency_single(origins, destinations, O0D0, O0O1, D0D1, O0D1)


# This method was adapted from code written by Sebastien Martin for the paper “Detours in Shared Rides”.
def match_efficiency_single(origins, destinations, O0D0, O0O1, D0D1, O0D1):
    """Calculate the request length matrix, shared cost, solo cost,
    and the best pick-up and drop-off order for all rider type pairs."""

    assert len(origins) == len(destinations)
    n_rider_types = len(origins)

    # Compute shortest ordering for each match
    AABB_triptime = O0D0 + O0D0.transpose(1, 0, 2)
    ABAB_triptime = (
        O0O1 + O0D1.transpose(1, 0, 2) + D0D1
    )  # route ABAB, we can transpose this matrix to get BABA
    ABBA_triptime = (
        O0O1 + O0D0.transpose(1, 0, 2) + D0D1.transpose(1, 0, 2)
    )  # route ABBA, we can transpose this matrix to get BAAB
    triptime_possibilities = np.stack(
        (
            AABB_triptime,  # 0
            ABAB_triptime,  # 1
            ABBA_triptime,  # 2
            ABAB_triptime.transpose(1, 0, 2),  # 3
            ABBA_triptime.transpose(1, 0, 2),  # 4
        ),
        axis=2,
    )
    best_triptime_choice = np.argmin(triptime_possibilities, axis=2)
    best_triptime = triptime_possibilities[
        np.arange(len(origins))[:, np.newaxis, np.newaxis],
        np.arange(len(origins))[np.newaxis, :, np.newaxis],
        best_triptime_choice,
        np.arange(1)[np.newaxis, np.newaxis, :],
    ]

    shared_length = np.zeros(
        (n_rider_types, n_rider_types)
    )  # matrix of the length of shared part of the trip
    A_solo_length = np.zeros(
        (n_rider_types, n_rider_types)
    )  # matrix of the length of type i's solo trip
    B_solo_length = np.zeros(
        (n_rider_types, n_rider_types)
    )  # matrix of the length of type j's solo trip
    for i, j in itertools.product(range(n_rider_types), range(n_rider_types)):
        if best_triptime_choice[i, j, 0] == AABB:
            shared_length[i, j] = 0  # AA()BB
            A_solo_length[i, j] = O0D1[i, i, 0]  # (AA)BB
            B_solo_length[i, j] = O0D1[j, j, 0]  # AA(BB)
        elif best_triptime_choice[i, j, 0] == ABAB:
            shared_length[i, j] = O0D1[j, i, 0]  # A(BA)B
            A_solo_length[i, j] = O0O1[i, j, 0]  # (AB)AB
            B_solo_length[i, j] = D0D1[i, j, 0]  # AB(AB)
        elif best_triptime_choice[i, j, 0] == ABBA:
            shared_length[i, j] = O0D1[j, j, 0]  # A(BB)A
            A_solo_length[i, j] = O0O1[i, j, 0] + D0D1[j, i, 0]  # (AB)(BA)
            B_solo_length[i, j] = 0
        elif best_triptime_choice[i, j, 0] == BABA:
            shared_length[i, j] = O0D1[i, j, 0]  # B(AB)A
            A_solo_length[i, j] = D0D1[j, i, 0]  # BA(BA)
            B_solo_length[i, j] = O0O1[j, i, 0]  # (BA)BA
        elif best_triptime_choice[i, j, 0] == BAAB:
            shared_length[i, j] = O0D1[i, i, 0]  # BAAB
            A_solo_length[i, j] = 0
            B_solo_length[i, j] = O0O1[j, i, 0] + D0D1[i, j, 0]  # (BA)(AB)

    # matrix of total trip length for each rider type pair
    request_length_matrix = best_triptime[:, :, 0]
    # matrix of pick-up and drop-off order for each rider type pair,
    # denoted by 0 to 4 (AABB, ABAB, ABBA, BABA, BAAB)
    trip_choice = best_triptime_choice[:, :, 0]

    # Note: A_solo_length + B_solo_length + shared_length == request_length_matrix

    return (
        request_length_matrix,
        shared_length,
        A_solo_length,
        B_solo_length,
        trip_choice,
    )
