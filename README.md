# Pricing Shared Rides

## Project Description
This repository provides the source code to reproduce all experiments in the paper [Pricing Shared Rides](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4551405). Please cite the paper if you use any materials from this repository.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1. `computational_experiments.ipynb` Notebook](#1-jupyter-notebook-computational_experimentsipynb)
  - [2. Files in `/lib`](#2-files-in-lib)
- [Chicago data description](#chicago-data-description)
  - [1. Chicago Network Data](#1-chicago-network-data)
  - [2. Chicago Demand Data](#2-chicago-demand-data)
  - [3. Chicago Zone Data](#3-chicago-zone-data)

## Installation
The project relies on several packages, including `osmnx`, `networkx`, `numpy`, `pandas`, `geopandas`, `matplotlib`, `graph_tool`, `gurobipy`, `pickle`, `time`, `sys`, `os`, `itertools`, `collections`, and `IPython`. 

**Note:** `graph_tool` is more easily installed on Mac/Linux systems. For Windows users, installation via Docker is recommended. See the [graph-tool installation guide](https://graph-tool.skewed.de/) for details.


## Usage

### 1. `computational_experiments.ipynb` Notebook:

This notebook implements the three pricing policies (static pricing, iterative pricing, and match-based pricing) on the Chicago dataset assuming a cost $c=0.9$, an average sojourn time of 3 min, a sharing disutility factor $\beta_0=0.05$, and a detour disutility factor $\beta_0=0.675$. Each policy is trained and evaluated on both training and test datasets. Detailed experimental methodology can be found in Section 4 of the paper, and the complete evaluation results are presented in Table 2 in Section 4 and Table EC.2 in Appendix EC.3.3.

#### Policy Implementation:
In the computational experiments, the following policies are implemented:
- **Static Pricing Policy:** This policy assigns a fixed price to each rider type (origin-destination pair) regardless of the system state, with the matching policy based on the Affine Linear Program (ALP) using a certain set of affine weights. This is trained in the codebase using the `StaticPricingAffineMatchingCombined` class, and implemented as the `StaticPricingPolicy` class combined with the `AffineMatchingPolicy` class.
- **Iterative Pricing Policy:** This policy sets static prices in a heuristic manner, with the matching policy also based on the ALP and affine weights. This policy is trained using the `IterativePricingAffineMatchingCombined` class, and implemented as the `StaticPricingPolicy` class combined with the `AffineMatchingPolicy` class.
- **Match-Based Pricing Policy:** This policy sets prices and matches requests both based on the ALP and affine weights. This policy is trained using the `MatchBasedPricingAffineMatchingCombined` class, and implemented as the `MatchBasedPricingPolicy` class combined with the `AffineMatchingPolicy` class.

The codebase also supports user-defined pricing and matching policies:
- **Custom Pricing Policy:** To create a custom pricing policy, define a class that inherits from `PricingPolicy`. Include the rider type-specific beliefs about disutility from sharing and detour within the `InstanceData` object used to construct the policy. Define the `pricing_function` method, which takes the rider type and state as inputs and returns the optimal price and conversion rate.
- **Custom Matching Policy:** To define a custom matching policy, create a class that inherits from `MatchingPolicy`. As with the pricing policy, the `InstanceData` object should contain rider type-specific beliefs about disutility. Implement the `matching_function` method, which takes the rider type and state as inputs and returns either a rider type index (to match a waiting request) or None (to let the request wait for future dispatch).

### 2. Files in `/lib`:

- `instance_data.py`: Contains the `InstanceData` class, which stores the network, demand data, and key parameters for each instance.

- `policies.py`: Contains the classes and functions for the training and implementation of the pricing and matching policies.

- `simulation.py`: Contains the Simulator class to run simulations with specified pricing and matching policies.

- `network.py`: Contains classes and functions to create and manage the network object.

- `utils.py`: Includes helper functions and variables.

####


## Chicago data description

### 1. Chicago network data

The Chicago road network data is loaded from OpenStreetMap via the `osmnx` package. This data is used to construct a `Network` object saved in `/data/Chicago_network.pickle`.

### 2. Chicago demand data

#### Demand data description

The Chicago ride-hailing dataset, available on the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p), records ride requests, including pickup and dropoff details. This study focuses on shared ride requests over an eight-week period in October and November 2019, specifically during Monday peak hours (7:30-8:30 a.m., from 2019-10-07 to 2019-11-25).

The original timestamps and locations in this dataset are imprecise:
- **Timestamps** are rounded to 15-minute intervals. For simulations, pickup times are generated at random within the 15-minute window.
- **Locations** are approximated to the centroids of 76 community areas within Chicago (excluding O’Hare International Airport). Locations are approximated to the centroids of 76 community areas within Chicago (excluding O’Hare International Airport); the community area data is available at [CARTO](https://toddwschneider.carto.com/viz/f4b909cc-ac58-4904-bfc2-a625f015540f/public_map). To create the rider types used in the experiments, we employ a two-step clustering method:
  - *Zone grouping:* We first group the original 76 community areas into 42 zones using k-means clustering.
  - *Dropoff clustering:* Using the centroids of these 42 zones as pickup locations, we then cluster riders within each pickup zone based on their dropoff locations, again using k-means clustering. Each resulting rider type has a minimum of 5 trip records in the training dataset.

#### Demand data format
The demand data, stored in `/data/Chicago_demand.pickle`, includes:
- `rider_types`: A list of 244 rider types, each represented by origin-destination centroid pairs.
- `arrival_rates`: Arrival rates (riders per second) for each rider type, based on training data.
- `arrival_types`: A dictionary with rider arrival sequencess, organized by training/test sets and the date (e.g., '2019-11-25').
- `arrival_times`: Arrival times as lists from 0 to 3600 seconds, similarly organized by the dataset key and the date.
- `all_trips_df`: Dataframes of all ride requests, mapped to closest rider types, for both training and test sets.
- `rider_types_df`: Dataframe with details on each rider type’s origin-destination nodes, arrival rate, and training set request count.


### 3. Chicago zone data

The zone data in `/data/Chicago_zones.pickle` includes:
- `shapes_gdf`: Geopandas data for the 76 community areas.
- `clusters_gdf`: Geopandas data for the 42 clustered zones.

This data is used to create the map figures presented in the paper.
