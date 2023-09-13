# Pricing Shared Rides

## Project Description
This repository contains relevant source code to reproduce all experiments in the paper [Pricing Shared Rides](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4551405). If you use any of the material here, please include a reference to the paper.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1. Jupyter Notebook `computational_experiments.ipynb`](#1-jupyter-notebook-computational_experimentsipynb)
  - [2. Files in `/lib`](#2-files-in-lib)
- [Chicago data description](#chicago-data-description)
  - [1. Chicago network data](#1-chicago-network-data)
  - [2. Chicago demand data](#2-chicago-demand-data)
  - [3. Chicago zone data](#3-chicago-zone-data)

## Installation
Relies on packages `osmnx`, `networkx`, `numpy`, `pandas`, `geopandas`, `matplotlib`, `tqdm`, `graph_tool`, `gurobipy`, `pickle`, `time`, `sys`, `os`, `itertools`, and `collections`. 

*Note*: It is easier to install and run the package `graph_tool` in Mac/Linux. If you use Windows, then `graph-tool` can be installed using Docker. See details [here](https://graph-tool.skewed.de/).


## Usage

### 1. Jupyter Notebook `computational_experiments.ipynb`:

This notebook implements three pricing policies (static pricing, iterative pricing, and match-based pricing) in Chicago. For a specific set of parameters, we train these policies and evaluate them on training and test datasets. For a detailed description of the experiments, please refer to Section 4 of the paper. Complete results are in Tables 2 and EC.2-4.

#### The notebook is organized as follows:
- Section 1: Import packages, classes, functions, and variables.
- Section 2: Load the Chicago data.
- Section 3: Set the parameters.
- Section 4: Train the pricing and matching policies on the Chicago data.
- Section 5: Evaluate the performance of trained policies on the Chicago data.
- Section 6: Save the results and print the performance metrics for each policy.

#### Policy Implementation:
For our experiments, we define two pricing policies (`MatchBasedPricingPolicy` and `StaticPricingPolicy`) and two matching policies (`GreedyMatchingPolicy` and `AffineMatchingPolicy`). The three policies presented in our paper use different combinations of these pricing and matching policies:
- Static and Iterative Policies: combine `StaticPricingPolicy` and `AffineMatchingPolicy`.
- Match-Based Policy: combines `MatchBasedPricingPolicy` and `AffineMatchingPolicy`.

The initial policy used for constraint sampling (described in Section 3.2) combines `StaticPricingPolicy` and `GreedyMatchingPolicy`.

Our code also allows users to define custom pricing and matching policies.
- Custom Pricing Policy: To construct a pricing policy, create a class that inherits from `PricingPolicy`. This class should accept the `instance_data` and other required arguments as inputs. Then, define the `pricing_function` method, which takes the request type and the state as inputs and returns the optimal conversion for the given request in the specified state.
- Custom Matching Policy: To construct a matching policy, create a class that inherits from `MatchingPolicy`. Similar to `PricingPolicy`, this class should accept the `instance_data` and other required arguments. Then, define the `matching_function` method, which takes the request type and the state as the inputs and returns the matching decision for the given request in the given state. The matching decision is either a single-element list containing the waiting request to match with the given request, or an empty list if the given request should wait in the system for another potential match.

Custom policy classes can be evaluated using the `Simulator.simulation` method. For examples, refer to Section 5, 'Performance Evaluation', in the Jupyter Notebook `computational_experiments.ipynb`.

### 2. Files in `/lib`:

- `instance_data.py`: This file contains the class `InstanceData` which saves the network and demand data.

- `policies.py`: This file contains all policy-related classes.

- `simulation.py`: This file contains the `Simulator` class, which can simulate a given pricing and matching policy on both real and synthetic data.

- `network.py`: This file contains the `Network` class and functions that help construct the network object provided to `InstanceData`.

- `utils.py`: This file contains miscellaneous helper functions and constants.

####


## Chicago data description

### 1. Chicago network data

The Chicago road network data is loaded from OpenStreetMap with the python package `osmnx`. This data is used to construct a `Network` object, then saved in the `/data/Chicago_network.pickle` file.

### 2. Chicago demand data

#### Demand data description

The Chicago ride-hailing dataset is available at the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p). Each row in the dataset describes a ride-hailing request, including the pickup and dropoff timestamps and locations, and the willingness of the rider to share the ride (“authorized”). We focus on authorized trips over eight weeks in October and November of 2019. Specifically, we consider two time windows within each week: Monday peak time (7:30-8:30 a.m., from 2019-10-07 to 2019-11-25) and Saturday off-peak time (7:30-8:30 a.m., from 2019-10-05 to 2019-11-23).

Note that the pickup and dropoff timestamps and locations in the original data are not precise.
- For timestamps, all pickups and dropoffs are rounded to 15-minute intervals. We generate precise, second-level pickup times for each request, distributed uniformly between 0 and 15 minutes after the rounded pickup time.
- For locations, all trips originate and end at the centroids of 76 community areas within the City of Chicago
(excluding the O’Hare International Airport). The community area data is available at the [CARTO](https://toddwschneider.carto.com/viz/f4b909cc-ac58-4904-bfc2-a625f015540f/public_map) website. We use a two-step method to create the rider types: 
  - First, we group the original 76 community areas into 42 clusters using k-means clustering.
  - Second, we use the centroids of these 42 zones as proxies for pickup locations. We then cluster riders originating from each pickup zone based on their dropoff locations using k-means clustering. Each rider type in the Monday peak time (Saturday off-peak time) dataset has a minimum of 5 (3) trip records in the training set.

#### Demand data format
The demand data for the two time windows is stored in `/data/Chicago_demands/MON-PEAK_train7_test1.pickle` and `/data/Chicago_demands/SAT-NONPEAK_train7_test1.pickle`. These demand data files contain the following:
- `rider_types`: A list of rider types (origin-destination pairs). The origin and destination node indices correspond to the centroid nodes of the 42 pickup zones in Chicago. There are 244 types for Monday peak time and 250 types for Saturday off-peak time.
- `arrival_rates`: A list of arrival rates, corresponding to the rider types listed in `rider_types`. These rates are estimated from the training dataset and are expressed in arrivals per second.
- `arrival_types`: A dictionary consisting of the rider arrival sequences (lists of rider type indices). To access the data, use the key to specify the training set or test set (`'training'` or `'test'`), followed by the date (such as `'2019-11-25'`).
- `arrival_times`: A dictionary containing the arrival times of riders as lists of increasing values from 0 to 3600 (in seconds). Similar to `arrival_types`, access the data using the key and date.
- `all_trips_df`: A dictionary containing the demand data as pandas dataframes, with a row for each rider request. All requests are mapped to the closest rider types. The dictionary keys `'training'` and `'test'` denote the training and test set, respectively. 
- `rider_types_df`: A pandas dataframe with a row for each rider type. The columns include the origin-destination nodes, arrival rate, and the number of requests in the training set. The dictionary keys `'training'` and `'test'` denote the training and test set, respectively.


### 3. Chicago zone data

The Chicago zone data is stored in `/data/Chicago_zones.pickle` and contains the following:
- `shapes_gdf`: Geopandas data of the 76 community areas.
- `clusters_gdf`: Geopandas data of the 42 zones.

Although the zone data is not used in the Jupyter Notebook `computational_experiments.ipynb`, it can be used to plot maps, as in Figures 7 and 8 in the paper.
