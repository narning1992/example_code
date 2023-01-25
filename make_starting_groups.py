"""
This script uses furthest neighbour algorithm for MCMC chain initialisation. Herein
starting groups for the MCMCs are chosen, so that each group is minimally
correlated within itself. This explores the space of all covariates better than random.
intialisation

The script is used as follows:

    python make_starting_groups.py \
        -n <name of the output> \
        -m <number of chains to start> \
        -l <number of covariates in starting group> \

It creates a single csv file containing where each starting
group is a seperate line. The columns are the same as in the correlation file.
The cells are either 0 or 1 denoting inclusion(1) in the starting group

The script also outputs results.comparison_sampling.png which shows the
correlation within and between groups
"""
import os
import argparse
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids

def reverse(x):
    """
    Just returns 1-x needed for turning correlation to distances
    """
    return(1-x)


def get_matrix(corr):
    """
    Takes a file that contains all column to column correlations
    returns groups that correlate more than p to each other.
    """

    # Get the file created by run-covid.R that correlates every covariate with
    # each other
    corr = pd.read_csv(corr, index_col=0)
    col_order = list(corr.columns)
    corr = corr.astype(float)

    # Check if missing columns
    common_corr = set(corr.index).intersection(set(corr.columns))
    corr = corr[common_corr]
    corr = corr.T[common_corr].T



    # Doesn't matter whether positive or negative correlation so get absolute
    corr = corr.abs()
    corr = corr.fillna(0)

    # reverse correlation to get distance
    distance = corr.applymap(reverse)

    return(distance, col_order)


def write_groups4MCMC(groups, col_order, folder):
    """
    Function that writes the chain start groups to file so its readable by doublethink
    """
    # Initialise array with zeros to become output. Number of rows is equal to the number of starting chains
    # number of columns is equal to number of columns
    out_array = np.zeros((len(groups), len(col_order)), dtype=np.uint8)
    for i, group in enumerate(groups):  # go through every starting group
        group = [col_order.index(x) for x in group]
        group = np.array(group)
        out_array[i, group] = 1  # Set all covariates included to one
    # Write to file
    np.savetxt(
        "{}_groups.csv".format(folder),
        out_array,
        delimiter=",",
        fmt="%i")


def get_farthest_neighbour(distance_matrix, covariate_seeds, auto_switch):
    """
    This function gets the farthest neighbour group using the distances
    from the distance file and the starting group containing all
    first variables from every chain
    """
    all_cols = set(distance_matrix.columns)
    num_cols = len(all_cols)

    # Sets to trace which covariates have already been taken
    cov_trace_FN = set()
    cov_trace_random = set()

    # Count covariates covered by furthest neighbour and random sampling
    cov_count_FN = []
    cov_count_random = []

    # This is a switch that denotes whether the chain number should be
    # automated
    if args.m == 0:
        covariate_seeds = all_cols
        auto_switch = True
    else:
        auto_switch = False

    # copy distance matrix for plotting as it will be deleted bit by bit
    distance_matrix_copy = distance_matrix.copy()

    # Store random sampling
    random_groups = []
    groups = []

    # Create group for every chain as denoted by m
    for i, start_seed in enumerate(covariate_seeds):
        # See if the starting seed for the group is still in the distance
        # matrix
        if not start_seed in distance_matrix.columns:
            start_seed = random.choice(distance_matrix.columns)

        # Get the farthest neihgbour group
        group = farthest_neighbour(distance_matrix, start_seed)
        # Get a random group for comparison
        group_random = random.choices(
            list(all_cols-cov_trace_random), k=args.l)

        # Trace which covariates get included
        cov_trace_FN = cov_trace_FN.union(set(group))
        cov_trace_random = cov_trace_random.union(set(group_random))

        # trace the number of covariates included
        cov_count_FN.append(len(cov_trace_FN))
        cov_count_random.append(len(cov_trace_random))

        # Drop all the covariates in this group from the remaining pool of
        # covariates (the distance matrix so we don't have to recalc the
        # distances
        distance_matrix.drop(columns=group, index=group, inplace=True)

        # add to the groups
        random_groups.append(group_random)
        groups.append(group)

        # If automatically calculating chain numbers, stop when all covariates
        # were used
        if auto_switch or distance_matrix.empty:
            if len(cov_trace_FN) == num_cols:
                print("""Every covariate has been included at least once at
                iteration {}\nExiting""".format(i))
                break

    # plot the chain initialisation
    plot_init(
        cov_count_FN,
        cov_count_random,
        len(cov_count_random),
        num_cols,
        distance_matrix_copy,
        groups,
        random_groups)

    return(groups)


def within_between_correlation(distance_matrix, groups):
    """
    Function that registers within and between correlation of groups using a distance matrix
    """
    correlations_within = []
    correlations_between = []

    for i, group in enumerate(groups): # Loop through all groups
        corr_within = distance_matrix[group].T[group].values # Within group correlation
        np.fill_diagonal(corr_within, np.nan) # Ignore diagonals (which will be 1)
        correlations_within.append(
            np.nanmean(corr_within)) # add the mean to output but ignore the NA (diagonal)
        other_groups = groups[:i] + groups[i+1:] # All other covariates
        other_groups = [group for subgroups in other_groups
                        for group in subgroups] # Flatten list of lists
        correlations_between.append(
            distance_matrix[group].T[other_groups].values.mean()) # Append between group correlation

    return(correlations_within, correlations_between)


def plot_init(cov_count_FN, cov_count_random, num_chains,
              num_cols, distance_matrix, groups, random_groups):
    """
    Function that plots furthest neighbour chain initialisation vs random chain initialisation.
    """

    # Make distance to correlation again
    corr_matrix = distance_matrix.applymap(reverse)

    # The following plots the percentage of covariates covered

    cov_count_FN = [x/num_cols for x in cov_count_FN]
    cov_count_random = [x/num_cols for x in cov_count_random]
    hue_column = []
    hue_column = ["Furthest Neighbour" for i in range(
        num_chains)] + ["Random" for i in range(num_chains)]
    x_column = list(range(num_chains)) + list(range(num_chains))

    df_plot = pd.DataFrame(zip(cov_count_FN+cov_count_random,
                               hue_column,
                               x_column),
                           columns=["Percentage of covariates covered",
                                    "Sampling method",
                                    "Number of chains"]
                           )
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    sns.lineplot(
        data=df_plot,
        x="Number of chains",
        y="Percentage of covariates covered",
        hue="Sampling method",
        ax=axes[0]
    )

    axes[0].title.set_text(
        "Comparison furthest neighbour vs. random sampling in covariates covered over number of chains")

    # Next we plot the within and between group correlations
    FN_correlations_within, FN_correlations_between = within_between_correlation(
        corr_matrix, groups)
    random_correlations_within, random_correlations_between = within_between_correlation(
        corr_matrix, random_groups)
    print("Random groups within correlation = {}".format(
              np.mean(random_correlations_within)))
    print("Furthest neighbour groups within correlation = {}".format(
              np.mean(FN_correlations_within)))
    print("Random groups between correlation = {}".format(
              np.mean(random_correlations_between)))
    print("Furthest neighbour groups between correlation = {}".format(
              np.mean(FN_correlations_between)))
    y_violin = []
    x_violin = []
    hue = []
    y_violin += FN_correlations_within
    x_violin += ["Correlations within chains"] * len(FN_correlations_within)
    hue += ["Furthest Neighbour"] * len(FN_correlations_within)

    y_violin += FN_correlations_between
    x_violin += ["Correlations between chains"] * len(FN_correlations_between)
    hue += ["Furthest Neighbour"] * len(FN_correlations_between)

    y_violin += random_correlations_within
    x_violin += ["Correlations within chains"] * \
        len(random_correlations_within)
    hue += ["Random Sampling"] * len(random_correlations_within)

    y_violin += random_correlations_between
    x_violin += ["Correlations between chains"] * \
        len(random_correlations_between)
    hue += ["Random Sampling"] * len(random_correlations_between)

    violin_df = pd.DataFrame(
        list(zip(x_violin, y_violin, hue)),
        columns=["Medians", "Correlation", "Sampling type"]
        )
    sns.barplot(
        data=violin_df,
        x="Medians",
        y="Correlation",
        hue="Sampling type",
        estimator=np.median,
        ax=axes[1],
        )
    axes[1].set_yscale("log")
    axes[1].title.set_text(
        "Comparison furthest neighbour vs. random sampling in within and between chain correlation")
    plt.tight_layout()
    plt.savefig("results.comparison_sampling.png")


def farthest_neighbour(distance_matrix, starting_point):
    """
    The function containing the farthest neighbour algorithm
    """

    # Get all columns from the distance matrix
    columns = list(distance_matrix.columns)

    # Shuffle
    random.shuffle(columns)

    # Keep track of the columns at the beginning as we will remove from the
    # columns variable
    remaining_columns = columns.copy()

    # Output
    start_group_MCMC = [starting_point]

    # For loop as long as the intitalisation group should be
    for _ in range(args.l-1):
        # Get the distance of every column to starting point. This is repeated
        # every time after a column gets added to the start group
        distances = [
            distance_matrix.at[rem_col, start_group_MCMC[0]]
            for rem_col in remaining_columns]

        # Loop through all remaining columns not in the start group
        for i, rem_col in enumerate(remaining_columns):
            # Loop through all the columns in the start group
            for j, start_col in enumerate(start_group_MCMC):
                # Take the distance to the starting seed covariate or other members of group whatevers closer
                distances[i] = min(
                        distances[i],
                        distance_matrix.at[rem_col, start_col])
        if len(distances) == 0:
            break
        # Get the column with the maximum distance into the start group
        start_group_MCMC.append(
            remaining_columns.pop(
                distances.index(
                    max(distances))))
    # Return the start group
    return start_group_MCMC


def get_start_groups(distance_matrix):
    """
    Get the one single starting covariate for every chain. Then fill the rest with furthest neighbour
    """
    # If no chain number is specified we can guess the correct amount of
    # chains needed
    auto_switch = False
    if args.m == 0:
        args.m = int(len(distance_matrix.columns)/args.l)
        auto_switch = True

    # Initialise the chains by choosing medoids from the distance matrix
    kmedoids = KMedoids(n_clusters=args.m+2,
                        random_state=0,
                        metric="precomputed",
                        init="heuristic")
    kmedoids.fit(distance_matrix)
    medoids = np.array(distance_matrix.index)[kmedoids.medoid_indices_]

    # Get the farthest neighbour
    groups = list(
        get_farthest_neighbour(
            distance_matrix,
            medoids,
            auto_switch))
    return(groups)


if __name__ == "__main__":
    # This is the main function

    # Set random seed
    random.seed(10)

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Get starting points for MCMC mixing")
    parser.add_argument(
        "-n",
        type=str,
        help="give outputs a name")
    parser.add_argument(
        "-l", type=int, default=10,
        help="length of the starting variable groups")
    parser.add_argument(
        "-m", type=int, default=0,
        help="how many groups for the medoid clustering")
    global args
    args = parser.parse_args()

    # Name of the correlation file created by run-covid.R
    corr_table = "{}_corr.csv".format(args.n)

    # Get the correlation matrix
    print("Getting distance matrix then making groups")
    distance_matrix, col_order = get_matrix(corr_table)

    # From here create start groups by taking the medoids from k-medoid (k=m) clustering
    # and initialising one chain with each. Then fill up the residual l members
    # By choosing the least correlated covariates from the pool of residual
    # covariates. Covariates included in a group are removed from the pool.
    print("Make start groups")
    out_groups = get_start_groups(distance_matrix)

    # Write the chains to file
    write_groups4MCMC(out_groups, col_order, args.n)
