# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def mad_based_outlier(values,
                      thresh=3):
    """
    MAD code adapted from:
    https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data

    """

    if len(values.shape) == 1:
        values = values[:,None]
    median = np.median(values, axis=0)              # Calculate median of values column-wise operation
    # diff = np.sum((values - median)**2, axis=-1)
    # diff = np.sqrt(diff)
    diff = abs(values - median)                     # Calculate abs deviation of each value from median
    med_abs_deviation = np.median(diff)             # Median of abs. deviations

    if med_abs_deviation == 0:
        med_abs_deviation = 1

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def percentile_based_outlier(values,
                             min_qt=5,
                             max_qt=95):

    minval, maxval = np.percentile(values, [min_qt, max_qt])
    return (values <= minval) | (values >= maxval)


def high_low_based_outliers(values,
                            min_thresh=5,
                            max_thresh=5,
                            upper_qt=95,
                            lower_qt=5):


    if isinstance(values, list):
        values = np.array(values)
    elif isinstance(values, pd.DataFrame):
        values = values.values

    try:
        m, n = values.shape
    except ValueError:
        raise BaseException("ERROR! When introducing a 1-D array, use .reshape(1, -1) or .reshape(-1, 1)")

    # Initialize boolean-array, filled with False (no outlier), with the shape of values arg.
    ref_bool = np.full(shape=values.shape, fill_value=False, dtype="bool")

    # Iterates through each column (for cases where 'values' is not 1D)
    for c in range(n):
        col_vec = values[:, c]  # Array w/ values of column index c
        # Get indexes of elements which value is higher than percentile upper_qt
        high_val_idx = np.where(col_vec > np.percentile(col_vec, upper_qt))
        # Get indexes of elements which value is lower than percentile lower_qt
        low_val_idx = np.where(col_vec < np.percentile(col_vec, lower_qt))

        # high_val_idx = np.argsort(col_vec)[-np.int(np.ceil(0.04*m)):]
        # low_val_idx = np.argsort(col_vec)[0:np.int(np.ceil(0.04*m))]

        # Search in col_vec the values of high_val_idx & high_val_idx
        high_val, low_val = col_vec[high_val_idx], col_vec[low_val_idx]

        # max_values_dist = np.abs(1 - high_val / np.nanmedian(high_val))
        # min_values_dist = np.abs(1 - low_val / np.nanmedian(low_val))
        #
        # # max_values_dist = np.sqrt(np.abs(high_val**2 + np.nanmedian(high_val)**2))
        # # max_values_dist = abs(1 - max_values_dist)/np.nanmedian(max_values_dist)
        # #
        # # min_values_dist = np.sqrt(np.abs(low_val ** 2 + np.nanmedian(low_val) ** 2))
        # # min_values_dist = abs(1 - min_values_dist) / np.nanmedian(min_values_dist)
        #
        # max_values_ref = max_values_dist > max_thresh
        # min_values_ref = min_values_dist > min_thresh

        # Find mad based outliers for each batch of values (high_val, low_val).
        # mad_based_outlier returns boolean (True -> is outlier)
        max_values_ref = mad_based_outlier(high_val, max_thresh)
        min_values_ref = mad_based_outlier(low_val, min_thresh)

        # Extract elements from high_val_idx and low_val_idx that satisfy max_values_ref == True condition
        # These elements are indexes of outliers in the array of original values for this column (col_vec) that have outliers
        upper_outliers_idx = np.extract(condition=max_values_ref, arr=high_val_idx)
        lower_outliers_idx = np.extract(condition=min_values_ref, arr=low_val_idx)

        # if plot_mad_dist:
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     fig, ax = plt.subplots(2, 1)
        #     sns.distplot(high_val, ax=ax[0], rug=False, hist=False)
        #     sns.distplot(low_val, ax=ax[1], rug=False, hist=False)
        #     ax[0].plot(col_vec[upper_outliers_idx], np.zeros_like(col_vec[upper_outliers_idx]), 'ro', clip_on=False)
        #     ax[1].plot(col_vec[lower_outliers_idx], np.zeros_like(col_vec[lower_outliers_idx]), 'ro', clip_on=False)
        #     plt.draw()

        # Update ref_bool with True for outlier indexes.
        ref_bool[upper_outliers_idx, c] = True
        ref_bool[lower_outliers_idx, c] = True

        del high_val_idx, low_val_idx
        del high_val, low_val
        # del max_values_dist, min_values_dist
        del upper_outliers_idx, lower_outliers_idx

    # flatten ref_bool
    return np.ravel(ref_bool)


def distribuction_based_outliers(values,
                                 max_threshold=0.05,
                                 min_threshold=0.05,
                                 upper_percentile=95,
                                 lower_percentile=5):
    # ref_bool = np.full(shape=values.shape, fill_value=False, dtype="bool")
    aux_values = np.copy(values)

    # Calculates the abs. difference between the maximum value of the timeseries and the percentile upper_percentile.
    # Then, divides by the percentile upper_percentile.
    max_dist = abs((aux_values.max() - np.percentile(aux_values, upper_percentile)) / np.percentile(aux_values, upper_percentile))
    max_val = []

    # If the max_dist is higher that a specified maximum threshold
    # (that is, the maximum value of the series is far superior than the percentile upper_percentile)
    # Stores the current maximum in a aux. list and deletes that value from the series (it was considered an outlier)
    # Then, calculates the max_dist again (same procedure as before, now without the previous outlier)
    while max_dist > max_threshold:
        max_val.append(aux_values.max())
        aux_values = np.delete(aux_values, aux_values.argmax())
        percentile = np.percentile(aux_values, upper_percentile)
        if percentile == 0:
            percentile += (10**-10)
        max_dist = abs((aux_values.max() - percentile)) / percentile

    # Same procedure as before, now applied to the lower values
    # (changes .max() to .min() and upper_percentile to lower_percentile)
    min_dist = abs(aux_values.min() - np.percentile(aux_values, lower_percentile)) / abs(np.percentile(aux_values, lower_percentile))
    min_val = []

    while min_dist > min_threshold:
        min_val.append(aux_values.min())
        aux_values = np.delete(aux_values, aux_values.argmin())
        percentile = np.percentile(aux_values, lower_percentile)
        if percentile == 0:
            percentile += (10**-10)
        min_dist = abs((aux_values.min() - percentile)) / abs(percentile)

    # With max_val and min_val lists filled with the outlier references compares those with initial array of values
    # creates two reference arrays "ref_bool_max" and "ref_bool_min" with boolean type elements.
    # True - Value is outlier / False - Value is not outlier
    ref_bool_max = np.isin(values, max_val)
    ref_bool_min = np.isin(values, min_val)

    ref_bool = [x or y for x, y in zip(ref_bool_max, ref_bool_min)]

    # Flatten ref_bool
    return np.ravel(ref_bool)

