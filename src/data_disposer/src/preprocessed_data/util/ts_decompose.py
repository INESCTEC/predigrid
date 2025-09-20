
import numpy as np
from pandas import DataFrame


def __adapted_op(data,
                 strftime_ref,
                 is_season=False,
                 method='additive',
                 return_df=False,
                 smoothing_window=None,
                 centered_window=False):

    data = data.copy()
    data_old_idx = data.index
    data.index = data.index.strftime(strftime_ref)
    pseudo_op = data.groupby(data.index).apply(np.nanmedian)

    for tref in pseudo_op.index:
        data.loc[tref, ] = pseudo_op.loc[tref, ]

    if smoothing_window is not None:
        data = data.rolling(window=smoothing_window, center=centered_window, min_periods=1).mean()

    if is_season:
        if method.lower() == 'additive':
            data -= np.nanmedian(data)
        elif method.lower() == 'multiplicative':
            data /= (np.nanmedian(data) + 0.000000001)

    if return_df:
        data.index = data_old_idx
        return data
    else:
        return data.values.ravel()


def astl_decomposition(data,
                       method="additive",
                       trend_strftime="%Y-%m-%d",
                       season_strftime="%w-%H",
                       smoothing_window=None,
                       centered_smoothing_window=False):
    """
    Adapted version of stl decomposition.

    Args:
        data:
        method:
        trend_strftime:
        season_strftime:
        smoothing_window:
        centered_smoothing_window:

    Returns:

    """
    data = data.copy()
    trend = __adapted_op(data, trend_strftime, False, method, return_df=True, smoothing_window=smoothing_window,
                         centered_window=centered_smoothing_window)

    if method.lower() == "additive":
        detrended = data - trend
    elif method.lower() == "multiplicative":
        detrended = data / trend
    else:
        exit("ERROR: %s not a valid method" % method)

    seasonal = __adapted_op(detrended, season_strftime, True, method)
    observed = data.values.ravel()
    detrended = detrended.values.ravel()
    trend = trend.values.ravel()

    if method.lower() == "additive":
        resid = detrended - seasonal
    elif method.lower() == "multiplicative":
        resid = observed / seasonal / trend
    else:
        exit("ERROR: %s not a valid method" % method)

    container = DataFrame({"observed": observed,
                           "seasonal": seasonal,
                           "residual": resid,
                           "trend": trend},
                          index=data.index)

    return container

