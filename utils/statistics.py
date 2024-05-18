import numpy as np
import xarray as xr


def _np_weighted_percentile(data, weights, percentile):
    """
    Compute the weighted percentile of the given data.
    :param data: numpy array of data
    :param weights: numpy array of weights
    :param percentile: percentile to compute
    :return: weighted percentile
    """
    # Ensure numpy arrays
    data, weights = map(np.asarray, (data, weights))
    assert data.size == weights.size, "data and weights should have the same size"
    # Remove nan values
    mask_data = ~np.isnan(data)
    mask_weights = ~np.isnan(weights)
    mask = mask_data & mask_weights
    data = data[mask]
    weights = weights[mask]
    assert np.all(weights >= 0), "weights should be non-negative"
    # Normalize the weights to sum to 1
    weights = weights / np.sum(weights)

    # Kish's effective sample size (L Kish. Survey sampling. 1965)
    ess = np.sum(weights) ** 2 / np.sum(weights**2)

    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(weights)
    percentile_value = total_weight * np.array(percentile)
    n_samples = data.shape[0]
    if n_samples == 0:
        return np.array([np.nan] * len(percentile)), ess
    elif n_samples == 1:
        return np.array([data[0]] * len(percentile)), ess
    else:
        return np.interp(percentile_value, cum_weights, sorted_data), ess


def weighted_percentile(data, weights, percentile, dim):
    """Compute the weighted percentile of the given data.

    Parameters
    ----------
    data : xr.DataArray
        Data array
    weights : xr.DataArray
        Weights array
    percentile : float or list of floats
        Percentile to compute
    dim : str
        List of core dimensions

    Returns
    -------
    xr.DataArray
        Weighted percentile
    xr.DataArray
        Effective sample size

    """
    weights = weights.broadcast_like(data)

    wp, sample_size = xr.apply_ufunc(
        _np_weighted_percentile,
        data,
        weights,
        input_core_dims=[dim, dim],
        output_core_dims=[["quantile"], []],
        exclude_dims=set(dim),
        vectorize=True,
        kwargs={"percentile": percentile},
    )
    wp = wp.assign_coords({"quantile": percentile})
    return wp, sample_size
