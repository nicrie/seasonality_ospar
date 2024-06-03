import numpy as np
import pymc as pm
import scipy.stats as st
import statsmodels.stats.multitest as ssm
import xarray as xr
from scipy.optimize import minimize_scalar


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


def _np_find_touching_hdi_size(samples1, samples2, initial_mass=0.95, tol=1e-5):
    """
    Find the maximal non-overlapping highest density interval (MNO-HDI).

    The MNO-HDI is determined by the HDI size for which the HDIs of the two
    distributions do not overlap. In other words, determine the size at which
    the boundaries of the HDIs of two distributions just touch.
    """

    def overlap_loss(hdi_mass, dim="n"):
        hdi1 = pm.hdi(samples1, hdi_prob=hdi_mass)
        hdi2 = pm.hdi(samples2, hdi_prob=hdi_mass)

        # Loss function based on the distance between HDI boundaries
        if hdi2[1] > hdi1[1]:
            return np.abs((hdi1[1] - hdi2[0]))
        else:
            return np.abs((hdi2[1] - hdi1[0]))

    result = minimize_scalar(
        overlap_loss, bounds=(0, 1), method="bounded", options={"xatol": tol}
    )
    if result.success:
        return result.x  # Returns the HDI mass for which the boundaries touch
    else:
        raise ValueError("Optimization did not converge")


def find_touching_hdi_size(da1, da2, dim):
    """
    Find the maximal non-overlapping highest density interval (MNO-HDI).

    The MNO-HDI is determined by the HDI size for which the HDIs of the two
    distributions do not overlap. In other words, determine the size at which
    the boundaries of the HDIs of two distributions just touch.
    """

    return xr.apply_ufunc(
        _np_find_touching_hdi_size,
        da1,
        da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
    )


def _np_bh_correction(arr, alpha=0.05):
    """
    Perform Benjamini-Hochberg correction on an array of p-values.

    Parameters
    ----------
    arr : ndarray
        Array of p-values.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    ndarray
        Array of corrected p-values.

    """
    pvals = np.zeros_like(arr) * np.nan
    if np.isnan(arr).all():
        return pvals
    mask = ~np.isnan(arr)
    arr = arr[mask]
    pvals[mask] = ssm.multipletests(arr, alpha=alpha, method="fdr_bh")[1]
    return pvals


def benjamini_hochberg_correction(arr, dim, alpha=0.05):
    """
    Apply the Benjamini-Hochberg correction to adjust p-values for multiple comparisons.

    Parameters
    ----------
    arr : xarray.DataArray
        The input array containing p-values.
    dim : str
        The dimension along which the correction should be applied.
    alpha : float, optional
        The significance level. Default is 0.05.

    Returns
    -------
    xarray.DataArray
        The corrected p-values.

    """
    return xr.apply_ufunc(
        _np_bh_correction,
        arr,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        kwargs={"alpha": alpha},
    )


def _np_kruskal_wallis(X, min_samples=5):
    """Perform Kruskal-Wallis test on each group in X.

    Parameters
    ----------
    X : np.ndarray
        Array with shape (n_groups, n_samples_per_group)
    min_samples : int, optional
        Minimum number of samples required to perform the test, by default 5

    Returns
    -------
    float
        p-value of the Kruskal-Wallis test

    """
    n_samples_per_group = (~np.isnan(X)).sum(axis=1)
    has_enough_samples = n_samples_per_group >= min_samples
    # set columns with insufficient samples to NaN
    X = np.where(has_enough_samples, X.T, np.nan).T

    is_valid_group = (~np.isnan(X)).any(axis=1)
    n_valid_groups = is_valid_group.sum()

    if n_valid_groups < 2:
        return np.nan
    else:
        try:
            statistic, pval = st.kruskal(*X[is_valid_group], nan_policy="omit")
            return pval
        except ValueError:
            # If all values are the same, the test cannot be performed
            # Set p-value to 1
            return 1.0


def kruskal_wallis_test(da, min_samples=5):
    """
    Perform the Kruskal-Wallis test on a given data array.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array.
    min_samples : int, optional
        The minimum number of samples required for the test. Default is 5.

    Returns
    -------
    xarray.DataArray
        The result of the Kruskal-Wallis test.

    """
    return xr.apply_ufunc(
        _np_kruskal_wallis,
        da,
        input_core_dims=[["season", "year"]],
        output_core_dims=[[]],
        exclude_dims=set(["season", "year"]),
        vectorize=True,
        kwargs={"min_samples": min_samples},
    )


def _np_mann_whitney_test(x, y):
    """
    Perform a Mann-Whitney U test using numpy.

    Parameters:
        x (array-like): The first sample.
        y (array-like): The second sample.

    Returns:
        float: The rank-biserial correlation coefficient.
        float: The p-value of the test.

    Notes:
        - This function handles missing values by omitting them.
        - The rank-biserial correlation coefficient is computed as 2 * U1 / (n1 * n2) - 1,
          where U1 is the U statistic and n1, n2 are the sample sizes of x and y, respectively.
    """
    if np.isnan(x).all() or np.isnan(y).all():
        return np.nan, np.nan
    n1 = (~np.isnan(x)).sum()
    n2 = (~np.isnan(y)).sum()
    U1, pvalue = st.mannwhitneyu(x, y, nan_policy="omit")
    # Compute effect size (rank-biserial correlation coefficient)
    r = 2 * U1 / (n1 * n2) - 1
    return r, pvalue


def mann_whitney_test(da1, da2, dim):
    """Perform Mann-Whitney U test for each combination of beaches.

    Parameters
    ----------
    da1 : xr.DataArray
        Array with shape (n_beaches, n_years)
    da2 : xr.DataArray
        Array with shape (n_beaches, n_years)
    dim : str
        Dimension along which to perform the test

    Returns
    -------
    xr.DataArray
        Effect size of the Mann-Whitney U test
    xr.DataArray
        p-values of the Mann-Whitney U test


    """
    res = xr.apply_ufunc(
        _np_mann_whitney_test,
        da1,
        da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    return res
