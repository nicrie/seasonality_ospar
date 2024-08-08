import numpy as np
import pytensor.tensor as pt
import scipy.stats as st
import xarray as xr


def _np_yeojohnson(arr):
    arr = arr.flatten()
    arr = arr[~np.isnan(arr)]
    lmbda = st.yeojohnson_normmax(arr)
    if lmbda < 0:
        print("Warning: Yeo-Johnson lambda < 0, using log transformation instead.")
        return 0
    else:
        return lmbda


def estimate_lmbda(da, dim: list[str]):
    """Estimate the lambda parameter for the Yeo-Johnson transformation.

    Parameters
    ----------
    da : xr.DataArray
        The input data.
    dim : str
        The dimension along which to estimate the lambda parameter. For example,
        dim=["beach_id", "year"]

    Returns
    -------
    xr.DataArray
        The lambda parameter.
    """
    return xr.apply_ufunc(
        _np_yeojohnson,
        da,
        input_core_dims=[dim],
        output_core_dims=[[]],
        exclude_dims=set(dim),
        vectorize=True,
    )


def yeojohnson(da, lmbda):
    if lmbda == 0:
        return np.log(da + 1)
    else:
        return ((da + 1) ** lmbda - 1) / lmbda


def yeojohnson_inv(da, lmbda):
    if lmbda == 0:
        return np.exp(da) - 1
    else:
        return (da * lmbda + 1) ** (1 / lmbda) - 1


def pt_yeojohnson(da, lmbda):
    if lmbda == 0:
        return pt.log(da + 1)
    else:
        return (pt.pow(da + 1, lmbda) - 1) / lmbda


def pt_yeojohnson_inv(da, lmbda):
    if lmbda == 0:
        return pt.exp(da) - 1
    else:
        return pt.pow(da * lmbda + 1, 1 / lmbda) - 1
