import numpy as np
import xarray as xr
import scipy.ndimage 
from tqdm import tqdm


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * sig**2))

def great_circle_distance(grid : xr.DataArray, lonlat : tuple):
    RADIUS_EARTH = 6371.
    
    lon1 = np.deg2rad(grid.lon)
    lat1 = np.deg2rad(grid.lat)
    
    lon2 = np.deg2rad(lonlat[0])
    lat2 = np.deg2rad(lonlat[1])
    
    sin_phi = np.sin(lat1) * np.sin(lat2)
    cos_phi = np.cos(lat1) * np.cos(lat2)
    cos_lbda = np.cos(abs(lon1 - lon2))
    
    tmp = sin_phi + cos_phi * cos_lbda
    
    # Ensure that the expression is bounded between -1...+1
    tmp = tmp.where(tmp < 1, 1)
    
    central_angle = np.arccos(tmp)
    # Compute distance 
    distance =  RADIUS_EARTH * central_angle
    return distance


def _numpy_weighted_mean(x, weights, min_samples=None):
    '''General weighed mean / standard deviation of a sample `x`.
    
    '''
    V1 = np.nansum(weights)
    V2 = np.nansum((weights**2))
    V = V1 - (V2 / V1)
    N_eff = V1**2 / V2
    mean = np.nansum(x * weights) / V1
    wss = np.nansum((weights * (x - mean)**2))
    std = np.sqrt(wss / V)
    if min_samples is not None:
        if N_eff < min_samples:
            mean = np.nan
            std = np.nan
    return mean, std, N_eff

def weighted_mean(data, weights, dim, min_samples=None):
    '''General weighted mean / standard deviation of a data sample using DataArray.
    
    '''
    weights = weights.where(data.notnull())
    return xr.apply_ufunc(
        _numpy_weighted_mean,
        data,
        weights,
        min_samples,
        input_core_dims=[dim, dim, []],
        output_core_dims=[[], [], []],
        vectorize=True
    )


def weights_spatial_distance(grid, location, sigma=1, mask=None, truncate=None):
    '''Weight matrix for a spatial field for a given location.

    The weights are computed based on the great circle distance from the location.

    Parameters
    ----------

    sigma : float
        Standard deviation (in km) for Gaussian
    truncate : float
        Distance (in km) at which to truncate
    '''

    # Compute great circle distance to location
    distance = great_circle_distance(grid, lonlat=location)

    # Define weights, 1 if grid is centered on location
    weights = gaussian(distance, mu=0, sig=sigma)

    # Cut off points very far away
    if truncate is not None:
        weights = weights.where(distance < truncate)

    # Mask out invalid cells (e.g. ocean)
    if mask is not None:
        weights = weights.where(mask)
    return weights


def weighted_spatial_mean(model, location, sigma=100, mask=None, truncate=None, return_std=False, min_samples=None):
    '''Compute weighted spatial mean for a given location. 
    
    Parameters
    ----------
    sigma : float
        Standard deviation (in km) used for local Gaussian smooting if point falls between two grid points
    truncate : float
        Maximal distance (in km) used for local Gaussian smoothing if points falls between two grid points 
    min_samples : int
        minimum number of non-NaN samples to calculate spatial mean. if there are less than minimum_number
        return NaN
    '''

    weights = weights_spatial_distance(model, location, sigma=sigma, mask=mask, truncate=truncate)
    # Only continue if more than `min_samples` non-NaN weights
    if min_samples is not None:
        n_samples = weights.notnull().sum()
        if n_samples < min_samples:
            return np.nan
    # Remove weights where there is no data
    weights = weights.where(model.notnull())
    # Calculate effective sample size
    V1 = weights.sum(['lat', 'lon'])
    # Weighted mean
    mean = (weights * model).sum(('lat', 'lon')) / V1
    # Weighted standard deviation
    if return_std:
        V2 = (weights**2).sum(['lat', 'lon'])
        effective_samples = V1 - (V2 / V1)
        wss = (weights * (model - mean)**2).sum(('lat', 'lon'))
        std = np.sqrt(wss / effective_samples)
        return mean, std
    # mean.name = model.name + '_mean'
    # std.name = model.name + '_std'
    # return xr.Dataset({mean.name: mean, std.name: std})
    return mean


def bootstrap_lonlat(
        da, lonlat, n_boots=10, radius=1000, sigma=100,
        mask=None, desc=None, truncate=None, min_samples=None
    ):
    ''' For a given location bootstrap within a given radius (in km)
    
    Parameters
    ----------

    radius : float
        radius (in km) defines a circle around location from which to create bootstrap samples
    sigma : float
        standard deviation (in km) of Gaussian smoothing applied to bootstrap sample if between grid points
    truncate : float
        maximum distance (in km) used for Gaussian smooting
    min_sample : int
        minimum number of weights necesarry to calculate a mean, otherwise return NaN
    
    '''
    bst = da.isel(lon=0, lat=0).drop(('lon', 'lat')) * np.nan
    bst = bst.expand_dims({
        'bst': np.arange(1, n_boots + 1)}
    ).copy(deep=True)

    # Convert radius from km to degree longitude
    radius_in_deg = radius / 111.2
    for n in tqdm(bst.coords['bst'].values, desc=desc):
        # Uniform random pertubation within a circle of radius of r
        # Sqrt is necessary to account for higher probabilty
        # to generate a sample around the center point
        r = np.sqrt(np.random.uniform(0, radius_in_deg))
        phi = np.random.uniform(0, 2 * np.pi)
        # Correct for smaller areas in higher latitudes
        alpha = np.cos(np.deg2rad(lonlat[1]))
        x = r * np.cos(phi)
        y = alpha * r * np.sin(phi)
        pertubation = np.array([x, y])

        new_lonlat = tuple(np.array(lonlat) + pertubation)

        temp = weighted_spatial_mean(
            da, new_lonlat, sigma=sigma, mask=mask, truncate=truncate, min_samples=min_samples
        )
        bst.loc[dict(bst=n)] = temp
    return bst


def bootstrap_locations(
        da, locations, n_boots=10, radius=1000, sigma=100,
        mask=None, truncate=None, min_samples=None
    ):
    ''' For a given locations bootstrap within a given radius (in km)
    
    Parameters
    ----------

    radius : float
        radius (in km) defines a circle around location from which to create bootstrap samples
    sigma : float
        standard deviation (in km) of Gaussian smoothing applied to bootstrap sample if between grid points
    truncate : float
        maximum distance (in km) used for Gaussian smooting
    min_sample : int
        minimum number of weights necesarry to calculate a mean, otherwise return NaN
    
    '''
    bst_regions = da.isel(lon=0, lat=0)
    bst_regions = np.nan * bst_regions.expand_dims({
        'region': list(locations.keys()),
        'bst': range(1, n_boots + 1),
    }).copy(deep=True)
    for region, lonlat in locations.items():
        bst_regions.loc[dict(region=region)] = bootstrap_lonlat(
            da, lonlat, n_boots=n_boots, sigma=sigma, radius=radius,
            mask=mask, desc=region, truncate=truncate, min_samples=min_samples
        )

    return bst_regions


def temporal_gaussian_mean(da, dim, window, std, center=True):
    '''DataArray implementation of Gaussian rolling mean.

    This implementation considers ``window`` to specifiy the number of time
    steps included to calculate the weighted mean (as it is normal for all
    xarray and pandas implementation). It DOES NOT calculate the
    weighted mean based on the actual TIME period.
    '''
    # Create weights using Gaussian kernel
    x_win = np.arange(window)
    mu = x_win[len(x_win) // 2]
    weights = gaussian(x_win, mu, std)
    weights = xr.DataArray(weights, dims=['window'])
    # Create weights for each time step; mask out the weights at the boundaries
    # of the time series
    weights = weights.expand_dims({dim : da.coords[dim]})
    n_samples, n_win = weights.shape
    temp1 = np.tri(n_win, n_samples, - n_win // 2, dtype=bool)
    temp2 = np.tri(n_win, n_samples, (n_samples - (n_win // 2) - 1), dtype=bool)
    mask_weights = np.logical_not(temp1) & temp2
    mask_weights = mask_weights[::-1, :]
    mask_weights = xr.DataArray(mask_weights.T, dims=weights.dims, coords=weights.coords)
    # Construct Gaussian window along time dimension
    da_windowed = da.rolling({dim : window}, center=center).construct('window')
    has_data = da_windowed.notnull()
    da_mean = (da_windowed.fillna(0) * weights).sum('window')
    da_mean = da_mean / weights.where(mask_weights & has_data).sum('window')
    return da_mean
    # Effective sample size based on weights
    # V1 = weights.where(mask_weights).sum('window')
    # V2 = (weights**2).where(mask_weights).sum('window')
    # effective_samples = V1 - (V2 / V1)
    # Weighted standard deviation
    # da_cntrd = (da - da_mean)**2
    # da_ss = da_cntrd.rolling({dim : window}, center=center).construct('window')
    # da_ss = (da_ss.fillna(0) * weights).sum('window')
    # da_var = da_ss / effective_samples
    # da_std = np.sqrt(da_var)
    # da_ci = (da_std / np.sqrt(V1))
    # return xr.Dataset({
    #     da.name : da_mean,
    #     da.name + '_std' : da_ci,
    # })


def gaussian_smoothing_with_nan(arr, sigma, truncate):
    '''2D Gaussian smoothing of a 2D  numpy array taking into account NaNs.

    Source: https://stackoverflow.com/a/36307291/3050730

    No worries if you receive a RuntimeWarning, that's normal!
    
    '''
    U = arr.copy()

    V = U.copy()
    V[np.isnan(U)] = 0
    VV=scipy.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z = VV / WW

    # Reinsert original NaNs
    Z[np.isnan(U)] = np.nan

    return Z

def gaussian_2d_smoothing(da, dims, sigma, truncate):
    '''xarray wrapper for 2d Gaussian smoothing with NaNs.

    You need to specify two spatial dimensions e.g. ``dims=['lon', 'lat']``.
    
    '''
    return xr.apply_ufunc(
        gaussian_smoothing_with_nan,
        da,
        sigma,
        truncate,
        input_core_dims=[dims, [], []],
        output_core_dims=[dims],
        vectorize=True
    )




