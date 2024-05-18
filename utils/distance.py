import numpy as np

def great_circle_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lon1 : float
        Longitude of the first point
    lat1 : float
        Latitude of the first point
    lon2 : float
        Longitude of the second point
    lat2 : float
        Latitude of the second point
        
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])    
    dlon = lon2 - lon1
    dlat = lat2 - lat1    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

def compute_minimum_distance(lons1, lats1, lons2, lats2):
    """Compute the minimum distance between two sets of points
    
    Parameters
    ----------
    lons1 : array-like
        Longitudes of the first set of points
    lats1 : array-like
        Latitudes of the first set of points
    lons2 : array-like
        Longitudes of the second set of points
    lats2 : array-like
        Latitudes of the second set of points
    """
    lons1 = lons1[:, None]
    lats1 = lats1[:, None]
    lons2 = lons2[None, :]
    lats2 = lats2[None, :]
    dmat = great_circle_distance(lons1, lats1, lons2, lats2)

    return np.nan

def find_closest_point(distance, coords):
    '''Find the closest point on the coastline to the given coordinates
    
    Parameters
    ----------
    distance : pd.DataFrame
        DataFrame with columns 'lon', 'lat', 'distance'
    coords : tuple
        Tuple with longitude and latitude of the point to find the closest point to
    '''
    idx_min = distance.apply(lambda row: great_circle_distance(row['lon'], row['lat'], coords[0], coords[1]), axis=1).argmin()
    return distance.iloc[idx_min]['distance']


def convert_coords_to_distance(df, lon, lat, distance):
    '''Convert coordinates to distance along the coastline
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'Longitude', 'Latitude' to convert to distance
    lon : str
        Name of the column with the longitude
    lat : str
        Name of the column with the latitude
    
    '''
    return df.apply(lambda row: find_closest_point(distance, (row[lon], row[lat])), axis=1)
    