import numpy as np
import pymc as pm
import pytensor.tensor as pt


class Matern32Haversine(pm.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Great circle distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def great_circle_distance(self, X, Xs=None):
        if Xs is None:
            Xs = X

        # Assume first column is longitude and second is latitude
        lat1_ = pt.deg2rad(X[:, 1])
        lon1_ = pt.deg2rad(X[:, 0])
        lat2_ = pt.deg2rad(Xs[:, 1])
        lon2_ = pt.deg2rad(Xs[:, 0])

        # Reshape lon/lat into 2D
        lat1 = lat1_[:, None]
        lon1 = lon1_[:, None]

        # Elementwise differnce of lats and lons
        dlat = lat2_ - lat1
        dlon = lon2_ - lon1

        # Compute haversine
        d = pt.sin(dlat / 2) ** 2 + pt.cos(lat1) * pt.cos(lat2_) * pt.sin(dlon / 2) ** 2
        return self.r * 2 * pt.arcsin(pt.sqrt(d)) + 1e-12

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.great_circle_distance(X, Xs)
        return (1.0 + np.sqrt(3.0) * r / self.ls) * pt.exp(-np.sqrt(3.0) * r / self.ls)


class Matern32Chordal(pm.gp.cov.Stationary):
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * pt.stack(
            [
                pt.cos(lonlat[..., 0]) * pt.cos(lonlat[..., 1]),
                pt.sin(lonlat[..., 0]) * pt.cos(lonlat[..., 1]),
                pt.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = pt.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return pt.sqrt(pt.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(3.0) * r) * pt.exp(-np.sqrt(3.0) * r)
