import numpy as np
import regionmask as rm

boundaries_nea = np.array([[-20, 35], [-6, 35], [15, 60], [15, 65], [-20, 65]])
nea_mask = rm.Regions([boundaries_nea], names=["North East Atlantic"], abbrevs=["NEA"])
