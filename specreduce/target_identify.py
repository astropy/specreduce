from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import logging
import numpy as np
from scipy.stats import sigma_clip



def identify_targets(data, nfind=1, axis=1, background_threshold=3, model_name='gaussian'):

    # assuming target is relatively well aligned with the spatial axis
    spatial_profile = np.median(data, axis=axis)


    # remove background
    clipped_spatial_profile = sigma_clip()

    # identify peaks


    # filter peaks


    # construct model and fit


    # return model or set of parameters such as target center, width.

