import numpy as np
import pytest
from astropy.modeling import models

from specreduce.utils.synth_data import make_2d_trace_image
from specreduce.tracing import Trace, FlatTrace, ArrayTrace, FitTrace

IM = make_2d_trace_image()


# test basic trace class
def test_basic_trace():
    t_pos = IM.shape[0] / 2
    t = Trace(IM)

    assert t[0] == t_pos
    assert t[0] == t[-1]
    assert t.shape[0] == IM.shape[1]

    t.shift(100)
    assert t[0] == 600.

    t.shift(-1000)
    assert np.ma.is_masked(t[0])


# test flat traces
def test_flat_trace():
    t = FlatTrace(IM, 550.)

    assert t.trace_pos == 550
    assert t[0] == 550.
    assert t[0] == t[-1]

    t.set_position(400.)
    assert t[0] == 400.

    t.set_position(-100)
    assert np.ma.is_masked(t[0])


# test array traces
def test_array_trace():
    arr = np.ones_like(IM[0]) * 550.
    t = ArrayTrace(IM, arr)

    assert t[0] == 550.
    assert t[0] == t[-1]

    t.shift(100)
    assert t[0] == 650.

    t.shift(-1000)
    assert np.ma.is_masked(t[0])

    arr_long = np.ones(5000) * 550.
    t_long = ArrayTrace(IM, arr_long)

    assert t_long.shape[0] == IM.shape[1]

    arr_short = np.ones(50) * 550.
    t_short = ArrayTrace(IM, arr_short)

    assert t_short[0] == 550.
    assert np.ma.is_masked(t_short[-1])
    assert t_short.shape[0] == IM.shape[1]


# test fitted traces
@pytest.mark.filterwarnings("ignore:Model is linear in parameters")
def test_fit_trace():
    # create image (process adapted from compare_extractions.ipynb)
    np.random.seed(7)
    nrows = 200
    ncols = 160
    sigma_pix = 4
    sigma_noise = 1

    col_model = models.Gaussian1D(amplitude=1, mean=nrows/2, stddev=sigma_pix)
    noise = np.random.normal(scale=sigma_noise, size=(nrows, ncols))

    index_arr = np.tile(np.arange(nrows), (ncols, 1))
    img = col_model(index_arr.T) + noise

    # calculate trace on normal image
    t = FitTrace(img, bins=20)

    # test shifting
    shift_up = int(-img.shape[0]/4)
    t_shift_up = t.trace + shift_up

    shift_out = img.shape[0]

    t.shift(shift_up)
    assert np.sum(t.trace == t_shift_up) == t.trace.size, 'valid shift failed'

    t.shift(shift_out)
    assert t.trace.mask.all(), 'invalid values not masked'

    # test peak_method and trace_model options
    tg = FitTrace(img, bins=20,
                  peak_method='gaussian', trace_model=models.Legendre1D(3))
    tc = FitTrace(img, bins=20,
                  peak_method='centroid', trace_model=models.Chebyshev1D(2))
    tm = FitTrace(img, bins=20,
                  peak_method='max', trace_model=models.Spline1D(degree=3))
    # traces should all be close to 100
    # (values may need to be updated on changes to seed, noise, etc.)
    assert np.max(abs(tg.trace-100)) < sigma_pix
    assert np.max(abs(tc.trace-100)) < 3 * sigma_pix
    assert np.max(abs(tm.trace-100)) < 6 * sigma_pix
    with pytest.raises(ValueError):
        t = FitTrace(img, peak_method='invalid')

    # create same-shaped variations of image with invalid values
    img_all_nans = np.tile(np.nan, (nrows, ncols))

    window = 10
    guess = int(nrows/2)
    img_win_nans = img.copy()
    img_win_nans[guess - window:guess + window] = np.nan

    # ensure float bin values trigger a warning but no issues otherwise
    with pytest.warns(UserWarning, match='TRACE: Converting bins to int'):
        FitTrace(img, bins=20., trace_model=models.Polynomial1D(2))

    # ensure non-equipped models are rejected
    with pytest.raises(ValueError, match=r'trace_model must be one of'):
        FitTrace(img, trace_model=models.Hermite1D(3))

    # ensure a bin number below 4 is rejected
    with pytest.raises(ValueError, match='bins must be >= 4'):
        FitTrace(img, bins=3)

    # ensure a bin number below degree of trace model is rejected
    with pytest.raises(ValueError, match='bins must be > '):
        FitTrace(img, bins=4, trace_model=models.Chebyshev1D(5))

    # ensure number of bins greater than number of dispersion pixels is rejected
    with pytest.raises(ValueError, match=r'bins must be <'):
        FitTrace(img, bins=ncols + 1)

    # error on trace of otherwise valid image with all-nan window around guess
    with pytest.raises(ValueError, match='pixels in window region are masked'):
        FitTrace(img_win_nans, guess=guess, window=window)

    # error on trace of all-nan image
    with pytest.raises(ValueError, match=r'image is fully masked'):
        FitTrace(img_all_nans)

    # could try to catch warning thrown for all-nan bins
