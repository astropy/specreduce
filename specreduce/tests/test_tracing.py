import numpy as np
import pytest
from astropy.modeling import models
from astropy.nddata import NDData
import astropy.units as u
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


class TestMasksTracing():

    def mk_img(self, nrows=200, ncols=160):

        np.random.seed(7)

        sigma_pix = 4
        sigma_noise = 1

        col_model = models.Gaussian1D(amplitude=1, mean=nrows/2, stddev=sigma_pix)
        noise = np.random.normal(scale=sigma_noise, size=(nrows, ncols))

        index_arr = np.tile(np.arange(nrows), (ncols, 1))
        img = col_model(index_arr.T) + noise

        return img

    def test_window_fit_trace(self):

        """This test function will test that masked values are treated correctly in
        FitTrace, and produce the correct results and warning messages based on
        `peak_method`."""
        img = self.mk_img()

        # create same-shaped variations of image with invalid values
        nrows = 200
        ncols = 160
        img_all_nans = np.tile(np.nan, (nrows, ncols))

        window = 10
        guess = int(nrows/2)
        img_win_nans = img.copy()
        img_win_nans[guess - window:guess + window] = np.nan

        # error on trace of otherwise valid image with all-nan window around guess
        with pytest.raises(ValueError, match='pixels in window region are masked'):
            FitTrace(img_win_nans, guess=guess, window=window)

        # error on trace of all-nan image
        with pytest.raises(ValueError, match=r'image is fully masked'):
            FitTrace(img_all_nans)

    @pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
    @pytest.mark.filterwarnings("ignore:Model is linear in parameters")
    @pytest.mark.filterwarnings("ignore:All pixels in bins")
    def test_fit_trace_all_nan_cols(self):

        # make sure that the actual trace that is fit is correct when
        # all-masked bin peaks are set to NaN
        img = self.mk_img(nrows=10, ncols=11)

        img[:, 7] = np.nan
        img[:, 4] = np.nan
        img[:, 0] = np.nan

        # peak_method = 'max'
        truth = [1.6346154, 2.2371795, 2.8397436, 3.4423077, 4.0448718,
                 4.6474359, 5.25, 5.8525641, 6.4551282, 7.0576923,
                 7.6602564]
        max_trace = FitTrace(img, peak_method='max')
        np.testing.assert_allclose(truth, max_trace.trace)

        # peak_method = 'gaussian'
        truth = [1.947455, 2.383634, 2.8198131, 3.2559921, 3.6921712,
                 4.1283502, 4.5645293, 5.0007083, 5.4368874, 5.8730665,
                 6.3092455]
        max_trace = FitTrace(img, peak_method='gaussian')
        np.testing.assert_allclose(truth, max_trace.trace)

        # peak_method = 'centroid'
        truth = [2.5318835, 2.782069, 3.0322546, 3.2824402, 3.5326257,
                 3.7828113, 4.0329969, 4.2831824, 4.533368, 4.7835536,
                 5.0337391]
        max_trace = FitTrace(img, peak_method='centroid')
        np.testing.assert_allclose(truth, max_trace.trace)

    @pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
    @pytest.mark.filterwarnings("ignore:Model is linear in parameters")
    def test_warn_msg_fit_trace_all_nan_cols(self):

        img = self.mk_img()

        # test that warning (dependent on choice of `peak_method`) is raised when a
        # few bins are masked, and that theyre listed individually
        mask = np.zeros(img.shape)
        mask[:, 100] = 1
        mask[:, 20] = 1
        mask[:, 30] = 1
        nddat = NDData(data=img, mask=mask, unit=u.DN)

        match_str = 'All pixels in bins 20, 30, 100 are fully masked. Setting bin peaks to NaN.'

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method='max')

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method='centroid')

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method='gaussian')

        # and when many bins are masked, that the message is consolidated
        mask = np.zeros(img.shape)
        mask[:, 0:21] = 1
        nddat = NDData(data=img, mask=mask, unit=u.DN)
        with pytest.warns(UserWarning, match='All pixels in bins '
                          '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 20 are '
                          'fully masked. Setting bin peaks to NaN.'):
            FitTrace(nddat)
