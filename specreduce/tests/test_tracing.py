import numpy as np
import pytest
from astropy.modeling import fitting, models
from astropy.nddata import NDData
import astropy.units as u
from specreduce.utils.synth_data import make_2d_trace_image
from specreduce.tracing import Trace, FlatTrace, ArrayTrace, FitTrace

IM = make_2d_trace_image()


def mk_img(nrows=200, ncols=160, nan_slices=None, add_noise=True):
    """
    Makes a gaussian image for testing, with optional added gaussian
    nosie and optional data values set to NaN.
    """

    # NOTE: Will move this to a fixture at some point.

    sigma_pix = 4
    col_model = models.Gaussian1D(amplitude=1, mean=nrows / 2, stddev=sigma_pix)
    noise = 0
    if add_noise:
        np.random.seed(7)
        sigma_noise = 1
        noise = np.random.normal(scale=sigma_noise, size=(nrows, ncols))

    index_arr = np.tile(np.arange(nrows), (ncols, 1))
    img = col_model(index_arr.T) + noise

    if nan_slices:  # add nans in data
        for s in nan_slices:
            img[s] = np.nan

    return img * u.DN


# test basic trace class
def test_basic_trace():
    t_pos = IM.shape[0] / 2
    t = Trace(IM)

    assert t[0] == t_pos
    assert t[0] == t[-1]
    assert t.shape[0] == IM.shape[1]

    t.shift(100)
    assert t[0] == 600.0

    t.shift(-1000)
    assert np.ma.is_masked(t[0])


# test flat traces
def test_flat_trace():
    t = FlatTrace(IM, 550.0)

    assert t.trace_pos == 550
    assert t[0] == 550.0
    assert t[0] == t[-1]

    t.set_position(400.0)
    assert t[0] == 400.0


def test_negative_flat_trace_err():
    # make sure correct error is raised when trying to create FlatTrace with
    # negative trace_pos
    with pytest.raises(ValueError, match="must be positive."):
        FlatTrace(IM, trace_pos=-1)
    with pytest.raises(ValueError, match="must be positive."):
        FlatTrace(IM, trace_pos=0)


# test array traces
def test_array_trace():
    arr = np.ones_like(IM[0]) * 550.0
    t = ArrayTrace(IM, arr)

    assert t[0] == 550.0
    assert t[0] == t[-1]

    t.shift(100)
    assert t[0] == 650.0

    t.shift(-1000)
    assert np.ma.is_masked(t[0])

    arr_long = np.ones(5000) * 550.0
    t_long = ArrayTrace(IM, arr_long)

    assert t_long.shape[0] == IM.shape[1]

    arr_short = np.ones(50) * 550.0
    t_short = ArrayTrace(IM, arr_short)

    assert t_short[0] == 550.0
    assert np.ma.is_masked(t_short[-1])
    assert t_short.shape[0] == IM.shape[1]

    # make sure nonfinite data in input `trace` is masked
    arr[0:5] = np.nan
    t = ArrayTrace(IM, arr)
    assert np.all(t.trace.mask[0:5])
    assert np.all(t.trace.mask[5:] == 0)


# test fitted traces
@pytest.mark.filterwarnings("ignore:Model is linear in parameters")
def test_fit_trace():
    # create image (process adapted from compare_extractions.ipynb)
    np.random.seed(7)
    nrows = 200
    ncols = 160
    sigma_pix = 4
    sigma_noise = 1

    col_model = models.Gaussian1D(amplitude=1, mean=nrows / 2, stddev=sigma_pix)
    noise = np.random.normal(scale=sigma_noise, size=(nrows, ncols))

    index_arr = np.tile(np.arange(nrows), (ncols, 1))
    img = col_model(index_arr.T) + noise

    # calculate trace on normal image
    t = FitTrace(img, bins=20)

    # test shifting
    shift_up = int(-img.shape[0] / 4)
    t_shift_up = t.trace + shift_up

    shift_out = img.shape[0]

    t.shift(shift_up)
    assert np.sum(t.trace == t_shift_up) == t.trace.size, "valid shift failed"

    t.shift(shift_out)
    assert t.trace.mask.all(), "invalid values not masked"

    # test peak_method and trace_model options
    tg = FitTrace(img, bins=20, peak_method="gaussian", trace_model=models.Legendre1D(3))
    tc = FitTrace(img, bins=20, peak_method="centroid", trace_model=models.Chebyshev1D(2))
    tm = FitTrace(img, bins=20, peak_method="max", trace_model=models.Spline1D(degree=3))
    # traces should all be close to 100
    # (values may need to be updated on changes to seed, noise, etc.)
    assert np.max(abs(tg.trace - 100)) < sigma_pix
    assert np.max(abs(tc.trace - 100)) < 3 * sigma_pix
    assert np.max(abs(tm.trace - 100)) < 6 * sigma_pix
    with pytest.raises(ValueError):
        t = FitTrace(img, peak_method="invalid")

    window = 10
    guess = int(nrows / 2)
    img_win_nans = img.copy()
    img_win_nans[guess - window: guess + window] = np.nan

    # ensure float bin values trigger a warning but no issues otherwise
    with pytest.warns(UserWarning, match="TRACE: Converting bins to int"):
        FitTrace(img, bins=20.0, trace_model=models.Polynomial1D(2))

    # ensure non-equipped models are rejected
    with pytest.raises(ValueError, match=r"trace_model must be one of"):
        FitTrace(img, trace_model=models.Hermite1D(3))

    # ensure a bin number below 4 is rejected
    with pytest.raises(ValueError, match="bins must be >= 4"):
        FitTrace(img, bins=3)

    # ensure a bin number below degree of trace model is rejected
    with pytest.raises(ValueError, match="bins must be > "):
        FitTrace(img, bins=4, trace_model=models.Chebyshev1D(5))

    # ensure number of bins greater than number of dispersion pixels is rejected
    with pytest.raises(ValueError, match=r"bins must be <"):
        FitTrace(img, bins=ncols + 1)


def test_fit_trace_gaussian_all_zero():
    """
    Test fit_trace when peak_method is 'gaussian', which uses DogBoxLSQFitter
    for the fit for each bin peak and does not work well with all-zero columns.
    In this case, an all zero bin should fall back to NaN to for its'
    peak to be filtered out in the final fit for the trace.
    """
    img = mk_img(ncols=100)
    # add some all-zero columns so there is an all-zero bin
    img[:, 10:20] = 0

    t = FitTrace(img, bins=10, peak_method='gaussian')

    # this is a pretty flat trace, so make sure the fit reflects that
    assert np.all((t.trace >= 99) & (t.trace <= 101))


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
@pytest.mark.filterwarnings("ignore:Model is linear in parameters")
class TestMasksTracing:
    """
    There are three currently implemented options for masking in FitTrace: filter,
    omit, and zero_fill. Trace, FlatTrace, and ArrayTrace do not have
    `mask_treatment` options as input because masked/nonfinite values in the data
    are not relevant for those trace types as they are not affected by masked
    input data. The tests in this class test masking options for FitTrace, as
    well as some basic tests (errors, etc) for the other trace types.
    """

    def test_flat_and_basic_trace_mask(self):
        """
        Mask handling is not relevant for basic and flat trace - nans or masked
        values in the input image will not impact the trace value. The attribute
        should be initialized though, and be one of the valid options ([None]
        in this case), for consistancy with all other Specreduce operations.
        Note that unlike FitTrace, a fully-masked image should NOT result in an
        error raised because the trace does not depend on the data.
        """

        img = mk_img(nrows=5, ncols=5)

        basic_trace = Trace(img)
        assert basic_trace.mask_treatment is None

        flat_trace = FlatTrace(img, trace_pos=2)
        assert flat_trace.mask_treatment is None

        arr = [1, 2, np.nan, 3, 4]
        array_trace = ArrayTrace(img, arr)
        assert array_trace.mask_treatment is None

    def test_array_trace_masking(self):
        """
        The `trace` input to ArrayTrace can be a masked array, or an array
        containing nonfinite data which will be converted to a masked array.
        Additionally, if any padding needs to be added, the returned trace will
        be a masked array. Otherwise, it should be a regular array.

        Even though an ArrayTrace may have nans or masked values
        in the input 1D array for the trace, `mask_treatment_method` refers
        to how masked values in the input data should be treated. Nans / masked
        values passed in the array trace should be considered intentional, so
        also test that `mask_treatment` is initialized to None.
        """
        img = mk_img(nrows=10, ncols=10)

        # non-finite data in input trace should be masked out
        trace_arr = np.array((1, 2, np.nan, 4, 5))
        array_trace = ArrayTrace(img, trace_arr)
        assert array_trace.trace.mask[2]
        assert isinstance(array_trace.trace, np.ma.MaskedArray)

        # and combined with any input masked data
        trace_arr = np.ma.MaskedArray([1, 2, np.nan, 4, 5], mask=[1, 0, 0, 0, 0])
        array_trace = ArrayTrace(img, trace_arr)
        assert array_trace.trace.mask[0]
        assert array_trace.trace.mask[2]
        assert isinstance(array_trace.trace, np.ma.MaskedArray)

        # check that mask_treatment is None as there are no valid choices
        assert array_trace.mask_treatment is None

        # check that if array is fully finite and not masked, that the returned
        # trace is a normal array, not a masked array
        trace = ArrayTrace(img, np.ones(100))
        assert isinstance(trace.trace, np.ndarray)
        assert not isinstance(trace.trace, np.ma.MaskedArray)

        # ensure correct warning is raised when entire trace is masked.
        trace_arr = np.ma.MaskedArray([1, 2, np.nan, 4, 5], mask=[1, 1, 0, 1, 1])
        with pytest.warns(UserWarning, match=r"Entire trace array is masked."):
            array_trace = ArrayTrace(img, trace_arr)

    def test_fit_trace_fully_masked_image(self):
        """
        Test that the correct warning is raised when a fully masked image is
        encountered. Also test that when a non-fully masked image is provided,
        but `window` is set and the image is fully masked within that window,
        that the correct error is raised.
        """

        # make simple gaussian image.
        img = mk_img()

        # create same-shaped variations of image with nans in data array
        # which will be masked within FitTrace.
        nrows = 200
        ncols = 160
        img_all_nans = np.tile(np.nan, (nrows, ncols))

        # error on trace of all-nan image
        with pytest.raises(ValueError, match=r"Image is fully masked. Check for invalid values."):
            FitTrace(img_all_nans)

        window = 10
        guess = int(nrows / 2)
        img_win_nans = img.copy()
        img_win_nans[guess - window : guess + window] = np.nan

        # error on trace of otherwise valid image with all-nan window around guess
        with pytest.raises(ValueError, match="pixels in window region are masked"):
            FitTrace(img_win_nans, guess=guess, window=window)

    def test_fit_trace_fully_masked_columns_warn_msg(self):
        """
        Test that the correct warning message is raised when fully masked columns
        (in a not-fully-masked image) are encountered in FitTrace. These columns
        will be set to NaN and filtered from the final all-bin fit (as tested in
        test_fit_trace_fully_masked_cols), but a warning message is raised.
        """
        img = mk_img()

        # test that warning (dependent on choice of `peak_method`) is raised when a
        # few bins are masked, and that theyre listed individually
        mask = np.zeros(img.shape, dtype=bool)
        mask[:, 100] = 1
        mask[:, 20] = 1
        mask[:, 30] = 1
        nddat = NDData(data=img, mask=mask, unit=u.DN)

        match_str = "All pixels in bins 20, 30, 100 are fully masked. Setting bin peaks to NaN."

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method="max")

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method="centroid")

        with pytest.warns(UserWarning, match=match_str):
            FitTrace(nddat, peak_method="gaussian")

        # and when many bins are masked, that the message is consolidated
        mask = np.zeros(img.shape, dtype=bool)
        mask[:, 0:21] = 1
        nddat = NDData(data=img, mask=mask, unit=u.DN)
        with pytest.warns(
            UserWarning,
            match="All pixels in bins "
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 20 are "
            "fully masked. Setting bin peaks to NaN.",
        ):
            FitTrace(nddat)

    @pytest.mark.filterwarnings("ignore:All pixels in bins")
    @pytest.mark.parametrize("mask_treatment", ["apply", "propagate", "apply_nan_only"])
    def test_fit_trace_fully_masked_cols(self, mask_treatment):
        """
        Create a test image with some fully-nan/masked columns, and test that
        when the final fit to all bin peaks is done for the trace, that these
        fully-masked columns are set to NaN and filtered during the final all-bin
        fit. Ignore the warning that is produced when this case is encountered (that
        is tested in `test_fit_trace_fully_masked_cols_warn_msg`.)
        """
        img = mk_img(nrows=10, ncols=11)

        # set some columns fully to nan, which will be masked out
        img[:, 7] = np.nan
        img[:, 4] = np.nan
        img[:, 0] = np.nan

        # also create an image that doesn't have nans in the data, but
        # is masked in the same locations, to make sure that is equivilant.

        # test peak_method = 'max'
        truth = [
            1.6346154,
            2.2371795,
            2.8397436,
            3.4423077,
            4.0448718,
            4.6474359,
            5.25,
            5.8525641,
            6.4551282,
            7.0576923,
            7.6602564,
        ]
        max_trace = FitTrace(img, peak_method="max", mask_treatment=mask_treatment)
        np.testing.assert_allclose(truth, max_trace.trace, atol=0.1)

        # peak_method = 'gaussian'
        truth = [
            1.947455,
            2.383634,
            2.8198131,
            3.2559921,
            3.6921712,
            4.1283502,
            4.5645293,
            5.0007083,
            5.4368874,
            5.8730665,
            6.3092455,
        ]
        max_trace = FitTrace(img, peak_method="gaussian", mask_treatment=mask_treatment)
        np.testing.assert_allclose(truth, max_trace.trace, atol=0.1)

        # peak_method = 'centroid'
        truth = [
            2.5318835,
            2.782069,
            3.0322546,
            3.2824402,
            3.5326257,
            3.7828113,
            4.0329969,
            4.2831824,
            4.533368,
            4.7835536,
            5.0337391,
        ]
        max_trace = FitTrace(img, peak_method="centroid", mask_treatment=mask_treatment)
        np.testing.assert_allclose(truth, max_trace.trace, atol=0.1)

    @pytest.mark.filterwarnings("ignore:All pixels in bins")
    @pytest.mark.parametrize(
        "peak_method,expected",
        [
            ("max", [5.0, 3.0, 5.0, 5.0, 7.0, 5.0, np.nan, 5.0, 5.0, 5.0, 2.0, 5.0]),
            ("gaussian", [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, np.nan, 5.0, 5.0, 5.0, 1.576, 5.0]),
            (
                "centroid",
                [
                    4.27108332,
                    2.24060342,
                    4.27108332,
                    4.27108332,
                    6.66827608,
                    4.27108332,
                    np.nan,
                    4.27108332,
                    4.27108332,
                    4.27108332,
                    1.19673467,
                    4.27108332,
                ],
            ),
        ],
    )
    def test_mask_treatment_apply(self, peak_method, expected):
        """
        Test for mask_treatment=apply for FitTrace.
        With this masking option, masked and non-finite data should be filtered
        when determining bin/column peak. Fully masked bins should be omitted
        from the final all-bin-peak fit for the Trace. Parametrized over different
        `peak_method` options.
        """

        # Make an image with some non-finite values.
        image1 = mk_img(
            nan_slices=[np.s_[4:8, 1:2], np.s_[2:7, 4:5], np.s_[:, 6:7], np.s_[3:9, 10:11]],
            nrows=10,
            ncols=12,
            add_noise=False,
        )

        # Also make an image that doesn't have nonf data values, but has masked
        # values at the same locations, to make sure they give the same results.
        mask = ~np.isfinite(image1)
        dat = mk_img(nrows=10, ncols=12, add_noise=False)
        image2 = NDData(dat, mask=mask)

        for imgg in [image1, image2]:
            # run FitTrace, with the testing-only flag _save_bin_peaks_testing set
            # to True to return the bin peak values before fitting the trace
            trace = FitTrace(imgg, peak_method=peak_method, _save_bin_peaks_testing=True)
            x_bins, y_bins = trace._bin_peaks_testing
            np.testing.assert_allclose(y_bins, expected, atol=0.1)

            # check that final fit to all bins, accouting for fully-masked bins,
            # matches the trace
            fitter = fitting.LMLSQFitter()
            mask = np.isfinite(y_bins)
            all_bin_fit = fitter(trace.trace_model, x_bins[mask], y_bins[mask])
            all_bin_fit = all_bin_fit((np.arange(12)))

            np.testing.assert_allclose(trace.trace, all_bin_fit)

    @pytest.mark.filterwarnings("ignore:All pixels in bins")
    @pytest.mark.parametrize("peak_method", ["max", "gaussian", "centroid"])
    def test_mask_treatment_unsupported(self, peak_method):
        """
        Test to ensure the unsupported mask treatment methods for FitTrace
        raise a `ValueError`. Parametrized over different `peak_method` options.
        """

        image = mk_img(
            nan_slices=[np.s_[4:8, 1:2], np.s_[2:7, 4:5], np.s_[:, 6:7], np.s_[3:9, 10:11]],
            nrows=10,
            ncols=12,
            add_noise=False,
        )

        for method in "ignore", "zero_fill", "nan_fill", "apply_mask_only":
            with pytest.raises(ValueError):
                FitTrace(image, peak_method=peak_method, mask_treatment=method)

    @pytest.mark.filterwarnings("ignore:All pixels in bins")
    @pytest.mark.parametrize(
        "peak_method,expected",
        [
            ("max", [5.0, np.nan, 5.0, 5.0, np.nan, 5.0, np.nan, 5.0, 5.0, 5.0, np.nan, 5.0]),
            ("gaussian", [5.0, np.nan, 5.0, 5.0, np.nan, 5.0, np.nan, 5.0, 5.0, 5.0, np.nan, 5.0]),
            (
                "centroid",
                [
                    4.27108332,
                    np.nan,
                    4.27108332,
                    4.27108332,
                    np.nan,
                    4.27108332,
                    np.nan,
                    4.27108332,
                    4.27108332,
                    4.27108332,
                    np.nan,
                    4.27108332,
                ],
            ),
        ],
    )
    def test_mask_treatment_propagate(self, peak_method, expected):
        """
        Test for mask_treatment=`propagate` for FitTrace. Columns (assuming
        disp_axis==1) with any masked data values will be fully masked and
        therefore not contribute to the bin peaks. Parametrized over different
        `peak_method` options.
        """

        # Make an image with some non-finite values.
        image1 = mk_img(
            nan_slices=[np.s_[4:8, 1:2], np.s_[2:7, 4:5], np.s_[:, 6:7], np.s_[3:9, 10:11]],
            nrows=10,
            ncols=12,
            add_noise=False,
        )

        # Also make an image that doesn't have non-finite data values, but has masked
        # values at the same locations, to make sure those cases are equivalent
        mask = ~np.isfinite(image1)
        dat = mk_img(nrows=10, ncols=12, add_noise=False)
        image2 = NDData(dat, mask=mask)

        for imgg in [image1, image2]:

            # run FitTrace, with the testing-only flag _save_bin_peaks_testing set
            # to True to return the bin peak values before fitting the trace
            trace = FitTrace(
                imgg,
                peak_method=peak_method,
                mask_treatment="propagate",
                _save_bin_peaks_testing=True,
            )
            x_bins, y_bins = trace._bin_peaks_testing
            np.testing.assert_allclose(y_bins, expected)

            # check that final fit to all bins, accouting for fully-masked bins,
            # matches the trace
            fitter = fitting.LevMarLSQFitter()
            mask = np.isfinite(y_bins)
            all_bin_fit = fitter(trace.trace_model, x_bins[mask], y_bins[mask])
            all_bin_fit = all_bin_fit((np.arange(12)))

            np.testing.assert_allclose(trace.trace, all_bin_fit)
