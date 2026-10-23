import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import fitting, models
from astropy.nddata import NDData

from specutils import Spectrum
from specreduce.tracing import FitTrace
from specreduce.utils.utils import measure_cross_dispersion_profile, measure_noise


def mk_gaussian_img(nrows=20, ncols=16, mean=10, stddev=4):
    """ Makes a simple horizontal gaussian image."""

    # note: this should become a fixture eventually, since other tests use
    # similar functions to generate test data.

    np.random.seed(7)
    col_model = models.Gaussian1D(amplitude=1, mean=mean, stddev=stddev)
    index_arr = np.tile(np.arange(nrows), (ncols, 1))

    return col_model(index_arr.T)


def mk_img_non_flat_trace(nrows=40, ncols=100, amp=10, stddev=2):
    """
    Makes an image with a gaussian source that has a non-flat trace dispersed
    along the x axis.
    """
    spec2d = np.zeros((nrows, ncols))

    for ii in range(spec2d.shape[1]):
        mgaus = models.Gaussian1D(amplitude=amp,
                                  mean=(9.+(20/spec2d.shape[1])*ii),
                                  stddev=stddev)
        rg = np.arange(0, spec2d.shape[0], 1)
        gaus = mgaus(rg)
        spec2d[:, ii] = gaus

    return spec2d


class TestMeasureCrossDispersionProfile():

    @pytest.mark.parametrize('pixel', [None, 1, [1, 2, 3]])
    @pytest.mark.parametrize('width', [10, 9])
    def test_measure_cross_dispersion_profile(self, pixel, width):
        """
        Basic test for `measure_cross_dispersion_profile`. Parametrized over
        different options for `pixel` to test using all wavelengths, a single
        wavelength, and a set of wavelengths, as well as different input types
        (plain array, quantity, Spectrum, and NDData), as well as `width` to
        use a window of all rows and a smaller window.
        """

        # test a few input formats
        images = []
        mean = 5.0
        stddev = 4.0
        dat = mk_gaussian_img(nrows=10, ncols=10, mean=mean, stddev=stddev)
        images.append(dat)  # test unitless
        images.append(dat * u.DN)
        images.append(NDData(dat * u.DN))
        images.append(Spectrum(flux=dat * u.DN, spectral_axis_index=dat.ndim - 1))

        for img in images:

            # use a flat trace at trace_pos=10, a window of width 10 around the trace
            # and use all 20 columns in image to create an average (median)
            # cross dispersion profile
            cdp = measure_cross_dispersion_profile(img, width=width, pixel=pixel)

            # make sure that if we fit a gaussian to the measured average profile,
            # that we get out the same profile that was used to create the image.
            # this should be exact since theres no noise in the image
            fitter = fitting.LevMarLSQFitter()
            mod = models.Gaussian1D()
            fit_model = fitter(mod, np.arange(width), cdp)

            assert fit_model.mean.value == np.where(cdp == max(cdp))[0][0]
            assert fit_model.stddev.value == stddev

            # test passing in a FlatTrace, and check the profile
            cdp = measure_cross_dispersion_profile(img, width=width, pixel=pixel)
            fit_model = fitter(mod, np.arange(width), cdp)
            assert fit_model.mean.value == np.where(cdp == max(cdp))[0][0]
            np.testing.assert_allclose(fit_model.stddev.value, stddev)

    @pytest.mark.filterwarnings("ignore:Model is linear in parameters")
    def test_cross_dispersion_profile_non_flat_trace(self):
        """
        Test measure_cross_dispersion_profile with a non-flat trace.
        Tests with 'align_along_trace' set to both True and False,
        to account for the changing center of the trace and measure
        the true profile shape, or to 'blur' the profile, respectivley.
        """

        image = mk_img_non_flat_trace()

        # fit the trace
        trace_fit = FitTrace(image)

        # when not aligning along trace and using the entire image
        # rows for the window, the center of the profile should follow
        # the shape of the trace
        peak_locs = [9, 10, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 26, 27, 29]
        for i, pixel in enumerate(range(0, image.shape[1], 7)):
            profile = measure_cross_dispersion_profile(image,
                                                       trace=trace_fit,
                                                       width=None,
                                                       pixel=pixel,
                                                       align_along_trace=False,
                                                       statistic='average')
            peak_loc = (np.where(profile == max(profile))[0][0])
            assert peak_loc == peak_locs[i]

        # when align_along_trace = True, the shape of the profile should
        # not change since (there is some wiggling around though due to the
        # fact that the trace is rolled to the nearest integer value. this can
        # be smoothed with an interpolation option later on, but it is 'rough'
        # for now). In this test case, the peak positions will all either
        # be at pixel 20 or 21.
        for i, pixel in enumerate(range(0, image.shape[1], 7)):
            profile = measure_cross_dispersion_profile(image,
                                                       trace=trace_fit,
                                                       width=None,
                                                       pixel=pixel,
                                                       align_along_trace=True,
                                                       statistic='average')
            peak_loc = (np.where(profile == max(profile))[0][0])
            assert peak_loc in [20, 21]

    def test_errors_warnings(self):
        img = mk_gaussian_img(nrows=10, ncols=10)
        with pytest.raises(ValueError,
                           match='`crossdisp_axis` must be 0 or 1'):
            measure_cross_dispersion_profile(img, crossdisp_axis=2)

        with pytest.raises(ValueError, match='`trace` must be Trace object, '
                                             'number to specify the location '
                                             'of a FlatTrace, or None to use '
                                             'center of image.'):
            measure_cross_dispersion_profile(img, trace='not a trace or a number')

        with pytest.raises(ValueError, match="`statistic` must be 'median' "
                                             "or 'average'."):
            measure_cross_dispersion_profile(img, statistic='n/a')

        with pytest.raises(ValueError, match='Both `pixel` and `pixel_range` '
                                             'can not be set simultaneously.'):
            measure_cross_dispersion_profile(img, pixel=2, pixel_range=(2, 3))

        with pytest.raises(ValueError, match='`pixels` must be an integer, '
                                             'or list of integers to specify '
                                             'where the crossdisperion profile '
                                             'should be measured.'):
            measure_cross_dispersion_profile(img, pixel='str')

        with pytest.raises(ValueError, match='`pixel_range` must be a tuple '
                                             'of integers.'):
            measure_cross_dispersion_profile(img, pixel_range=(2, 3, 5))

        with pytest.raises(ValueError, match='Pixels chosen to measure cross '
                                             'dispersion profile are out of '
                                             'image bounds.'):
            measure_cross_dispersion_profile(img, pixel_range=(2, 12))

        with pytest.raises(ValueError, match='`width` must be an integer, '
                                             'or None to use all '
                                             'cross-dispersion pixels.'):
            measure_cross_dispersion_profile(img, width='.')


class TestMeasureNoise:
    """Tests for the robust second-difference noise estimator."""

    @staticmethod
    def _arc(rng, x, centers, amplitudes, fwhm=4.0):
        sigma = fwhm / 2.3548
        flux = np.zeros_like(x)
        for c, a in zip(centers, amplitudes):
            flux += a * np.exp(-0.5 * ((x - c) / sigma) ** 2)
        return flux

    def test_white_noise(self):
        rng = np.random.default_rng(1)
        flux = rng.normal(0.0, 3.0, 2000)
        assert measure_noise(flux) == pytest.approx(3.0, rel=0.05)

    def test_lines_and_residual_background(self):
        rng = np.random.default_rng(2)
        x = np.arange(2000.0)
        flux = self._arc(rng, x, rng.uniform(0, 2000, 40), 10 ** rng.uniform(1, 3.3, 40))
        flux += 120.0 * (x / 2000.0 - 0.5) ** 2
        flux += rng.normal(0.0, 3.0, x.size)
        noise = measure_noise(flux)
        assert 0.9 * 3.0 < noise < 1.3 * 3.0

    def test_2d_per_row_along_dispersion_axis(self):
        rng = np.random.default_rng(3)
        sigmas = np.array([1.0, 2.0, 4.0, 8.0])
        flux = rng.normal(0.0, sigmas[:, None], (4, 1000))
        noise = measure_noise(flux, axis=-1)
        assert noise.shape == (4,)
        np.testing.assert_allclose(noise, sigmas, rtol=0.1)
        noise_t = measure_noise(flux.T, axis=0)
        np.testing.assert_allclose(noise_t, noise)

    def test_quantity_input_keeps_unit(self):
        rng = np.random.default_rng(4)
        flux = rng.normal(0.0, 3.0, 2000) * u.DN
        noise = measure_noise(flux)
        assert isinstance(noise, u.Quantity)
        assert noise.unit == u.DN
        assert noise.value == pytest.approx(3.0, rel=0.05)

    def test_nan_pixels_are_ignored(self):
        rng = np.random.default_rng(5)
        flux = rng.normal(0.0, 3.0, 2000)
        flux[100:110] = np.nan
        assert measure_noise(flux) == pytest.approx(3.0, rel=0.05)

    def test_too_short_input_raises(self):
        with pytest.raises(ValueError, match="at least 5"):
            measure_noise(np.zeros(4))
