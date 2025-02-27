import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import NDData, VarianceUncertainty, UnknownUncertainty
from astropy.tests.helper import assert_quantity_allclose
from specutils import Spectrum1D

from specreduce.background import Background
from specreduce.extract import (
    BoxcarExtract, HorneExtract, OptimalExtract, _align_along_trace
)
from specreduce.tracing import FitTrace, FlatTrace, ArrayTrace


def add_gaussian_source(image, amps=2, stddevs=2, means=None):

    """ Modify `image.data` to add a horizontal spectrum across the image.
        Each column can have a different amplitude, stddev or mean position
        if these are arrays (otherwise, constant across image).
    """

    nrows, ncols = image.shape

    if means is None:
        means = nrows // 2

    if not isinstance(means, np.ndarray):
        means = np.ones(ncols) * means
    if not isinstance(amps, np.ndarray):
        amps = np.ones(ncols) * amps
    if not isinstance(stddevs, np.ndarray):
        stddevs = np.ones(ncols) * stddevs

    for i, col in enumerate(range(ncols)):
        mod = models.Gaussian1D(amplitude=amps[i], mean=means[i],
                                stddev=stddevs[i])
        image.data[:, i] = mod(np.arange(nrows))


def test_boxcar_extraction(mk_test_img):
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #

    image = mk_test_img

    trace = FlatTrace(image, 15.0)
    boxcar = BoxcarExtract(image, trace)

    spectrum = boxcar.spectrum
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))
    assert spectrum.unit is not None and spectrum.unit == u.Jy

    trace.set_position(14.5)
    spectrum = boxcar()
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 72.5))

    trace.set_position(14.7)
    spectrum = boxcar()
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 73.5))

    trace.set_position(15.0)
    boxcar.width = 6
    spectrum = boxcar()
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 90.))

    trace.set_position(14.5)
    spectrum = boxcar(width=6)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 87.))

    trace.set_position(15.0)
    spectrum = boxcar(width=4.5)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.5))

    trace.set_position(15.0)
    spectrum = boxcar(width=4.7)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 70.5))

    trace.set_position(14.3)
    spectrum = boxcar(width=4.7)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.15))


def test_boxcar_nonfinite_handling(mk_test_img):
    image = mk_test_img
    image.data[14, 2] = np.nan
    image.data[14, 4] = np.inf

    trace = FlatTrace(image, 15.0)
    boxcar = BoxcarExtract(image, trace, width=6, mask_treatment='apply')
    spectrum = boxcar()
    target = np.full_like(spectrum.flux.value, 90.)
    target[2] = np.nan
    target[4] = np.inf
    np.testing.assert_equal(spectrum.flux.value, target)


def test_boxcar_outside_image_condition(mk_test_img):
    #
    # Trace is such that extraction aperture lays partially outside the image
    #
    image = mk_test_img

    trace = FlatTrace(image, 3.0)
    boxcar = BoxcarExtract(image, trace, mask_treatment='apply')

    spectrum = boxcar(width=10.)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 32.0))


def test_boxcar_array_trace(mk_test_img):
    image = mk_test_img

    trace_array = np.ones_like(image[1]) * 15.
    trace = ArrayTrace(image, trace_array)

    boxcar = BoxcarExtract(image, trace)
    spectrum = boxcar()
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))


def test_horne_image_validation(mk_test_img):
    #
    # Test HorneExtract scenarios specific to its use with an image of
    # type `~numpy.ndarray` (instead of the default `~astropy.nddata.NDData`).
    #
    image = mk_test_img

    trace = FlatTrace(image, 15.0)
    extract = OptimalExtract(image.data, trace)  # equivalent to HorneExtract

    # an array-type image must come with a variance argument
    with pytest.raises(ValueError, match=r'.*variance must be specified.*'):
        ext = extract()

    # an NDData-type image can't have an empty uncertainty attribute
    with pytest.raises(ValueError, match=r'.*NDData object lacks uncertainty'):
        ext = extract(image=image)

    # an NDData-type image's uncertainty must be of type VarianceUncertainty
    # or type StdDevUncertainty
    with pytest.raises(ValueError, match=r'.*unexpected uncertainty type.*'):
        err = UnknownUncertainty(np.ones_like(image))
        image.uncertainty = err
        ext = extract(image=image)

    # an array-type image must have the same dimensions as its variance argument
    with pytest.raises(ValueError, match=r'.*shapes must match.*'):
        # remember variance, mask, and unit args are only checked if image
        # object doesn't have those attributes (e.g., numpy and Quantity arrays)
        err = np.ones_like(image[0])
        ext = extract(image=image.data, variance=err)

    # an array-type image must have the same dimensions as its mask argument
    with pytest.raises(ValueError, match=r'.*shapes must match.*'):
        err = np.ones_like(image)
        mask = np.zeros_like(image[0])
        ext = extract(image=image.data, variance=err, mask=mask)

    # an array-type image given without mask and unit arguments is fine
    # and produces an extraction with flux in DN and spectral axis in pixels
    err = np.ones_like(image)
    ext = extract(image=image.data, variance=err, mask=None, unit=None)
    assert ext.unit == u.Unit('DN')
    assert np.all(ext.spectral_axis
                  == np.arange(image.shape[extract.disp_axis]) * u.pix)


# ignore Astropy warning for extractions that aren't best fit with a Gaussian:
@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_horne_variance_errors(mk_test_img):
    image = mk_test_img

    trace = FlatTrace(image, 3.0)

    # all zeros are treated as non-weighted (i.e., as non-zero fluxes)
    image.uncertainty = VarianceUncertainty(np.zeros_like(image))
    image.mask = np.zeros_like(image)
    extract = HorneExtract(image, trace)
    ext = extract.spectrum
    assert not np.all(ext == 0)

    # single zero value adjusts mask and does not raise error
    err = np.ones_like(image)
    err[0][0] = 0
    image.uncertainty = VarianceUncertainty(err)
    ext = extract(image)
    assert not np.all(ext == 1)

    # single negative value raises error
    err = image.uncertainty.array
    err[0][0] = -1
    with pytest.raises(ValueError, match='variance must be fully positive'):
        # remember variance, mask, and unit args are only checked if image
        # object doesn't have those attributes (e.g., numpy and Quantity arrays)
        ext = extract(image=image.data, variance=err,
                      mask=image.mask, unit=u.Jy)


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_horne_non_flat_trace():
    # create a synthetic "2D spectrum" and its non-flat trace

    n_rows, n_cols = (10, 50)
    original = np.zeros((n_rows, n_cols))
    original[n_rows // 2] = 1

    # create small offsets along each column to specify a non-flat trace
    trace_offset = np.polyval([2e-3, -0.01, 0], np.arange(n_cols)).astype(int)
    exact_trace = n_rows // 2 - trace_offset

    # re-index the array with the offsets applied to the trace (make it non-flat):
    rows = np.broadcast_to(np.arange(n_rows)[:, None], original.shape)
    cols = np.broadcast_to(np.arange(n_cols), original.shape)
    roll_rows = np.mod(rows + trace_offset[None, :], n_rows)
    rolled = original[roll_rows, cols]

    # all zeros are treated as non-weighted (give non-zero fluxes)
    err = 0.1 * np.ones_like(rolled)
    mask = np.zeros_like(rolled).astype(bool)

    # unroll the trace using the Horne extract utility function for alignment:
    unrolled = _align_along_trace(rolled, n_rows // 2 - trace_offset)

    # ensure that mask is correctly unrolled back to its original alignment:
    np.testing.assert_allclose(unrolled, original)

    # Extract the spectrum from the non-flat image+trace
    extract_non_flat = HorneExtract(
        rolled, ArrayTrace(rolled, exact_trace),
        variance=err, mask=mask, unit=u.Jy
    )()

    # Also extract the spectrum from the image after alignment with a flat trace
    extract_flat = HorneExtract(
        unrolled, FlatTrace(unrolled, n_rows // 2),
        variance=err, mask=mask, unit=u.Jy
    )()

    # ensure both extractions are equivalent:
    assert_quantity_allclose(extract_non_flat.flux, extract_flat.flux)


def test_horne_bad_profile(mk_test_img):
    image = mk_test_img
    trace = FlatTrace(image, 3.0)
    extract = HorneExtract(image.data, trace,
                           spatial_profile='bad_profile_name',
                           variance=np.ones(image.data.shape))
    with pytest.raises(ValueError, match='spatial_profile must be one of'):
        extract.spectrum


def test_horne_nonfinite_column(mk_test_img):
    image = mk_test_img
    image.data[:, 4] = np.nan
    trace = FlatTrace(image, 3.0)
    extract = HorneExtract(image.data, trace,
                           spatial_profile='gaussian',
                           variance=np.ones(image.data.shape))
    sp = extract.spectrum
    assert np.isnan(sp.flux.value[4])
    assert np.all(np.isfinite(sp.flux.value[:4]))
    assert np.all(np.isfinite(sp.flux.value[5:]))


def test_horne_no_bkgrnd(mk_test_img):
    # Test HorneExtract when using bkgrd_prof=None

    image = mk_test_img

    trace = FlatTrace(image, 3.0)
    extract = HorneExtract(image.data, trace, bkgrd_prof=None,
                           variance=np.ones(image.data.shape))

    # This is just testing that it runs with no errors and returns something
    assert len(extract.spectrum.flux) == 10


def test_horne_interpolated_profile(mk_test_img):
    # basic test for HorneExtract using the spatial_profile == `interpolated_profile`
    # option add a perfectly gaussian source and make sure gaussian extraction
    # and self-profile extraction agree (since self=gaussian in this case)

    image = mk_test_img
    add_gaussian_source(image)  # add source across image, flat trace

    trace = FlatTrace(image, image.shape[0] // 2)

    # horne extraction using spatial_profile=='Gaussian'
    horne_extract_gauss = HorneExtract(image.data, trace,
                                       spatial_profile='gaussian',
                                       bkgrd_prof=None,
                                       variance=np.ones(image.data.shape))

    # horne extraction with spatial_profile=='interpolated_profile'
    horne_extract_self = HorneExtract(image.data, trace,
                                      spatial_profile={'name': 'interpolated_profile',
                                                       'n_bins_interpolated_profile': 3},
                                      bkgrd_prof=None,
                                      variance=np.ones(image.data.shape))

    assert_quantity_allclose(horne_extract_gauss.spectrum.flux,
                             horne_extract_self.spectrum.flux)


def test_horne_interpolated_profile_norm(mk_test_img):
    # ensure that when using spatial_profile == `interpolated_profile`, the fit profile
    # represents the shape as a funciton of wavelength, and correctly accounts
    # for variations in trace position and flux.

    image = mk_test_img
    nrows, ncols = image.shape

    # create sawtooth pattern trace. right now, the _align_along_trace function
    # will rectify the trace to the integer-pixel level, so in this test
    # case the specturm will be totally straightened out.
    trace_shape = np.ones(ncols) * nrows // 2
    trace_shape[::2] += 4
    trace = ArrayTrace(image, trace_shape)

    # add gaussian source to image with increasing amplitude and that follows
    # along the trace. looks like a sawtooth pattern of increasing brightness
    add_gaussian_source(image, amps=np.linspace(1, 5, image.shape[1]),
                        means=trace_shape)

    # horne extraction with spatial_profile=='interpolated_profile'
    # also tests that non-default parameters in the input dictionary format work
    ex = HorneExtract(image.data, trace,
                      spatial_profile={'name': 'interpolated_profile',
                                       'n_bins_interpolated_profile': 3,
                                       'interp_degree_interpolated_profile': (1, 1)},
                      bkgrd_prof=None,
                      variance=np.ones(image.data.shape))

    # need to run produce extract.spectrum to access _interp_spatial_prof
    ex.spectrum

    # evaulate interpolated profile on entire grid
    interp_prof = ex._interp_spatial_prof(np.arange(ncols),
                                          np.arange(nrows)).T

    # the shifting position and amplitude should be accounted for, so the fit
    # spatial profile should just represent the shape as a function of
    # wavelength in this case, that is a gaussian with the area normalized at
    # each wavelength and a constant mean position

    # make sure that the fit spatial prof is normalized correctly
    assert_quantity_allclose(np.sum(interp_prof, axis=0), 1.0)

    # and that shifts in trace position are accounted for (to integer level)
    assert (np.all(np.argmax(interp_prof, axis=0) == nrows // 2))


def test_horne_interpolated_nbins_fails(mk_test_img):
    # make sure that HorneProfile spatial_profile='interpolated_profile' correctly
    # fails when the number of samples is greater than the size of the image
    image = mk_test_img
    trace = FlatTrace(image, 5)

    with pytest.raises(ValueError):
        ex = HorneExtract(image.data, trace,
                          spatial_profile={'name': 'interpolated_profile',
                                           'n_bins_interpolated_profile': 100})
        ex.spectrum


class TestMasksExtract():

    def mk_flat_gauss_img(self, nrows=200, ncols=160, nan_slices=None, add_noise=True):

        """
        Makes a flat gaussian image for testing, with optional added gaussian
        nosie and optional data values set to NaN. Variance is included, which
        is required by HorneExtract. Returns a Spectrum1D with flux, spectral
        axis, and uncertainty.
        """

        sigma_pix = 4
        col_model = models.Gaussian1D(amplitude=1, mean=nrows/2,
                                      stddev=sigma_pix)
        spec2dvar = np.ones((nrows, ncols))
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

        wave = np.arange(0, img.shape[1], 1)
        objectspec = Spectrum1D(spectral_axis=wave*u.m, flux=img*u.Jy,
                                uncertainty=VarianceUncertainty(spec2dvar*u.Jy*u.Jy))

        return objectspec

    def test_boxcar_fully_masked(self):
        """
        Test that the appropriate error is raised by `BoxcarExtract` when image
        is fully masked/NaN.
        """
        return

        img = self.mk_flat_gauss_img()
        trace = FitTrace(img)

        with pytest.raises(ValueError, match='Image is fully masked.'):
            # fully NaN image
            img = np.zeros((4, 5)) * np.nan
            Background(img, traces=trace, width=2)

        with pytest.raises(ValueError, match='Image is fully masked.'):
            # fully masked image (should be equivalent)
            img = NDData(np.ones((4, 5)), mask=np.ones((4, 5)))
            Background(img, traces=trace, width=2)

        # Now test that an image that isn't fully masked, but is fully masked
        # within the window determined by `width`, produces the correct result
        msg = 'Image is fully masked within background window determined by `width`.'
        with pytest.raises(ValueError, match=msg):
            img = self.mk_img(nrows=12, ncols=12, nan_slices=[np.s_[3:10, :]])
            Background(img, traces=FlatTrace(img, 6), width=7)
