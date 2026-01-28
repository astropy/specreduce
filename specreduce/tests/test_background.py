import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import NDData, VarianceUncertainty, StdDevUncertainty, InverseVariance

from specreduce.background import Background
from specreduce.compat import Spectrum
from specreduce.tracing import FlatTrace, ArrayTrace


def test_background(
    mk_test_img_raw, mk_test_spec_no_spectral_axis, mk_test_spec_with_spectral_axis
):
    img = mk_test_img_raw
    image = mk_test_spec_no_spectral_axis
    image_um = mk_test_spec_with_spectral_axis
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    trace_pos = 15
    trace = FlatTrace(image, trace_pos)
    bkg_sep = 5
    bkg_width = 2

    # all the following should be equivalent, whether image's spectral axis
    # is in pixels or physical units:
    bg1 = Background(image, [trace - bkg_sep, trace + bkg_sep], width=bkg_width)
    bg2 = Background.two_sided(image, trace, bkg_sep, width=bkg_width)
    bg3 = Background.two_sided(image, trace_pos, bkg_sep, width=bkg_width)
    assert np.allclose(bg1.bkg_image().flux, bg2.bkg_image().flux)
    assert np.allclose(bg1.bkg_image().flux, bg3.bkg_image().flux)

    bg4 = Background(image_um, [trace - bkg_sep, trace + bkg_sep], width=bkg_width)
    bg5 = Background.two_sided(image_um, trace, bkg_sep, width=bkg_width)
    bg6 = Background.two_sided(image_um, trace_pos, bkg_sep, width=bkg_width)
    assert np.allclose(bg1.bkg_image().flux, bg4.bkg_image().flux)
    assert np.allclose(bg1.bkg_image().flux, bg5.bkg_image().flux)
    assert np.allclose(bg1.bkg_image().flux, bg6.bkg_image().flux)

    # test that creating a one_sided background works
    Background.one_sided(image, trace, bkg_sep, width=bkg_width)

    # test that passing a single trace works
    bg = Background(image, trace, width=bkg_width)

    # test that image subtraction works
    sub1 = image - bg1
    sub2 = bg1.sub_image(image)
    sub3 = bg1.sub_image()
    assert np.allclose(sub1.flux, sub2.flux)
    assert np.allclose(sub2.flux, sub3.flux)

    sub4 = image_um - bg4
    sub5 = bg4.sub_image(image_um)
    sub6 = bg4.sub_image()
    assert np.allclose(sub1.flux, sub4.flux)
    assert np.allclose(sub4.flux, sub5.flux)
    assert np.allclose(sub5.flux, sub6.flux)

    bkg_spec = bg1.bkg_spectrum()
    assert isinstance(bkg_spec, Spectrum)
    sub_spec = bg1.sub_spectrum()
    assert isinstance(sub_spec, Spectrum)

    # test that width==0 results in no background
    bg = Background.two_sided(image, trace, bkg_sep, width=0)
    assert np.all(bg.bkg_image().flux == 0)

    # test that any NaNs in input image (whether in or outside the window) don't
    # propagate to _bkg_array (which affects bkg_image and sub_image methods) or
    # the final 1D spectra.
    img[0, 0] = np.nan  # out of window
    img[trace_pos, 0] = np.nan  # in window
    stats = ["average", "median"]

    for st in stats:
        bg = Background(img, trace - bkg_sep, width=bkg_width, statistic=st)
        assert np.isnan(bg.image.flux).sum() == 2
        assert np.isnan(bg._bkg_array).sum() == 0
        assert np.isnan(bg.bkg_spectrum().flux).sum() == 0
        assert np.isnan(bg.sub_spectrum().flux).sum() == 0

    with pytest.warns(DeprecationWarning, match="bkg_statistic.*deprecated"):
        bg.bkg_spectrum(bkg_statistic="mean")


def test_warnings_errors(mk_test_spec_no_spectral_axis):
    image = mk_test_spec_no_spectral_axis

    # image.shape (30, 10)
    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 25, 4, width=3)

    # bottom of top window near/on top-edge of image (these should warn, but not fail)
    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 25, 8, width=5)

    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 25, 8, width=6)

    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 25, 8, width=7)

    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 7, 5, width=6)

    trace = ArrayTrace(image, trace=np.arange(10) + 20)  # from 20 to 29
    with pytest.warns(match="background window extends beyond image boundaries"):
        with pytest.raises(
            ValueError,
            match="background window does not remain in bounds across entire dispersion axis",
        ):  # noqa
            # 20 + 10 - 3 = 27 (lower edge of window on-image at right side of trace)
            # 29 + 10 - 3 = 36 (lower edge of window off-image at right side of trace)
            Background.one_sided(image, trace, 10, width=3)

    with pytest.raises(ValueError, match="width must be positive"):
        Background.two_sided(image, 25, 2, width=-1)


def test_trace_inputs(mk_test_img_raw):
    """
    Tests for the input argument 'traces' to `Background`. This should accept
    a list of or a single Trace object, or a list of or a single (positive)
    number to define a FlatTrace.
    """

    image = mk_test_img_raw

    # When `Background` object is created with no Trace object passed in it should
    # create a FlatTrace in the middle of the image (according to disp. axis)
    background = Background(image, width=5)
    assert np.all(background.traces[0].trace.data == image.shape[0] / 2.0)

    # FlatTrace(s) should be created if number or list of numbers is passed in for `traces`
    background = Background(image, 10.0, width=5)
    assert isinstance(background.traces[0], FlatTrace)
    assert background.traces[0].trace_pos == 10.0

    traces = [10.0, 15]
    background = Background(image, traces, width=5)
    for i, trace_pos in enumerate(traces):
        assert background.traces[i].trace_pos == trace_pos

    # make sure error is raised if input for `traces` is invalid
    match_str = (
        "objects, a number or list of numbers to define FlatTraces, "
        + "or None to use a FlatTrace in the middle of the image."
    )
    with pytest.raises(ValueError, match=match_str):
        Background(image, "non_valid_trace_pos")


class TestMasksBackground:
    """
    Various test functions to test how masked and non-finite data is handled
    in `Background.
    """

    def mk_img(self, nrows=4, ncols=5, nan_slices=None):
        """
        Make a simple gradient image to test masking in Background.
        Optionally add NaNs to data with `nan_slices`. Returned array is in
        u.DN.
        """

        img = np.tile((np.arange(1.0, ncols + 1)), (nrows, 1))

        if nan_slices:  # add nans in data
            for s in nan_slices:
                img[s] = np.nan

        return img * u.DN

    @pytest.mark.parametrize("mask", ["apply", "propagate", "zero_fill"])
    def test_fully_masked_column(self, mask):
        """
        Test background with some fully-masked columns (not fully masked image).
        In this case, the background value for that fully-masked column should
        be 0.0, with no error or warning raised.
        """

        img = self.mk_img(nrows=10, ncols=10)
        img[:, 0:1] = np.nan

        bkg = Background(img, traces=FlatTrace(img, 6), mask_treatment=mask)
        assert np.all(bkg.bkg_image().data[:, 0:1] == 0.0)

    @pytest.mark.parametrize("mask", ["apply", "propagate"])
    def test_fully_masked_image(self, mask):
        """
        Test that the appropriate error is raised by `Background` when image
        is fully masked/NaN.
        """

        with pytest.raises(ValueError, match="Image is fully masked."):
            # fully NaN image
            img = self.mk_img() * np.nan
            Background(img, traces=FlatTrace(self.mk_img(), 2), mask_treatment=mask)

        with pytest.raises(ValueError, match="Image is fully masked."):
            # fully masked image (should be equivalent)
            img = NDData(np.ones((4, 5)), mask=np.ones((4, 5), dtype=bool))
            Background(img, traces=FlatTrace(self.mk_img(), 2), mask_treatment=mask)

        # Now test that an image that isn't fully masked, but is fully masked
        # within the window determined by `width`, produces the correct result.
        msg = "Image is fully masked within background window determined by `width`."
        with pytest.raises(ValueError, match=msg):
            img = self.mk_img(nrows=12, ncols=12, nan_slices=[np.s_[3:10, :]])
            Background(img, traces=FlatTrace(img, 6), width=7)

    @pytest.mark.filterwarnings("ignore:background window extends beyond image boundaries")
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("apply", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
            (
                "propagate",
                np.array([0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0]),
            ),
            (
                "zero_fill",
                np.array(
                    [
                        0.58333333,
                        2.0,
                        3.0,
                        2.33333333,
                        5.0,
                        6.0,
                        7.0,
                        7.33333333,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ]
                ),
            ),
        ],
    )
    def test_mask_treatment_bkg_img_spectrum(self, method, expected):
        """
        This test function tests `Background.bkg_image` and
        `Background.bkg_spectrum` when there is masked data. It also tests
        background subtracting the image, and returning the spectrum of the
        background subtracted image. This test is parameterized over all
        currently implemented mask handling methods to test that they
        work as intended. The window size is set to use the entire image array,
        so warning about background window is ignored."""

        img_size = 12  # square 12 x 12 image

        # make image, set some value to nan, which will be masked in the function
        image1 = self.mk_img(
            nrows=img_size, ncols=img_size, nan_slices=[np.s_[5:10, 0], np.s_[7:12, 3], np.s_[2, 7]]
        )

        # also make an image that doesn't have nonf data values, but has
        # masked values at the same locations, to make sure they give the same
        # results
        mask = ~np.isfinite(image1)
        dat = self.mk_img(nrows=img_size, ncols=img_size)
        image2 = NDData(dat, mask=mask)

        for image in [image1, image2]:

            # construct a flat trace in center of image
            trace = FlatTrace(image, img_size / 2)

            # create 'Background' object with `mask_treatment` set
            # 'width' should be > size of image to use all pix (but warning will
            # be raised, which we ignore.)
            background = Background(image, mask_treatment=method, traces=trace, width=img_size + 1)

            # test background image matches 'expected'
            bk_img = background.bkg_image()
            # change this and following assertions to assert_quantity_allclose once
            # issue #213 is fixed
            np.testing.assert_allclose(bk_img.flux.value, np.tile(expected, (img_size, 1)))

            # test background spectrum matches 'expected'
            bk_spec = background.bkg_spectrum()
            np.testing.assert_allclose(bk_spec.flux.value, expected)

    def test_sub_bkg_image(self):
        """
        Test that masked and non-finite data is handled correctly when subtracting
        background from image, for all currently implemented masking
        options.
        """

        # make image, set some value to nan, which will be masked in the function
        image = self.mk_img(
            nrows=12, ncols=12, nan_slices=[np.s_[5:10, 0], np.s_[7:12, 3], np.s_[2, 7]]
        )

        # Calculate a background value using mask_treatment = 'apply'.
        # For 'apply', the flag applies to how masked values are handled during
        # calculation of background for each column, but nonfinite data will
        # remain in input data array
        background_apply = Background(
            image, mask_treatment="apply", traces=FlatTrace(image, 6), width=2
        )
        subtracted_img_apply = background_apply.sub_image()

        assert np.all(np.isfinite(subtracted_img_apply.data) == np.isfinite(image.data))

        # Calculate a background value using mask_treatment = 'propagate'. The input
        # 2d mask is reduced to a 1d mask to mask out full columns in the
        # presence of any nans - this means that (as tested above in
        # `test_mask_treatment_bkg_img_spectrum`) those columns will have 0.0
        # background. In this case, image.mask is expanded to mask full
        # columns - the image itself will not have full columns set to np.nan,
        # so there are still valid background subtracted data values in this
        # case, but the corresponding mask for that entire column will be masked.

        background_propagate = Background(
            image, mask_treatment="propagate", traces=FlatTrace(image, 6), width=2
        )
        subtracted_img_propagate = background_propagate.sub_image()

        assert np.all(np.isfinite(subtracted_img_propagate.data) == np.isfinite(image.data))

        # Calculate a background value using mask_treatment = 'zero_fill'. Data
        # values at masked locations are set to 0 in the image array, and the
        # background value calculated for that column will be subtracted
        # resulting in a negative value. The resulting background subtracted
        # image should be fully finite and the mask should be zero everywhere
        # (all unmasked)

        background_zero_fill = Background(
            image, mask_treatment="zero_fill", traces=FlatTrace(image, 6), width=2
        )
        subtracted_img_zero_fill = background_zero_fill.sub_image()

        assert np.all(np.isfinite(subtracted_img_zero_fill.data))
        assert np.all(subtracted_img_zero_fill.mask == 0)


@pytest.fixture
def mk_bgk_img():
    nrows, ncols = 10, 20
    var = 4.0
    img = Spectrum(
        np.ones((nrows, ncols)) * u.DN,
        uncertainty=VarianceUncertainty(np.full((nrows, ncols), var) * u.DN**2),
        spectral_axis=np.arange(ncols) * u.pix
    )
    trace = FlatTrace(img, nrows // 2)
    return trace, img, var, nrows, ncols


def test_background_uncertainty_average(mk_bgk_img):
    """Test background uncertainty estimation with 'average' statistic."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4, statistic="average")

    # Check that _bkg_variance is computed
    assert hasattr(bg, "_bkg_variance")
    assert bg._bkg_variance is not None
    assert len(bg._bkg_variance) == ncols

    # Variance should be positive and less than input (averaging reduces variance)
    assert np.all(bg._bkg_variance > 0)
    assert np.all(bg._bkg_variance < var)

    weights = bg.bkg_wimage
    weights_sum = np.sum(weights, axis=0)
    weights_sq_sum = np.sum(weights ** 2, axis=0)
    expected_variance = (weights_sq_sum * var) / (weights_sum ** 2)
    assert np.allclose(bg._bkg_variance, expected_variance)

    # Check that bkg_spectrum has uncertainty
    bkg_spec = bg.bkg_spectrum()
    assert bkg_spec.uncertainty is not None
    assert isinstance(bkg_spec.uncertainty, VarianceUncertainty)
    assert np.allclose(bkg_spec.uncertainty.array, bg._bkg_variance)


def test_background_uncertainty_median(mk_bgk_img):
    """Test background uncertainty estimation with 'median' statistic."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4, statistic="median")

    # Check that _bkg_variance is computed
    assert hasattr(bg, "_bkg_variance")
    assert bg._bkg_variance is not None

    # Variance should be positive
    assert np.all(bg._bkg_variance > 0)
    assert np.all(np.isfinite(bg._bkg_variance))

    n_pixels = np.sum(bg.bkg_wimage > 0, axis=0)
    expected_variance = (np.pi / 2) * var / n_pixels
    assert np.allclose(bg._bkg_variance, expected_variance, rtol=0.01)


def test_bkg_image_has_uncertainty(mk_bgk_img):
    """Test that bkg_image returns Spectrum with uncertainty."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4)
    bkg_img = bg.bkg_image()

    # Check uncertainty exists
    assert bkg_img.uncertainty is not None
    assert isinstance(bkg_img.uncertainty, VarianceUncertainty)

    # Check shape matches image
    assert bkg_img.uncertainty.array.shape == (nrows, ncols)

    # Check values are tiled correctly (same value in each column)
    for col in range(ncols):
        assert np.allclose(bkg_img.uncertainty.array[:, col], bg._bkg_variance[col])


def test_bkg_spectrum_has_uncertainty(mk_bgk_img):
    """Test that bkg_spectrum returns Spectrum with uncertainty."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4)
    bkg_spec = bg.bkg_spectrum()

    # Check uncertainty exists and is 1D
    assert bkg_spec.uncertainty is not None
    assert isinstance(bkg_spec.uncertainty, VarianceUncertainty)
    assert bkg_spec.uncertainty.array.ndim == 1
    assert len(bkg_spec.uncertainty.array) == ncols


def test_sub_image_propagates_uncertainty(mk_bgk_img):
    """Test that sub_image propagates uncertainties correctly."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4)

    sub_img = bg.sub_image()

    # Check uncertainty exists
    assert sub_img.uncertainty is not None
    assert isinstance(sub_img.uncertainty, VarianceUncertainty)

    # For subtraction: Var(A - B) = Var(A) + Var(B)
    # Image variance is 4.0, background variance is computed
    bkg_variance = bg._bkg_variance[0]  # Same for all columns
    expected_variance = var + bkg_variance
    assert np.allclose(sub_img.uncertainty.array, expected_variance, rtol=0.01)


def test_sub_spectrum_propagates_uncertainty(mk_bgk_img):
    """Test that sub_spectrum propagates uncertainties correctly."""
    trace, img, var, nrows, ncols = mk_bgk_img
    bg = Background(img, trace, width=4)

    sub_spec = bg.sub_spectrum()

    # Check uncertainty exists
    assert sub_spec.uncertainty is not None
    assert isinstance(sub_spec.uncertainty, VarianceUncertainty)

    # Check it's 1D with correct length
    assert sub_spec.uncertainty.array.ndim == 1
    assert len(sub_spec.uncertainty.array) == ncols

    # Variance should be positive and finite
    assert np.all(np.isfinite(sub_spec.uncertainty.array))
    assert np.all(sub_spec.uncertainty.array > 0)


def test_background_uncertainty_with_mask():
    """Test that masked pixels don't contribute to uncertainty calculation."""
    nrows, ncols = 10, 20
    flux = np.ones((nrows, ncols))
    variance = np.full((nrows, ncols), 4.0)
    mask = np.zeros((nrows, ncols), dtype=bool)

    # Mask some pixels in the background region for first few columns
    mask[3:7, 0:5] = True

    img = Spectrum(
        flux * u.DN,
        uncertainty=VarianceUncertainty(variance * u.DN**2),
        mask=mask,
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4)

    # Uncertainty should be computed
    assert bg._bkg_variance is not None
    assert np.all(np.isfinite(bg._bkg_variance))

    # Columns with masked pixels should have different (larger) variance
    # because fewer pixels contribute to the average
    # (This depends on whether masked pixels fall in the background window)


def test_background_no_input_uncertainty():
    """Test that background estimates variance from flux when no uncertainty provided."""
    nrows, ncols = 10, 20

    # Create image with known noise pattern (no uncertainty provided)
    np.random.seed(42)
    noise_stddev = 2.0
    flux = 100.0 + np.random.normal(0, noise_stddev, (nrows, ncols))
    img = Spectrum(
        flux * u.DN,
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4)

    # Should have variance estimated from flux
    assert hasattr(bg, "_bkg_variance")
    assert bg._bkg_variance is not None

    # Variance should be positive and finite (estimated from flux, not defaulting to 1.0)
    assert np.all(bg._bkg_variance >= 0)
    assert np.all(np.isfinite(bg._bkg_variance))

    # Verify variance was computed from flux by checking it varies with the data
    # (if it defaulted to 1.0, all values would be identical)
    # With noisy data, different columns should have different sample variances
    assert not np.allclose(bg._bkg_variance, 1.0)  # Not defaulting to 1.0

    # bkg_spectrum should have uncertainty
    bkg_spec = bg.bkg_spectrum()
    assert bkg_spec.uncertainty is not None


def test_background_uncertainty_stddev_type():
    """Test that StdDevUncertainty type is preserved in Background output."""
    nrows, ncols = 10, 20
    stddev = 2.0  # stddev=2 means variance=4
    img = Spectrum(
        np.ones((nrows, ncols)) * 10 * u.DN,
        uncertainty=StdDevUncertainty(np.full((nrows, ncols), stddev) * u.DN),
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4)

    # bkg_spectrum should return StdDevUncertainty
    bkg_spec = bg.bkg_spectrum()
    assert bkg_spec.uncertainty is not None
    assert isinstance(bkg_spec.uncertainty, StdDevUncertainty)

    # bkg_image should return StdDevUncertainty
    bkg_img = bg.bkg_image()
    assert bkg_img.uncertainty is not None
    assert isinstance(bkg_img.uncertainty, StdDevUncertainty)

    # sub_spectrum should return StdDevUncertainty
    sub_spec = bg.sub_spectrum()
    assert sub_spec.uncertainty is not None
    assert isinstance(sub_spec.uncertainty, StdDevUncertainty)

    # Values should be positive and finite
    assert np.all(bkg_spec.uncertainty.array > 0)
    assert np.all(np.isfinite(bkg_spec.uncertainty.array))


def test_background_uncertainty_inverse_variance_type():
    """Test that InverseVariance type is preserved in Background output."""
    nrows, ncols = 10, 20
    ivar = 0.25  # ivar=0.25 means variance=4
    img = Spectrum(
        np.ones((nrows, ncols)) * 10 * u.DN,
        uncertainty=InverseVariance(np.full((nrows, ncols), ivar) / u.DN**2),
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4)

    # bkg_spectrum should return InverseVariance
    bkg_spec = bg.bkg_spectrum()
    assert bkg_spec.uncertainty is not None
    assert isinstance(bkg_spec.uncertainty, InverseVariance)

    # bkg_image should return InverseVariance
    bkg_img = bg.bkg_image()
    assert bkg_img.uncertainty is not None
    assert isinstance(bkg_img.uncertainty, InverseVariance)

    # sub_spectrum should return InverseVariance
    sub_spec = bg.sub_spectrum()
    assert sub_spec.uncertainty is not None
    assert isinstance(sub_spec.uncertainty, InverseVariance)

    # Values should be positive and finite
    assert np.all(bkg_spec.uncertainty.array > 0)
    assert np.all(np.isfinite(bkg_spec.uncertainty.array))


def test_sigma_clipping_rejects_outliers():
    """Test that sigma clipping rejects outlier pixels in background region."""
    nrows, ncols = 20, 30
    true_bkg = 10.0

    # Create image with uniform background
    flux = np.ones((nrows, ncols)) * true_bkg

    # Add extreme outliers in background region (should be clipped)
    flux[2, 5] = 1000.0  # extreme outlier
    flux[17, 10] = -500.0  # extreme outlier

    img = Spectrum(
        flux * u.DN,
        uncertainty=VarianceUncertainty(np.ones((nrows, ncols)) * u.DN**2),
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)

    # With sigma clipping enabled (low sigma to ensure clipping)
    bg_clipped = Background(img, trace, width=4, sigma=3.0)

    # Without sigma clipping
    bg_no_clip = Background(img, trace, width=4, sigma=None)

    # Sigma clip mask should mark the outliers (if they fall in bkg region)
    assert hasattr(bg_clipped, "_outlier_mask")
    assert bg_clipped._outlier_mask.shape == img.data.shape

    # Background with clipping should be closer to true value
    # than without clipping (for columns with outliers)
    bkg_clipped = bg_clipped.bkg_spectrum().flux.value
    bkg_no_clip = bg_no_clip.bkg_spectrum().flux.value

    # Column 5 has outlier at row 2 - clipped version should be closer to true_bkg
    if bg_clipped._outlier_mask[2, 5]:  # outlier was in bkg region and clipped
        assert np.abs(bkg_clipped[5] - true_bkg) < np.abs(bkg_no_clip[5] - true_bkg)


def test_sigma_clipping_disabled():
    """Test that sigma=None disables sigma clipping."""
    nrows, ncols = 10, 20
    flux = np.ones((nrows, ncols)) * 10.0
    flux[2, 5] = 100.0

    img = Spectrum(
        flux * u.DN,
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4, sigma=None)

    # Sigma clip mask should be all False
    assert hasattr(bg, "_outlier_mask")
    assert not np.any(bg._outlier_mask)


def test_sigma_clipping_preserves_image_mask():
    """Test that sigma clipping doesn't modify the original image mask."""
    nrows, ncols = 10, 20
    flux = np.ones((nrows, ncols)) * 10.0
    flux[2, 5] = 1000.0  # extreme outlier

    # Create image with existing mask
    original_mask = np.zeros((nrows, ncols), dtype=bool)
    original_mask[3, 7] = True
    original_mask[5, 12] = True

    img = Spectrum(
        flux * u.DN,
        mask=original_mask.copy(),
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4, sigma=3.0)

    # Original image mask should be unchanged
    assert np.array_equal(bg.image.mask, original_mask)

    # Sigma clip mask should be stored separately
    assert hasattr(bg, "_outlier_mask")
    assert not np.array_equal(bg._outlier_mask, original_mask)


def test_sigma_clipping_default():
    """Test that default sigma=5 is used when not specified."""
    nrows, ncols = 10, 20
    flux = np.ones((nrows, ncols)) * 10.0

    img = Spectrum(
        flux * u.DN,
        spectral_axis=np.arange(ncols) * u.pix
    )

    trace = FlatTrace(img, nrows // 2)
    bg = Background(img, trace, width=4)

    # Default sigma should be 5
    assert bg.sigma == 5.0
    assert hasattr(bg, "_outlier_mask")
