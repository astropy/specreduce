from astropy.nddata import NDData
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
import pytest
from specutils import Spectrum1D

from specreduce.background import Background
from specreduce.tracing import FlatTrace, ArrayTrace


def test_background(mk_test_img_raw, mk_test_spec_no_spectral_axis,
                    mk_test_spec_with_spectral_axis):
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
    assert isinstance(bkg_spec, Spectrum1D)
    sub_spec = bg1.sub_spectrum()
    assert isinstance(sub_spec, Spectrum1D)

    # test that width==0 results in no background
    bg = Background.two_sided(image, trace, bkg_sep, width=0)
    assert np.all(bg.bkg_image().flux == 0)

    # test that any NaNs in input image (whether in or outside the window) don't
    # propagate to _bkg_array (which affects bkg_image and sub_image methods) or
    # the final 1D spectra.
    img[0, 0] = np.nan  # out of window
    img[trace_pos, 0] = np.nan  # in window
    stats = ['average', 'median']

    for st in stats:
        bg = Background(img, trace - bkg_sep, width=bkg_width, statistic=st)
        assert np.isnan(bg.image.flux).sum() == 2
        assert np.isnan(bg._bkg_array).sum() == 0
        assert np.isnan(bg.bkg_spectrum().flux).sum() == 0
        assert np.isnan(bg.sub_spectrum().flux).sum() == 0

    bkg_spec_avg = bg1.bkg_spectrum(bkg_statistic='average')
    assert_allclose(bkg_spec_avg.mean().value, 14.5, rtol=0.5)

    bkg_spec_median = bg1.bkg_spectrum(bkg_statistic='median')
    assert_allclose(bkg_spec_median.mean().value, 14.5, rtol=0.5)

    with pytest.raises(ValueError, match="Background statistic max is not supported. "
                                         "Please choose from: average, median, or sum."):
        bg1.bkg_spectrum(bkg_statistic='max')


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
        with pytest.raises(ValueError,
                           match="background window does not remain in bounds across entire dispersion axis"):  # noqa
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
    assert np.all(background.traces[0].trace.data == image.shape[1] / 2.)

    # FlatTrace(s) should be created if number or list of numbers is passed in for `traces`
    background = Background(image, 10., width=5)
    assert isinstance(background.traces[0], FlatTrace)
    assert background.traces[0].trace_pos == 10.

    traces = [10., 15]
    background = Background(image, traces, width=5)
    for i, trace_pos in enumerate(traces):
        assert background.traces[i].trace_pos == trace_pos

    # make sure error is raised if input for `traces` is invalid
    match_str = 'objects, a number or list of numbers to define FlatTraces, ' +\
                'or None to use a FlatTrace in the middle of the image.'
    with pytest.raises(ValueError, match=match_str):
        Background(image, 'non_valid_trace_pos')


class TestMasksBackground():

    """
    Various test functions to test how masked and non-finite data is handled
    in `Background. There are three currently implemented options for masking
    in Background: filter, omit, and zero-fill.
    """

    def mk_img(self, nrows=4, ncols=5, nan_slices=None):
        """
        Make a simple gradient image to test masking in Background.
        Optionally add NaNs to data with `nan_slices`. Returned array is in
        u.DN.
        """

        img = np.tile((np.arange(1., ncols + 1)), (nrows, 1))

        if nan_slices:  # add nans in data
            for s in nan_slices:
                img[s] = np.nan

        return img * u.DN

    @pytest.mark.parametrize("mask", ["filter", "omit", "zero-fill"])
    def test_fully_masked_column(self, mask):
        """
        Test background with some fully-masked columns (not fully masked image).
        In this case, the background value for that fully-masked column should
        be 0.0, with no error or warning raised. This is the case for
        mask_treatment=filter, omit, or zero-fill.
        """

        img = self.mk_img(nrows=10, ncols=10)
        img[:, 0:1] = np.nan

        bkg = Background(img, traces=FlatTrace(img, 6), mask_treatment=mask)
        assert np.all(bkg.bkg_image().data[:, 0:1] == 0.0)

    @pytest.mark.parametrize("mask", ["filter", "omit"])
    def test_fully_masked_image(self, mask):
        """
        Test that the appropriate error is raised by `Background` when image
        is fully masked/NaN.
        """

        with pytest.raises(ValueError, match='Image is fully masked.'):
            # fully NaN image
            img = self.mk_img() * np.nan
            Background(img, traces=FlatTrace(self.mk_img(), 2), mask_treatment=mask)

        with pytest.raises(ValueError, match='Image is fully masked.'):
            # fully masked image (should be equivalent)
            img = NDData(np.ones((4, 5)), mask=np.ones((4, 5)))
            Background(img, traces=FlatTrace(self.mk_img(), 2), mask_treatment=mask)

        # Now test that an image that isn't fully masked, but is fully masked
        # within the window determined by `width`, produces the correct result.
        # only applicable for mask_treatment=filter, because this is the only
        # option that allows a slice of masked values that don't span all rows.
        msg = 'Image is fully masked within background window determined by `width`.'
        with pytest.raises(ValueError, match=msg):
            img = self.mk_img(nrows=12, ncols=12, nan_slices=[np.s_[3:10, :]])
            Background(img, traces=FlatTrace(img, 6), width=7)

    @pytest.mark.filterwarnings("ignore:background window extends beyond image boundaries")
    @pytest.mark.parametrize("method,expected",
                             [("filter", np.array([1., 2., 3., 4., 5., 6., 7.,
                                                  8., 9., 10., 11., 12.])),
                              ("omit", np.array([0., 2., 3., 0., 5., 6.,
                                                 7., 0., 9., 10., 11., 12.])),
                              ("zero-fill", np.array([0.58333333, 2., 3.,
                                                      2.33333333, 5., 6., 7.,
                                                      7.33333333, 9., 10., 11.,
                                                      12.]))])
    def test_mask_treatment_bkg_img_spectrum(self, method, expected):
        """
        This test function tests `Background.bkg_image` and
        `Background.bkg_spectrum` when there is masked data. It also tests
        background subtracting the image, and returning the spectrum of the
        background subtracted image. This test is parameterized over all
        currently implemented mask handling methods (filter, omit, and
        zero-fill) to test that all three work as intended. The window size is
        set to use the entire image array, so warning about background window
        is ignored."""

        img_size = 12  # square 12 x 12 image

        # make image, set some value to nan, which will be masked in the function
        image1 = self.mk_img(nrows=img_size, ncols=img_size,
                             nan_slices=[np.s_[5:10, 0], np.s_[7:12, 3],
                                         np.s_[2, 7]])

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
            background = Background(image, mask_treatment=method,
                                    traces=trace, width=img_size + 1)

            # test background image matches 'expected'
            bk_img = background.bkg_image()
            # change this and following assertions to assert_quantity_allclose once
            # issue #213 is fixed
            np.testing.assert_allclose(bk_img.flux.value,
                                       np.tile(expected, (img_size, 1)))

            # test background spectrum matches 'expected' times the number of rows
            # in cross disp axis, since this is a sum and all values in a col are
            # the same.
            bk_spec = background.bkg_spectrum()
            np.testing.assert_allclose(bk_spec.flux.value, expected * img_size)

    def test_sub_bkg_image(self):
        """
        Test that masked and nonfinite data is handled correctly when subtracting
        background from image, for all currently implemented masking
        options ('filter', 'omit', and 'zero-fill').
        """

        # make image, set some value to nan, which will be masked in the function
        image = self.mk_img(nrows=12, ncols=12,
                            nan_slices=[np.s_[5:10, 0], np.s_[7:12, 3],
                                        np.s_[2, 7]])

        # Calculate a background value using mask_treatment = 'filter'.
        # For 'filter', the flag applies to how masked values are handled during
        # calculation of background for each column, but nonfinite data will
        # remain in input data array
        background_filter = Background(image, mask_treatment='filter',
                                       traces=FlatTrace(image, 6),
                                       width=2)
        subtracted_img_filter = background_filter.sub_image()

        assert np.all(np.isfinite(subtracted_img_filter.data) == np.isfinite(image.data))

        # Calculate a background value using mask_treatment = 'omit'. The input
        # 2d mask is reduced to a 1d mask to mask out full columns in the
        # presence of any nans - this means that (as tested above in
        # `test_mask_treatment_bkg_img_spectrum`) those columns will have 0.0
        # background. In this case, image.mask is expanded to mask full
        # columns - the image itself will not have full columns set to np.nan,
        # so there are still valid background subtracted data values in this
        # case, but the corresponding mask for that entire column will be masked.

        background_omit = Background(image, mask_treatment='omit',
                                     traces=FlatTrace(image, 6),
                                     width=2)
        subtracted_img_omit = background_omit.sub_image()

        assert np.all(np.isfinite(subtracted_img_omit.data) == np.isfinite(image.data))

        # Calculate a background value using mask_treatment = 'zero-fill'. Data
        # values at masked locations are set to 0 in the image array, and the
        # background value calculated for that column will be subtracted
        # resulting in a negative value. The resulting background subtracted
        # image should be fully finite and the mask should be zero everywhere
        # (all unmasked)

        background_zero_fill = Background(image, mask_treatment='zero-fill',
                                          traces=FlatTrace(image, 6),
                                          width=2)
        subtracted_img_zero_fill = background_zero_fill.sub_image()

        assert np.all(np.isfinite(subtracted_img_zero_fill.data))
        assert np.all(subtracted_img_zero_fill.mask == 0)
