import pytest
import numpy as np

import astropy.units as u
from astropy.nddata import VarianceUncertainty
from specutils import Spectrum1D

from specreduce.background import Background
from specreduce.tracing import FlatTrace, ArrayTrace


# NOTE: same test image as in test_extract.py
# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
image = np.ones(shape=(30, 10))
for j in range(image.shape[0]):
    image[j, ::] *= j
image = Spectrum1D(image * u.DN,
                   uncertainty=VarianceUncertainty(np.ones_like(image)))
image_um = Spectrum1D(image.flux,
                      spectral_axis=np.arange(image.data.shape[1]) * u.um,
                      uncertainty=VarianceUncertainty(np.ones_like(image.data)))


def test_background():
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    trace_pos = 15.0
    trace = FlatTrace(image, trace_pos)
    bkg_sep = 5
    bkg_width = 2

    # all the following should be equivalent, whether image's spectral axis
    # is in pixels or physical units:
    bg1 = Background(image, [trace-bkg_sep, trace+bkg_sep], width=bkg_width)
    bg2 = Background.two_sided(image, trace, bkg_sep, width=bkg_width)
    bg3 = Background.two_sided(image, trace_pos, bkg_sep, width=bkg_width)
    assert np.allclose(bg1.bkg_image().flux, bg2.bkg_image().flux)
    assert np.allclose(bg1.bkg_image().flux, bg3.bkg_image().flux)

    bg4 = Background(image_um, [trace-bkg_sep, trace+bkg_sep], width=bkg_width)
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


def test_warnings_errors():
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

    trace = ArrayTrace(image, trace=np.arange(10)+20)  # from 20 to 29
    with pytest.warns(match="background window extends beyond image boundaries"):
        with pytest.raises(ValueError,
                           match="background window does not remain in bounds across entire dispersion axis"):  # noqa
            # 20 + 10 - 3 = 27 (lower edge of window on-image at right side of trace)
            # 29 + 10 - 3 = 36 (lower edge of window off-image at right side of trace)
            Background.one_sided(image, trace, 10, width=3)

    with pytest.raises(ValueError, match="width must be positive"):
        Background.two_sided(image, 25, 2, width=-1)
