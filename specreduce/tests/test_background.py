import pytest
import numpy as np

import astropy.units as u
from astropy.nddata import CCDData
from specutils import Spectrum1D

from specreduce.background import Background
from specreduce.tracing import FlatTrace, KosmosTrace


# NOTE: same test image as in test_extract.py
# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
image = np.ones(shape=(30, 10))
for j in range(image.shape[0]):
    image[j, ::] *= j
image = CCDData(image, unit=u.Jy)


def test_background():
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    trace_pos = 15.0
    trace = FlatTrace(image, trace_pos)
    bkg_sep = 5
    bkg_width = 2
    # all the following should be equivalent:
    bg1 = Background(image, [trace-bkg_sep, trace+bkg_sep], width=bkg_width)
    bg2 = Background.two_sided(image, trace, bkg_sep, width=bkg_width)
    bg3 = Background.two_sided(image, trace_pos, bkg_sep, width=bkg_width)
    assert np.allclose(bg1.bkg_array, bg2.bkg_array)
    assert np.allclose(bg1.bkg_array, bg3.bkg_array)

    # test that creating a one_sided background works
    Background.one_sided(image, trace, bkg_sep, width=bkg_width)

    # test that image subtraction works
    sub1 = image - bg1
    sub2 = bg1.sub_image(image)
    sub3 = bg1.sub_image()
    assert np.allclose(sub1, sub2)
    assert np.allclose(sub1, sub3)

    bkg_spec = bg1.bkg_spectrum()
    assert isinstance(bkg_spec, Spectrum1D)
    sub_spec = bg1.sub_spectrum()
    assert isinstance(sub_spec, Spectrum1D)


def test_oob():
    # image.shape (30, 10)
    with pytest.warns(match="background window extends beyond image boundaries"):
        Background.two_sided(image, 25, 4, width=3)

    trace = KosmosTrace(image, guess=15, bins=5)
    with pytest.raises(ValueError,
                       match="background window does not remain in bounds across entire dispersion axis"):  # noqa
        Background.two_sided(image, trace, 15, width=3)
