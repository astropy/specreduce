import numpy as np

import astropy.units as u
from astropy.nddata import CCDData

from specreduce.extract import BoxcarExtract
from specreduce.tracing import FlatTrace


# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
image = np.ones(shape=(30, 10))
for j in range(image.shape[0]):
    image[j, ::] *= j
image = CCDData(image, unit=u.count)


def test_extraction():
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    boxcar = BoxcarExtract()

    boxcar.apwidth = 5

    trace = FlatTrace(image, 15.0)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))

    trace.set_position(14.5)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 72.5))

    trace.set_position(14.7)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 73.5))

    boxcar.apwidth = 6

    trace.set_position(15.0)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 90.))

    trace.set_position(14.5)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 87.))

    boxcar.apwidth = 4.5

    trace.set_position(15.0)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.5))


def test_outside_image_condition():
    #
    # Trace is such that one of the sky regions lays partially outside the image
    #
    boxcar = BoxcarExtract()

    boxcar.apwidth = 5.
    boxcar.skysep = int(2)
    boxcar.skywidth = 5.

    trace = FlatTrace(image, 22.0)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 99.375))
