import numpy as np

from ..extract import BoxcarExtract


# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
image = np.ones(shape=(30, 10))
for j in range(image.shape[0]):
    image[j, ::] *= j


# Mock a Trace class that represents a line parallel to the image rows.
class Trace:
    def __init__(self, position):
        self.line = np.ones(shape=(10,)) * position


def test_extraction():
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    boxcar = BoxcarExtract()

    boxcar.apwidth = 5

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 72.5))

    trace = Trace(14.7)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 73.5))

    boxcar.apwidth = 6

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 90.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 87.))

    boxcar.apwidth = 4.5

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.5))


def test_sky_extraction():
    #
    # Try combinations of sky extraction parameters
    #
    boxcar = BoxcarExtract()

    boxcar.apwidth = 5.
    boxcar.skysep: int = 2
    boxcar.skywidth = 5.

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 75.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 70.))

    boxcar.skydeg = 1

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 75.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 70.))

    boxcar.skydeg = 2

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 75.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 70.))

    boxcar.apwidth = 7.
    boxcar.skysep: int = 3
    boxcar.skywidth = 8.

    trace = Trace(15.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 105.))

    trace = Trace(14.5)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 98.))


def test_outside_image_condition():
    #
    # Trace is such that one of the sky regions lays partially outside the image
    #
    boxcar = BoxcarExtract()

    boxcar.apwidth = 5.
    boxcar.skysep: int = 2
    boxcar.skywidth = 5.

    trace = Trace(22.0)
    spectrum, bkg_spectrum = boxcar(image, trace)
    assert np.allclose(bkg_spectrum.flux.value, np.full_like(bkg_spectrum.flux.value, 99.375))
