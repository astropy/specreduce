import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import CCDData

from specreduce.extract import BoxcarExtract, HorneExtract
from specreduce.tracing import FlatTrace, ArrayTrace


# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
image = np.ones(shape=(30, 10))
for j in range(image.shape[0]):
    image[j, ::] *= j
image = CCDData(image, unit=u.Jy)


def test_boxcar_extraction():
    #
    # Try combinations of extraction center, and even/odd
    # extraction aperture sizes.
    #
    boxcar = BoxcarExtract()
    trace = FlatTrace(image, 15.0)

    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))
    assert spectrum.unit is not None and spectrum.unit == u.Jy

    trace.set_position(14.5)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 72.5))

    trace.set_position(14.7)
    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 73.5))

    trace.set_position(15.0)
    spectrum = boxcar(image, trace, width=6)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 90.))

    trace.set_position(14.5)
    spectrum = boxcar(image, trace, width=6)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 87.))

    trace.set_position(15.0)
    spectrum = boxcar(image, trace, width=4.5)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.5))

    trace.set_position(15.0)
    spectrum = boxcar(image, trace, width=4.7)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 70.5))

    trace.set_position(14.3)
    spectrum = boxcar(image, trace, width=4.7)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 67.0))


def test_boxcar_outside_image_condition():
    #
    # Trace is such that extraction aperture lays partially outside the image
    #
    boxcar = BoxcarExtract()
    trace = FlatTrace(image, 3.0)

    spectrum = boxcar(image, trace, width=10.)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 32.0))


def test_boxcar_array_trace():
    boxcar = BoxcarExtract()

    trace_array = np.ones_like(image[1]) * 15.

    trace = ArrayTrace(image, trace_array)

    spectrum = boxcar(image, trace)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))


def test_horne_variance_errors():
    extract = HorneExtract()
    trace = FlatTrace(image, 3.0)

    # all zeros are treated as non-weighted (give non-zero fluxes)
    err = np.zeros_like(image)
    mask = np.zeros_like(image)
    ext = extract(image.data, trace, variance=err, mask=np.zeros_like(err), unit=u.Jy)
    assert not np.all(ext == 0)

    # single non-positive value raises error
    err = np.ones_like(image)
    err[0] = 0
    mask = np.zeros_like(image)
    with pytest.raises(ValueError, match='variance must be fully positive'):
        ext = extract(image.data, trace, variance=err, mask=mask, unit=u.Jy)
