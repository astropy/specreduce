import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import CCDData

from specreduce.extract import BoxcarExtract, HorneExtract, OptimalExtract
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


def test_boxcar_outside_image_condition():
    #
    # Trace is such that extraction aperture lays partially outside the image
    #
    trace = FlatTrace(image, 3.0)
    boxcar = BoxcarExtract(image, trace)

    spectrum = boxcar(width=10.)
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 32.0))


def test_boxcar_array_trace():
    trace_array = np.ones_like(image[1]) * 15.
    trace = ArrayTrace(image, trace_array)

    boxcar = BoxcarExtract(image, trace)
    spectrum = boxcar()
    assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))


def test_horne_array_validation():
    #
    # Test HorneExtract scenarios specific to its use with an image of
    # type `~numpy.ndarray` (instead of the default `~astropy.nddata.NDData`).
    #
    trace = FlatTrace(image, 15.0)
    extract = OptimalExtract(image.data, trace)  # equivalent to HorneExtract

    # an array-type image must come with a variance argument
    with pytest.raises(ValueError, match=r'.*array.*variance.*specified.*'):
        ext = extract()

    # an array-type image must have the same dimensions as its variance argument
    with pytest.raises(ValueError, match=r'.*shapes must match.*'):
        err = np.ones_like(image[0])
        ext = extract(variance=err)

    # an array-type image must have the same dimensions as its mask argument
    with pytest.raises(ValueError, match=r'.*shapes must match.*'):
        err = np.ones_like(image)
        mask = np.zeros_like(image[0])
        ext = extract(variance=err, mask=mask)

    # an array-type image given without mask and unit arguments is fine
    # and produces a unitless result
    err = np.ones_like(image)
    ext = extract(variance=err)
    assert ext.unit == u.Unit()


def test_horne_variance_errors():
    trace = FlatTrace(image, 3.0)

    # all zeros are treated as non-weighted (give non-zero fluxes)
    err = np.zeros_like(image)
    mask = np.zeros_like(image)
    extract = HorneExtract(image.data, trace, variance=err, mask=mask, unit=u.Jy)
    ext = extract.spectrum
    assert not np.all(ext == 0)

    # single zero value adjusts mask (does not raise error)
    err = np.ones_like(image)
    err[0] = 0
    mask = np.zeros_like(image)
    ext = extract(variance=err, mask=mask, unit=u.Jy)
    assert not np.all(ext == 0)

    # single negative value raises error
    err = np.ones_like(image)
    err[0] = -1
    mask = np.zeros_like(image)
    with pytest.raises(ValueError, match='variance must be fully positive'):
        ext = extract(variance=err, mask=mask, unit=u.Jy)
