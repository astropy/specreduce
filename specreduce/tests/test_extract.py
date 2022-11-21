import numpy as np
import pytest

import astropy.units as u
from astropy.nddata import CCDData, VarianceUncertainty, UnknownUncertainty

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


def test_horne_image_validation():
    #
    # Test HorneExtract scenarios specific to its use with an image of
    # type `~numpy.ndarray` (instead of the default `~astropy.nddata.NDData`).
    #
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


def test_horne_variance_errors():
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
