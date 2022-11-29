# Licensed under a 3-clause BSD style license - see LICENSE.rst

import inspect
import numpy as np

from astropy import units as u
from astropy.nddata import VarianceUncertainty
from dataclasses import dataclass
from specutils import Spectrum1D

__all__ = ['SpecreduceOperation']


class _ImageParser:
    """
    Coerces images from accepted formats to Spectrum1D objects for
    internal use in specreduce's operation classes.

    Fills any and all of uncertainty, mask, units, and spectral axis
    that are missing in the provided image with generic values.
    Accepted image types are:

        - `~specutils.spectra.spectrum1d.Spectrum1D` (preferred)
        - `~astropy.nddata.ccddata.CCDData`
        - `~astropy.nddata.ndddata.NDDData`
        - `~astropy.units.quantity.Quantity`
        - `~numpy.ndarray`
    """
    def _parse_image(self, image, disp_axis=1):
        """
        Convert all accepted image types to a consistently formatted
        Spectrum1D object.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, required
            The image to be parsed. If None, defaults to class' own
            image attribute.
        disp_axis : int, optional
            The index of the image's dispersion axis. Should not be
            changed until operations can handle variable image
            orientations. [default: 1]
        """

        # would be nice to handle (cross)disp_axis consistently across
        # operations (public attribute? private attribute? argument only?) so
        # it can be called from self instead of via kwargs...

        if image is None:
            # useful for Background's instance methods
            return self.image

        if isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, u.quantity.Quantity):
            img = image.value
        else:  # NDData, including CCDData and Spectrum1D
            img = image.data

        # mask and uncertainty are set as None when they aren't specified upon
        # creating a Spectrum1D object, so we must check whether these
        # attributes are absent *and* whether they are present but set as None
        if getattr(image, 'mask', None) is not None:
            mask = image.mask
        else:
            mask = np.ma.masked_invalid(img).mask

        if getattr(image, 'uncertainty', None) is not None:
            uncertainty = image.uncertainty
        else:
            uncertainty = VarianceUncertainty(np.ones(img.shape))

        unit = getattr(image, 'unit', u.Unit('DN'))

        spectral_axis = getattr(image, 'spectral_axis',
                                np.arange(img.shape[disp_axis]) * u.pix)

        return Spectrum1D(img * unit, spectral_axis=spectral_axis,
                          uncertainty=uncertainty, mask=mask)


@dataclass
class SpecreduceOperation(_ImageParser):
    """
    An operation to perform as part of a spectroscopic reduction pipeline.

    This class primarily exists to define the basic API for operations:
    parameters for the operation are provided at object creation,
    and then the operation object is called with the data objects required for
    the operation, which then *return* the data objects resulting from the
    operation.
    """
    def __call__(self):
        raise NotImplementedError('__call__ on a SpecreduceOperation needs to '
                                  'be overridden')

    @classmethod
    def as_function(cls, *args, **kwargs):
        """
        Run this operation as a function.  Syntactic sugar for e.g.,
        ``Operation.as_function(arg1, arg2, keyword=value)`` maps to
        ``Operation(arg2, keyword=value)(arg1)`` (if the ``__call__`` of
        ``Operation`` has only one argument)
        """
        argspec = inspect.getargs(SpecreduceOperation.__call__.__code__)
        if argspec.varargs:
            raise NotImplementedError('There is not a way to determine the '
                                      'number of inputs of a *args style '
                                      'operation')
        ninputs = len(argspec.args) - 1

        callargs = args[:ninputs]
        noncallargs = args[ninputs:]
        op = cls(*noncallargs, **kwargs)
        return op(*callargs)
