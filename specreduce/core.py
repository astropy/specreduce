# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
import inspect
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.nddata import VarianceUncertainty
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

        img = self._get_data_from_image(image, disp_axis=disp_axis)

        return img

    @staticmethod
    def _get_data_from_image(image, disp_axis=1):
        """Extract data array from various input types for `image`.
           Retruns `np.ndarray` of image data."""

        if isinstance(image, u.quantity.Quantity):
            img = image.value
        elif isinstance(image, np.ndarray):
            img = image
        else:  # NDData, including CCDData and Spectrum1D
            img = image.data

        mask = getattr(image, 'mask', None)

        # next, handle masked and nonfinite data in image.
        # A mask will be created from any nonfinite image data, and combined
        # with any additional 'mask' passed in. If image is being parsed within
        # a specreduce operation that has 'mask_treatment' options, this will be
        # handled as well. Note that input data may be modified if a fill value
        # is chosen to handle masked data. The returned image will always have
        # `image.mask` even if there are no nonfinte or masked values.
        img, mask = self._mask_and_nonfinite_data_handling(image=img, mask=mask)

        # mask (handled above) and uncertainty are set as None when they aren't
        # specified upon creating a Spectrum1D object, so we must check whether
        # these attributes are absent *and* whether they are present but set as None
        if getattr(image, 'uncertainty', None) is not None:
            uncertainty = image.uncertainty
        else:
            uncertainty = VarianceUncertainty(np.ones(img.shape))

        unit = getattr(image, 'unit', u.Unit('DN'))

        spectral_axis = getattr(image, 'spectral_axis',
                                np.arange(img.shape[disp_axis]) * u.pix)

        img = Spectrum1D(img * unit, spectral_axis=spectral_axis,
                         uncertainty=uncertainty, mask=mask)

        return img

    @staticmethod
    def _get_data_from_image(image):
        """Extract data array from various input types for `image`.
           Retruns `np.ndarray` of image data."""

        if isinstance(image, u.quantity.Quantity):
            img = image.value
        if isinstance(image, np.ndarray):
            img = image
        else:  # NDData, including CCDData and Spectrum1D
            img = image.data
        return img

    def _mask_and_nonfinite_data_handling(self, image, mask):
        """
        This function handles the treatment of masked and nonfinite data,
        including input validation.

        All operations in Specreduce can take in a mask for the data as
        part of the input NDData. Additionally, any non-finite values in the
        data that aren't in the user-supplied mask will be combined bitwise
        with the input mask.

        There are three options currently implemented for the treatment
        of masked and nonfinite data - filter, omit, and zero-fill.
        Depending on the step, all or a subset of these three options are valid.

        """

        # valid options depend on Specreduce step, and are set accordingly there
        # for steps that this isn't implemeted for yet, default to 'filter',
        # which will return unmodified input image and mask
        mask_treatment = getattr(self, 'mask_treatment', 'filter')

        # make sure chosen option is valid. if _valid_mask_treatment_methods
        # is not an attribue, proceed with 'filter' to return back inupt data
        # and mask that is combined with nonfinite data.
        if mask_treatment is not None:  # None in operations where masks aren't relevant (FlatTrace)
            valid_mask_treatment_methods = getattr(self, '_valid_mask_treatment_methods', ['filter'])  # noqa
            if mask_treatment not in valid_mask_treatment_methods:
                raise ValueError(f"`mask_treatment` must be one of {valid_mask_treatment_methods}")

        # make sure there is always a 'mask', even when all data is unmasked and finite.
        if mask is not None:
            mask = self.image.mask
            # always mask any previously uncaught nonfinite values in image array
            # combining these with the (optional) user-provided mask on `image.mask`
            mask = np.logical_or(mask, ~np.isfinite(image))
        else:
            mask = ~np.isfinite(image)

        # if mask option is the default 'filter' option, or None,
        # nothing needs to be done. input mask (combined with nonfinite data)
        # remains with data as-is.

        if mask_treatment == 'zero-fill':
            # make a copy of the input image since we will be modifying it
            image = deepcopy(image)

            # if mask_treatment is 'zero_fill', set masked values to zero in
            # image data and drop image.mask. note that this is done after
            # _combine_mask_with_nonfinite_from_data, so non-finite values in
            # data (which are now in the mask) will also be set to zero.
            # set masked values to zero
            image[mask] = 0.

            # masked array with no masked values, so accessing image.mask works
            # but we don't need the actual mask anymore since data has been set to 0
            mask = np.zeros(image.shape)

        elif mask_treatment == 'omit':
            # collapse 2d mask (after accounting for addl non-finite values in
            # data) to a 1d mask, along dispersion axis, to fully mask columns
            # that have any masked values.

            # must have a crossdisp_axis specified to use 'omit' optoin
            if hasattr(self, 'crossdisp_axis'):
                crossdisp_axis = self.crossdisp_axis
            if hasattr(self, '_crossdisp_axis'):
                crossdisp_axis = self._crossdisp_axis

            # create a 1d mask along crossdisp axis - if any column has a single nan,
            # the entire column should be masked
            reduced_mask = np.logical_or.reduce(mask,
                                                axis=crossdisp_axis)

            # back to a 2D mask
            shape = (image.shape[0], 1) if crossdisp_axis == 0 else (1, image.shape[1])
            mask = np.tile(reduced_mask, shape)

        # check for case where entire image is masked.
        if mask.all():
            raise ValueError('Image is fully masked. Check for invalid values.')

        return image, mask

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
