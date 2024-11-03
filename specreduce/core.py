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

    implemented_mask_treatment_methods = 'filter', 'zero-fill', 'omit'

    def _parse_image(self, image,
                     disp_axis: int = 1,
                     mask_treatment: str = 'filter') -> Spectrum1D:
        """
        Convert all accepted image types to a consistently formatted Spectrum1D object.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like
            The image to be parsed. If None, defaults to class' own
            image attribute.
        disp_axis
            The index of the image's dispersion axis. Should not be
            changed until operations can handle variable image
            orientations. [default: 1]
        mask_treatment
            Treatment method for the mask.

        Returns
        -------
        Spectrum1D
        """
        # would be nice to handle (cross)disp_axis consistently across
        # operations (public attribute? private attribute? argument only?) so
        # it can be called from self instead of via kwargs...

        if image is None:
            # useful for Background's instance methods
            return self.image

        return self._get_data_from_image(image, disp_axis=disp_axis,
                                         mask_treatment=mask_treatment)

    @staticmethod
    def _get_data_from_image(image,
                             disp_axis: int = 1,
                             mask_treatment: str = 'filter') -> Spectrum1D:
        """
        Extract data array from various input types for `image`.

        Parameters
        ----------
        image : array-like or Quantity
            Input image from which data is extracted. This can be a 2D numpy
            array, Quantity, or an NDData object.
        disp_axis : int, optional
            The dispersion axis of the image.
        mask_treatment : str, optional
            Treatment method for the mask:
            - 'filter' (default): Return the unmodified input image and combined mask.
            - 'zero-fill': Set masked values in the image to zero.
            - 'omit': Mask all pixels along the cross dispersion axis if any value is masked.

        Returns
        -------
        Spectrum1D
        """
        # This works only with 2D images.
        crossdisp_axis = (disp_axis + 1) % 2

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
        img, mask = _ImageParser._mask_and_nonfinite_data_handling(image=img,
                                                                   mask=mask,
                                                                   mask_treatment=mask_treatment,
                                                                   crossdisp_axis=crossdisp_axis)

        # mask (handled above) and uncertainty are set as None when they aren't
        # specified upon creating a Spectrum1D object, so we must check whether
        # these attributes are absent *and* whether they are present but set as None
        if hasattr(image, 'uncertainty'):
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
    def _mask_and_nonfinite_data_handling(image, mask,
                                          mask_treatment: str = 'filter',
                                          crossdisp_axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Handle the treatment of masked and nonfinite data.

        All operations in Specreduce can take in a mask for the data as
        part of the input NDData. Additionally, any non-finite values in the
        data that aren't in the user-supplied mask will be combined bitwise
        with the input mask.

        There are three options currently implemented for the treatment
        of masked and nonfinite data - filter, omit, and zero-fill.
        Depending on the step, all or a subset of these three options are valid.

        Parameters
        ----------
        image : array-like
            The input image data array that may contain nonfinite values.
        mask : array-like or None
            An optional mask array. Nonfinite values in the image will be added to this mask.
        mask_treatment : str
            Specifies how to handle masked data:
            - 'filter' (default): Returns the unmodified input image and combined mask.
            - 'zero-fill': Sets masked values in the image to zero.
            - 'omit': Masks entire columns or rows if any value is masked.
        crossdisp_axis : int
            Axis along which to collapse the 2D mask into a 1D mask for treatment 'omit'.
        """
        if mask_treatment not in _ImageParser.implemented_mask_treatment_methods:
            raise ValueError("`mask_treatment` must be one of "
                             f"{_ImageParser.implemented_mask_treatment_methods}")

        # make sure there is always a 'mask', even when all data is unmasked and finite.
        if mask is not None:
            # always mask any previously uncaught nonfinite values in image array
            # combining these with the (optional) user-provided mask on `image.mask`
            mask = np.logical_or(mask, ~np.isfinite(image))
        else:
            mask = ~np.isfinite(image)

        # if mask option is the default 'filter' option,
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
            mask = np.zeros(image.shape, dtype=bool)

        elif mask_treatment == 'omit':
            # collapse 2d mask (after accounting for addl non-finite values in
            # data) to a 1d mask, along dispersion axis, to fully mask columns
            # that have any masked values.

            # create a 1d mask along crossdisp axis - if any column has a single nan,
            # the entire column should be masked
            reduced_mask = np.logical_or.reduce(mask, axis=crossdisp_axis)

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
