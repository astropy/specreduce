# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
import inspect
from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.nddata import VarianceUncertainty, NDData
from specutils import Spectrum1D

__all__ = ["SpecreduceOperation"]

MaskingOption = Literal[
    "apply", "ignore", "propagate", "zero-fill", "nan-fill", "apply_mask_only", "apply_nan_only"
]

ImageLike = np.ndarray | NDData | u.Quantity


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

    # The '_valid_mask_treatment_methods' in the Background, Trace, and Extract
    # classes is a subset of implemented methods.
    implemented_mask_treatment_methods = (
        "apply",
        "ignore",
        "propagate",
        "zero-fill",
        "nan-fill",
        "apply_mask_only",
        "apply_nan_only",
    )

    def _parse_image(
        self, image: ImageLike, disp_axis: int = 1, mask_treatment: MaskingOption = "apply"
    ) -> Spectrum1D:
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
            orientations.
        mask_treatment
            Specifies how to handle masked or non-finite values in the input image.
            The accepted values are:
              - ``apply``: The image remains unchanged, and any existing mask is combined
                with a mask derived from non-finite values.
              - ``ignore``: The image remains unchanged, and any existing mask is dropped.
              - ``propagate``: The image remains unchanged, and any masked or non-finite pixel
                causes the mask to extend across the entire cross-dispersion axis.
              - ``zero-fill``: Pixels that are either masked or non-finite are replaced with 0.0,
                and the mask is dropped.
              - ``nan-fill``:  Pixels that are either masked or non-finite are replaced with nan,
                and the mask is dropped.
              - ``apply_mask_only``: The  image and mask are left unmodified.
              - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a
                new mask is created based on non-finite values.

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

        return self._get_data_from_image(image, disp_axis=disp_axis, mask_treatment=mask_treatment)

    @staticmethod
    def _get_data_from_image(
        image: ImageLike, disp_axis: int = 1, mask_treatment: MaskingOption = "apply"
    ) -> Spectrum1D:
        """
        Extract data array from various input types for `image`.

        Parameters
        ----------
        image
            Input image from which data is extracted. This can be a 2D numpy
            array, Quantity, or an NDData object.
        disp_axis
            The dispersion axis of the image.
        mask_treatment
            Specifies how to handle masked or non-finite values in the input image.

        Returns
        -------
        Spectrum1D
        """
        if isinstance(image, u.quantity.Quantity):
            img = image.value
        elif isinstance(image, np.ndarray):
            img = image
        else:  # NDData, including CCDData and Spectrum1D
            img = image.data

        mask = getattr(image, "mask", None)
        crossdisp_axis = (disp_axis + 1) % 2

        # next, handle masked and non-finite data in image.
        # A mask will be created from any non-finite image data, and combined
        # with any additional 'mask' passed in. If image is being parsed within
        # a specreduce operation that has 'mask_treatment' options, this will be
        # handled as well. Note that input data may be modified if a fill value
        # is chosen to handle masked data. The returned image will always have
        # `image.mask` even if there are no non-finite or masked values.
        img, mask = _ImageParser._mask_and_nonfinite_data_handling(
            image=img, mask=mask, mask_treatment=mask_treatment, crossdisp_axis=crossdisp_axis
        )

        # mask (handled above) and uncertainty are set as None when they aren't
        # specified upon creating a Spectrum1D object, so we must check whether
        # these attributes are absent *and* whether they are present but set as None
        if hasattr(image, "uncertainty"):
            uncertainty = image.uncertainty
        else:
            uncertainty = VarianceUncertainty(np.ones(img.shape))

        unit = getattr(image, "unit", u.Unit("DN"))

        spectral_axis = getattr(image, "spectral_axis", np.arange(img.shape[disp_axis]) * u.pix)

        img = Spectrum1D(
            img * unit, spectral_axis=spectral_axis, uncertainty=uncertainty, mask=mask
        )
        return img

    @staticmethod
    def _mask_and_nonfinite_data_handling(
        image: ImageLike,
        mask: ImageLike | None = None,
        mask_treatment: str = "apply",
        crossdisp_axis: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Handle the treatment of masked and non-finite data.

        All operations in Specreduce can take in a mask for the data as
        part of the input NDData.

        There are five options currently implemented for the treatment
        of masked and non-finite data - apply, ignore, zero-fill, nan-fill,
        apply_mask_only, and apply_nan_only. Depending on the routine,
        all or a subset of these three options are valid.

        Parameters
        ----------
        image : array-like
            The input image data array that may contain non-finite values.
        mask : array-like of bool or None
            An optional Boolean mask array. Non-finite values in the image will be added
            to this mask.
        mask_treatment
            Specifies how to handle masked or non-finite values in the input image.
        """
        if mask_treatment not in _ImageParser.implemented_mask_treatment_methods:
            raise ValueError(
                "'mask_treatment' must be one of "
                f"{_ImageParser.implemented_mask_treatment_methods}"
            )

        if mask is not None and (mask.dtype not in (bool, int)):
            raise ValueError("'mask' must be a boolean or integer array.")

        match mask_treatment:
            case "apply":
                mask = mask | (~np.isfinite(image)) if mask is not None else ~np.isfinite(image)
            case "ignore":
                mask = np.zeros(image.shape, dtype=bool)
            case "propagate":
                if mask is None:
                    mask = ~np.isfinite(image)
                else:
                    mask = mask | (~np.isfinite(image))
                mask[:] = mask.any(axis=crossdisp_axis, keepdims=True)
            case "zero-fill" | "nan-fill":
                mask = mask | (~np.isfinite(image)) if mask is not None else ~np.isfinite(image)
                image = deepcopy(image)
                if mask_treatment == "zero-fill":
                    image[mask] = 0.0
                else:
                    image[mask] = np.nan
                mask[:] = False
            case "apply_nan_only":
                mask = ~np.isfinite(image)
            case "apply_mask_only":
                mask = mask.copy() if mask is not None else np.zeros(image.shape, dtype=bool)

        if mask.all():
            raise ValueError("Image is fully masked. Check for invalid values.")

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
        raise NotImplementedError("__call__ on a SpecreduceOperation needs to " "be overridden")

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
            raise NotImplementedError(
                "There is not a way to determine the "
                "number of inputs of a *args style "
                "operation"
            )
        ninputs = len(argspec.args) - 1

        callargs = args[:ninputs]
        noncallargs = args[ninputs:]
        op = cls(*noncallargs, **kwargs)
        return op(*callargs)
