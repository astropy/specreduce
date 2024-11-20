import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import (NDData, NDDataRef, VarianceUncertainty, NDUncertainty,
                            StdDevUncertainty, InverseVariance)

CROSSDISP_AXIS: int = 0
DISP_AXIS: int = 1


def as_image(image, disp_axis: int = 1, crossdisp_axis: int | None = None,
             unit: u.Unit | None = None,
             mask: np.ndarray | None = None, mask_nonfinite: bool = True,
             uncertainty: [np.ndarray | NDUncertainty | None] = None,
             uncertainty_type: str = 'var',
             ensure_var_uncertainty: bool = False,
             require_uncertainty: bool = False):
    if isinstance(image, SRImage):
        return image
    else:
        return SRImage(image, disp_axis=disp_axis, crossdisp_axis=crossdisp_axis,
                       unit=unit, mask=mask, mask_nonfinite=mask_nonfinite,
                       uncertainty=uncertainty, uncertainty_type=uncertainty_type,
                       ensure_var_uncertainty=ensure_var_uncertainty,
                       require_uncertainty=require_uncertainty)


class SRImage:
    implemented_masking_methods = 'filter', 'zero-fill', 'omit'

    def __init__(self, image, disp_axis: int = 1, crossdisp_axis: int | None = None,
                 unit: u.Unit | None = None,
                 mask: np.ndarray | None = None, mask_nonfinite: bool = True,
                 uncertainty: [np.ndarray | NDUncertainty | None] = None,
                 uncertainty_type: str = 'var',
                 ensure_var_uncertainty: bool = False,
                 require_uncertainty: bool = False) -> None:
        self._nddata: NDDataRef | None = None

        # Extract the ndarray from the image object
        # -----------------------------------------
        if isinstance(image, SRImage):
            data = image.data
            disp_axis = image.disp_axis
        elif isinstance(image, u.quantity.Quantity):
            data = image.value
        elif isinstance(image, np.ndarray):
            data = image
        elif isinstance(image, NDData):
            data = image.data
        else:
            raise ValueError('Unrecognized image type.')

        # Carry out dispersion and cross-dispersion axis sanity checks
        # ------------------------------------------------------------
        if crossdisp_axis is None:
            if data.ndim == 2:
                crossdisp_axis = (disp_axis + 1) % 2
            else:
                raise ValueError('The cross-dispersion axis must be given '
                                 ' for for image cubes with ndim > 2.')

        if disp_axis == crossdisp_axis:
            raise ValueError('The dispersion and cross-dispersion axes cannot be the same.')
        if disp_axis < 0 or crossdisp_axis < 0:
            raise ValueError('The dispersion and cross-dispersion axes cannot be negative.')
        if disp_axis >= data.ndim or crossdisp_axis >= data.ndim:
            raise ValueError('The dispersion and cross-dispersion axes '
                             'must be smaller than the number of image dimensions.')

        # Create the mask
        # ---------------
        # TODO: Shouldn't the user-given mask override the default mask in the image?
        #       The order of tests should probably be changed.
        if getattr(image, 'mask', None) is not None:
            mask = image.mask.astype(bool)
        elif mask is not None:
            mask = mask.astype(bool)
        else:
            mask = np.zeros(data.shape, dtype=bool)

        if mask_nonfinite:
            mask |= ~np.isfinite(data)

        if mask.all():
            raise ValueError('Image is fully masked. Check for invalid values.')
        if data.shape != mask.shape:
            raise ValueError('Image and mask shapes must match.')

        # Extract the unit
        # ----------------
        if unit is None:
            unit = getattr(image, 'unit', u.DN)

        # Extract the uncertainty
        # -----------------------
        unc_types = {'var': VarianceUncertainty,
                     'std': StdDevUncertainty,
                     'ivar': InverseVariance}
        if uncertainty is not None:
            if isinstance(uncertainty, np.ndarray):
                uncertainty = unc_types[uncertainty_type](uncertainty)
            elif isinstance(uncertainty, NDUncertainty):
                pass
            else:
                raise ValueError('Unrecognized uncertainty type.')
        elif getattr(image, 'uncertainty', None) is not None:
            uncertainty = image.uncertainty
        elif not require_uncertainty:
            uncertainty = None
        else:
            raise ValueError('Uncertainty information is required but missing.')

        # Try to convert the uncertainty into VarianceUncertainty
        if ensure_var_uncertainty:
            if uncertainty.uncertainty_type != 'var':
                warnings.warn("image object's uncertainty is not"
                              "given as VarianceUncertainty. Trying to "
                              "convert the uncertainty to VarianceUncertainty.")
                uncertainty = uncertainty.represent_as(VarianceUncertainty)

        if uncertainty is not None:
            if uncertainty.array.shape != data.shape:
                raise ValueError('Image and uncertainty shapes must match.')
            if np.any(uncertainty.array < 0):
                raise ValueError("Uncertainty must be fully positive")
            if np.all(uncertainty.array == 0):
                # technically would result in infinities, but since they're all
                # zeros, we can override ones to simulate an unweighted case
                uncertainty._array[:] = 1.0
            if np.any(uncertainty.array == 0):
                m = uncertainty.array == 0
                # exclude such elements by editing the input mask
                mask[m] = True
                # replace the variances to avoid a divide by zero warning
                uncertainty._array[m] = np.nan

        # Standardise the axes
        # --------------------
        # Force the cross-dispersion axis as the first axis and the dispersion
        # axis as the second axis. This could be done using transpose as well, but this
        # approach works also with image cubes (although we're not supporting them yet).
        if (crossdisp_axis, disp_axis) != (0, 1):
            data = np.moveaxis(data, [crossdisp_axis, disp_axis], [0, 1])
            mask = np.moveaxis(mask, [crossdisp_axis, disp_axis], [0, 1])
            uncertainty._array = np.moveaxis(uncertainty._array,
                                             [crossdisp_axis, disp_axis],
                                             [0, 1])

        self._nddata = NDDataRef(data * unit, uncertainty=uncertainty, mask=mask)

    def __repr__(self) -> str:
        return f"<Image unit={self.unit} shape={self.shape}>"

    @property
    def nddata(self) -> NDDataRef:
        return self._nddata

    @property
    def data(self) -> np.ndarray:
        return self._nddata.data

    @property
    def flux(self) -> np.ndarray:
        return self._nddata.data

    @property
    def mask(self) -> np.ndarray:
        return self._nddata.mask

    @property
    def uncertainty(self):
        return self._nddata.uncertainty

    @property
    def unit(self) -> u.Unit:
        return self._nddata.unit

    @property
    def shape(self) -> tuple:
        return self.flux.shape

    @property
    def ndim(self) -> int:
        return self.flux.ndim

    @property
    def wcs(self):
        return self._nddata.wcs

    @property
    def meta(self):
        return self._nddata.meta

    @property
    def disp_axis(self) -> int:
        return DISP_AXIS

    @property
    def crossdisp_axis(self) -> int:
        return CROSSDISP_AXIS

    @property
    def size_disp_axis(self):
        return self.shape[self.disp_axis]

    @property
    def size_crossdisp_axis(self):
        return self.shape[self.crossdisp_axis]

    def subtract(self, image, propagate_uncertainties: bool = True) -> 'SRImage':
        image = SRImage(image)
        return SRImage(self._nddata.subtract(image.nddata,
                                             propagate_uncertainties=propagate_uncertainties))

    def to_masked_array(self, mask_treatment: str = 'filter') -> np.ma.MaskedArray:
        image = self.apply_mask(mask_treatment)
        return np.ma.masked_array(image.flux, mask=image.mask)

    def copy(self) -> 'SRImage':
        """Copy the SRImage object."""
        return SRImage(self._nddata, disp_axis=DISP_AXIS, crossdisp_axis=CROSSDISP_AXIS)

    def apply_mask(self, mask_treatment: str = 'filter') -> 'SRImage':
        """
        Get the image with the given mask treatment applied.

        Parameters
        ----------
        mask_treatment : str, optional
            The method to be used for handling the mask in the image data.
            Allowed values are 'filter', 'zero-fill', and 'omit'.
            - 'filter': Leaves the masked data as is.
            - 'zero-fill': Sets the values of the masked elements to zero and removes the mask.
            - 'omit': Collapses the mask across the specified cross-dispersion axis.

        Returns
        -------
        SRImage
            An image object with the specified mask treatment applied.

        Raises
        ------
        ValueError
            If the `mask_treatment` is not one of the implemented masking methods.
        """
        if mask_treatment not in self.implemented_masking_methods:
            raise ValueError("`mask_treatment` must be one of "
                             f"{self.implemented_masking_methods}")

        image = self.copy()
        if mask_treatment == 'filter':
            return image
        elif mask_treatment == 'zero-fill':
            image._nddata.data[image.mask] = 0.
            image._nddata.mask[:] = False
            return image
        elif mask_treatment == 'omit':
            image._nddata.mask[:] = np.any(image.mask, axis=CROSSDISP_AXIS, keepdims=True)
            return image
