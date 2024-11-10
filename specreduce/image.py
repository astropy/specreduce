import astropy.units as u
import numpy as np
from astropy.nddata import NDData, NDDataRef, VarianceUncertainty

CROSSDISP_AXIS: int = 0
DISP_AXIS: int = 1

def as_image(image, disp_axis: int = 1, crossdisp_axis: int | None = None):
    if isinstance(image, SRImage):
        return image
    else:
        return SRImage(image, disp_axis=disp_axis, crossdisp_axis=crossdisp_axis)


class SRImage:
    implemented_masking_methods = 'filter', 'zero-fill', 'omit'

    def __init__(self, image, disp_axis: int = 1, crossdisp_axis: int | None = None):
        self._nddata: NDDataRef | None = None

        # Extract the ndarray from the image object
        # -----------------------------------------
        if isinstance(image, u.quantity.Quantity):
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
                raise ValueError('The cross-dispersion axis must be given for for image cubes with ndim > 2.')

        if disp_axis == crossdisp_axis:
            raise ValueError('The dispersion and cross-dispersion axes cannot be the same.')
        if disp_axis < 0 or crossdisp_axis < 0:
            raise ValueError('The dispersion and cross-dispersion axes cannot be negative.')
        if disp_axis >= data.ndim or crossdisp_axis >= data.ndim:
            raise ValueError('The dispersion and cross-dispersion axes '
                             'must be smaller than the number of image dimensions.')

        # Create the mask
        # ---------------
        if getattr(image, 'mask', None) is not None:
            mask = image.mask.astype(bool) | (~np.isfinite(data))
        else:
            mask =  ~np.isfinite(data)
        if mask.all():
            raise ValueError('Image is fully masked. Check for invalid values.')

        # Extract the unit and uncertainty
        # --------------------------------
        unit = getattr(image, 'unit', u.DN)
        uncertainty = getattr(image, 'uncertainty', None) or VarianceUncertainty(np.ones_like(data))

        # Standardise the axes
        # --------------------
        # This step forces the cross-dispersion axis as the first axis and the dispersion
        # axis as the second axis. This could be done using transpose as well, but this
        # approach works also with image cubes (although we're not supporting them yet).
        data = np.moveaxis(data, [crossdisp_axis, disp_axis], [0, 1])
        mask = np.moveaxis(mask, [crossdisp_axis, disp_axis], [0, 1])
        uncertainty._array = np.moveaxis(uncertainty._array, [crossdisp_axis, disp_axis], [0, 1])

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
