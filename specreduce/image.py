from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, VarianceUncertainty


class SRImage:
    implemented_masking_methods = 'filter', 'zero-fill', 'omit'

    def __init__(self, image, disp_axis: int = 1, crossdisp_axis: int | None = None):
        self._image: NDData = SRImage._parse_image(image)

        if disp_axis == crossdisp_axis:
            raise ValueError('The dispersion and cross-dispersion axes cannot be the same.')
        if disp_axis < 0 or crossdisp_axis < 0:
            raise ValueError('The dispersion and cross-dispersion axes cannot be negative.')
        if disp_axis >= self.ndim or crossdisp_axis >= self.ndim:
            raise ValueError('The dispersion and cross-dispersion axes '
                             'must be smaller than the number of image dimensions.')

        self._disp_axis: int = disp_axis
        self._crossdisp_axis: int = crossdisp_axis

        if self._crossdisp_axis is None and self._image.data.ndim == 2:
            self._crossdisp_axis = (self._disp_axis + 1) % 2
        else:
            raise ValueError('The cross dispersion axis must be given for for image cubes with ndim > 2.')

        # A view to the image data with cross-dispersion axis as the first dimension
        # and dispersion axis as the second dimension.
        self._st_data = np.moveaxis(self._image.data,
                                    [self._crossdisp_axis, self._disp_axis],
                                    [0, 1])

        # A view to the image mask with cross-dispersion axis as the first dimension
        # and dispersion axis as the second dimension.
        self._st_mask = np.moveaxis(self._image.mask,
                                    [self._crossdisp_axis, self._disp_axis],
                                    [0, 1])

        # A view to the image mask with cross-dispersion axis as the first dimension
        # and dispersion axis as the second dimension.
        self._st_uncertainty = np.moveaxis(self._image.uncertainty,
                                    [self._crossdisp_axis, self._disp_axis],
                                    [0, 1])

    def __repr__(self) -> str:
        return f"<Image unit={self.unit} shape={self.shape}>"

    @staticmethod
    def _parse_image(image) -> NDData:
        """
        Parse any of the supported image data types into NDData.

        Parameters
        ----------
        image : Quantity, numpy.ndarray, NDData
            The input image which can be of type Quantity, numpy.ndarray, or NDData.

        Returns
        -------
        NDData
            An NDData object containing the parsed data, with units and uncertainty.

        Raises
        ------
        ValueError
            If the image type is unrecognized.
            If the image is fully masked or contains only invalid values.
        """
        if isinstance(image, u.quantity.Quantity):
            data = image.value
        elif isinstance(image, np.ndarray):
            data = image
        elif isinstance(image, NDData):
            data = image.data
        else:
            raise ValueError('Unrecognized image type.')

        if getattr(image, 'mask', None) is not None:
            mask = image.mask | (~np.isfinite(data))
        else:
            mask =  ~np.isfinite(data)
        if mask.all():
            raise ValueError('Image is fully masked. Check for invalid values.')

        unit = getattr(image, 'unit', u.Unit('DN'))
        uncertainty = getattr(image, 'uncertainty', VarianceUncertainty(np.ones(data.shape)))
        return NDData(data * unit, uncertainty=uncertainty, mask=mask)

    @property
    def nddata(self) -> NDData:
        return self._image

    @property
    def data(self) -> np.ndarray:
        return self._image.data

    @property
    def ordered_data(self) -> np.ndarray:
        """Image data arranged to a shape (crossdisp_axis, disp_axis).

        A view to the data with cross-dispersion axis as the first dimension
        and the dispersion axis as the second dimension.
        """
        return self._st_data

    @property
    def mask(self) -> np.ndarray:
        return self._image.mask

    @property
    def ordered_mask(self) -> np.ndarray:
        """
        Image mask arranged to a shape (crossdisp_axis, disp_axis).

        A view to the image mask with cross-dispersion axis as the first
        dimension and the dispersion axis as the second dimension.
        """
        return self._st_mask

    @property
    def uncertainty(self):
        """
        Image uncertainty arranged to a shape (crossdisp_axis, disp_axis, ...).

        A view to the image uncertainty with cross-dispersion axis as the first
        dimension and the dispersion axis as the second dimension.
        """
        return self._image.uncertainty

    @property
    def ordered_uncertainty(self):
        return self._st_uncertainty

    @property
    def unit(self) -> u.Unit:
        return self._image.unit

    @property
    def shape(self) -> tuple:
        return self._image.data.shape

    @property
    def ordered_shape(self) -> tuple:
        return self._st_data.shape

    @property
    def ndim(self) -> int:
        return self._image.data.ndim

    @property
    def wcs(self):
        return self._image.wcs

    @property
    def meta(self):
        return self._image.meta

    @property
    def disp_axis(self) -> int:
        return self._disp_axis

    @property
    def crossdisp_axis(self) -> int:
        return self._crossdisp_axis

    def copy(self) -> 'SRImage':
        """Copy the image object."""
        return deepcopy(self)

    def masked(self, mask_treatment: str = 'filter') -> 'SRImage':
        """
        Get the NDData image with the given mask treatment applied.

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
        NDData
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
            image._image.data[image.mask] = 0.
            image._image.mask[:] = False
            return image
        elif mask_treatment == 'omit':
            image._image.mask[:] = np.any(image.mask, axis=self._crossdisp_axis, keepdims=True)
            return image
