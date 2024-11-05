from copy import deepcopy

import astropy.units as u
import numpy as np

from astropy.nddata import NDData, VarianceUncertainty


class Image:
    implemented_masking_methods = 'filter', 'zero-fill', 'omit'

    def __init__(self, image, disp_axis: int = 1, crossdisp_axis: int | None = None):
        self._image: NDData = Image._parse_image(image)
        self._disp_axis: int = disp_axis
        self._crossdisp_axis: int = crossdisp_axis

        if self._crossdisp_axis is None and self._image.data.ndim == 2:
            self._crossdisp_axis = (self._disp_axis + 1) % 2
        else:
            raise ValueError('The cross dispersion axis must be given for for image cubes with ndim > 2.')

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
    def unit(self) -> u.Unit:
        return self._image.unit

    @property
    def shape(self) -> tuple:
        return self._image.data.shape

    @property
    def data(self) -> np.ndarray:
        return self._image.data

    @property
    def mask(self) -> np.ndarray:
        return self._image.mask

    def masked(self, mask_treatment: str = 'filter') -> NDData:
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

        image = deepcopy(self._image)
        if mask_treatment == 'filter':
            return image
        elif mask_treatment == 'zero-fill':
            image = deepcopy(self._image)
            image.data[image.mask] = 0.
            image.mask[:] = False
            return image
        elif mask_treatment == 'omit':
            image = deepcopy(self._image)
            image.mask[:] = np.any(image.mask, axis=self._crossdisp_axis, keepdims=True)
            return image
