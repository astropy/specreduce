from typing import Sequence

import numpy as np

import astropy.units as u
from astropy.table import QTable
from astropy.wcs import WCS as astropy_WCS
from gwcs.wcs import WCS as gWCS


def match_lines_wcs(
    pixel_positions: np.ndarray | u.Quantity,
    catalog_wavelengths: Sequence[float],
    spectral_wcs: gWCS | astropy_WCS,
    tolerance: float = 5.0,
) -> QTable:
    """
    Use an input spectral WCS to match lines in an extracted spectrum to a catalog of known lines.
    Create matched table of pixel/wavelength positions for lines within a given tolerance of their
    WCS-predicted positions.

    Parameters
    ----------
    pixel_positions : The pixel positions of the lines in the calibration spectrum.

    catalog_wavelengths : The wavelengths of the lines in the catalog.

    spectral_wcs : The spectral WCS of the calibration spectrum.

    tolerance : The matching tolerance in pixels

    Returns
    -------
    QTable
        A table of the matched lines and their pixel/wavelength positions.
    """
    if isinstance(pixel_positions, u.Quantity):
        pixel_positions = pixel_positions.value

    catalog_pixels = spectral_wcs.world_to_pixel(catalog_wavelengths)
    separations = pixel_positions[:, np.newaxis] - catalog_pixels
    matched_loc = np.where(np.abs(separations) < tolerance)
    matched_table = QTable()
    matched_table["pixel_position"] = pixel_positions[matched_loc[0]] * u.pix
    matched_table["wavelength"] = catalog_wavelengths[matched_loc[1]]
    return matched_table

