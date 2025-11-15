import warnings
from copy import deepcopy
from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling import models
from astropy.table import QTable
from astropy.wcs import WCS as astropy_WCS
from gwcs.wcs import WCS as gWCS
from specutils.fitting import find_lines_threshold, fit_lines

from specreduce.compat import Spectrum

__all__ = ["find_arc_lines", "match_lines_wcs"]


def find_arc_lines(
    spectrum: Spectrum,
    fwhm: float | u.Quantity = 5.0 * u.pix,
    window: float = 3.0,
    noise_factor: float = 5.0,
) -> QTable:
    """
    Find arc lines in a spectrum using `~specutils.fitting.find_lines_threshold` and
    then perform gaussian fits to each detected line to refine position and FWHM.

    Parameters
    ----------
    spectrum : The extracted arc spectrum to search for lines. It should be background-subtracted
        and must have an "uncertainty" attribute.

    fwhm
        Estimated full-width half-maximum of the lines in pixels.

    window
        The window size in units of fwhm to use for the gaussian fits.

    noise_factor
        The factor to multiply the uncertainty by to determine the noise threshold
        in the `~specutils.fitting.find_lines_threshold` routine.

    Returns
    -------
    QTable
        A table of detected arc lines and their properties: centroid, fwhm, and amplitude.
    """
    # If fwhm is a float, convert it to a Quantity with the same unit as the spectral axis
    # of the input spectrum.
    if not isinstance(fwhm, u.Quantity):
        fwhm *= spectrum.spectral_axis.unit

    if fwhm.unit != spectrum.spectral_axis.unit:
        raise ValueError("fwhm must have the same units as spectrum.spectral_axis.")

    if spectrum.uncertainty is None:
        spectrum = deepcopy(spectrum)
        spectrum.uncertainty = StdDevUncertainty(np.sqrt(np.abs(spectrum.flux.value)))

    detected_lines = find_lines_threshold(spectrum, noise_factor=noise_factor)
    detected_lines = detected_lines[detected_lines["line_type"] == "emission"]

    centroids = []
    widths = []
    amplitudes = []
    for r in detected_lines:
        g_init = models.Gaussian1D(
            amplitude=spectrum.flux[r["line_center_index"]],
            mean=r["line_center"],
            stddev=fwhm * gaussian_fwhm_to_sigma,
        )
        g_fit = fit_lines(spectrum, g_init, window=window * fwhm)
        centroids.append(g_fit.mean.value * g_fit.mean.unit)
        widths.append(g_fit.stddev * gaussian_sigma_to_fwhm)
        amplitudes.append(g_fit.amplitude.value * g_fit.amplitude.unit)
    line_table = QTable()
    line_table["centroid"] = centroids
    line_table["fwhm"] = widths
    line_table["amplitude"] = amplitudes
    return line_table


def match_lines_wcs(
    pixel_positions: Sequence[float],
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
    pixel_positions
        The pixel positions of the lines in the calibration spectrum.

    catalog_wavelengths
        The wavelengths of the lines in the catalog.

    spectral_wcs
        The spectral WCS of the calibration spectrum.

    tolerance
        The matching tolerance in pixels

    Returns
    -------
    QTable
        A table of the matched lines and their pixel/wavelength positions.
    """

    # This routine uses numpy broadcasting which doesn't always behave with Quantity objects.
    # Pull out the np.ndarray values to avoid those issues.
    if isinstance(pixel_positions, u.Quantity):
        pixel_positions = pixel_positions.value

    # Extra sanity handling to make sure the input Sequence can be converted to an np.array
    try:
        pixel_positions = np.array(pixel_positions, dtype=float)
    except ValueError as e:
        raise ValueError(f"pixel_positions must be convertable to np.array with dtype=float: {e}")

    catalog_pixels = spectral_wcs.world_to_pixel(catalog_wavelengths)
    separations = pixel_positions[:, np.newaxis] - catalog_pixels
    matched_loc = np.where(np.abs(separations) < tolerance)
    matched_table = QTable()
    matched_table["pixel_center"] = pixel_positions[matched_loc[0]] * u.pix
    matched_table["wavelength"] = catalog_wavelengths[matched_loc[1]]

    if len(matched_table) == 0:
        warnings.warn("No lines matched within the given tolerance.")

    return matched_table
