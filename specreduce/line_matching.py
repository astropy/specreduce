import warnings
from copy import deepcopy
from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.modeling import models
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS as astropy_WCS
from gwcs.wcs import WCS as gWCS
from specutils.fitting import find_lines_threshold, fit_lines

from specutils import Spectrum

__all__ = ["find_arc_lines", "match_lines_wcs"]


def _estimate_baseline(
    flux: np.ndarray, window: int | None = None, sigma: float = 3.0, maxiters: int = 5
) -> np.ndarray:
    """Estimate the baseline (pedestal) flux of an arc spectrum with a sigma-clipped median.

    Emission lines are positive outliers, so the sigma-clipped median tracks the
    line-free background level. When ``window`` is given, the median is computed
    in consecutive chunks of ``window`` pixels and linearly interpolated between
    the chunk centres, which lets the estimate follow a slowly varying background.

    Parameters
    ----------
    flux
        The spectrum flux values.
    window
        Approximate width of the chunks in pixels. If ``None``, or if the spectrum
        is too short to hold at least two chunks, a single global median is used.
    sigma
        Clipping threshold in standard deviations passed to
        `~astropy.stats.sigma_clipped_stats`.
    maxiters
        Maximum number of clipping iterations.

    Returns
    -------
    numpy.ndarray
        The baseline estimate, with the same shape as ``flux``.
    """
    flux = np.asarray(flux, dtype=float)
    n = flux.size
    global_median = sigma_clipped_stats(flux, sigma=sigma, maxiters=maxiters)[1]

    if window is None:
        return np.full(n, global_median)

    if isinstance(window, bool) or int(window) != window or window <= 0:
        raise ValueError("baseline window must be a positive integer number of pixels.")
    window = int(window)

    # Split the spectrum into equal-sized chunks of roughly ``window`` pixels.
    nchunks = int(round(n / window))
    if nchunks < 2:
        return np.full(n, global_median)

    edges = np.linspace(0, n, nchunks + 1).astype(int)
    centres = 0.5 * (edges[:-1] + edges[1:] - 1)
    medians = np.array(
        [
            sigma_clipped_stats(flux[lo:hi], sigma=sigma, maxiters=maxiters)[1]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    ok = np.isfinite(medians)
    if ok.sum() < 2:
        return np.full(n, global_median)
    centres, medians = centres[ok], medians[ok]

    # Interpolate between the chunk centres and extrapolate linearly beyond the
    # outermost ones, so a smooth gradient is followed all the way to the edges.
    x = np.arange(n)
    baseline = np.interp(x, centres, medians)
    left = x < centres[0]
    right = x > centres[-1]
    baseline[left] = medians[0] + (x[left] - centres[0]) * np.diff(medians[:2]) / np.diff(
        centres[:2]
    )
    baseline[right] = medians[-1] + (x[right] - centres[-1]) * np.diff(medians[-2:]) / np.diff(
        centres[-2:]
    )
    return baseline


def find_arc_lines(
    spectrum: Spectrum,
    fwhm: float | u.Quantity = 5.0 * u.pix,
    window: float = 3.0,
    noise_factor: float = 5.0,
    subtract_baseline: bool = True,
    baseline_window: int | None = None,
) -> QTable:
    """
    Find arc lines in a spectrum using `~specutils.fitting.find_lines_threshold` and
    then perform gaussian fits to each detected line to refine position and FWHM.

    Parameters
    ----------
    spectrum : The extracted arc spectrum to search for lines. It should be background-subtracted
        unless ``subtract_baseline`` is set, and must have an "uncertainty" attribute.

    fwhm
        Estimated full-width half-maximum of the lines in pixels.

    window
        The window size in units of fwhm to use for the gaussian fits.

    noise_factor
        The factor to multiply the uncertainty by to determine the noise threshold
        in the `~specutils.fitting.find_lines_threshold` routine.

    subtract_baseline
        If ``True``, estimate the baseline (background) flux with a sigma-clipped median
        and subtract it before the line detection. The threshold routine compares the flux
        against zero, so a spectrum with a pedestal above ``noise_factor × uncertainty``
        cannot be searched without this.

    baseline_window
        Width in pixels of the chunks used for a running baseline estimate. If ``None``,
        a single global median is subtracted. Only used when ``subtract_baseline`` is ``True``.

    Returns
    -------
    QTable
        A table of detected arc lines and their properties: centroid, fwhm, and amplitude.
        When the baseline is subtracted, the amplitudes are measured above the baseline.
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

    if subtract_baseline:
        baseline = _estimate_baseline(spectrum.flux.value, window=baseline_window)
        spectrum = Spectrum(
            (spectrum.flux.value - baseline) * spectrum.flux.unit,
            spectral_axis=spectrum.spectral_axis,
            uncertainty=spectrum.uncertainty,
            mask=spectrum.mask,
        )

    with warnings.catch_warnings():
        if subtract_baseline:
            # The baseline is known to be removed, so specutils' continuum heuristic
            # (median flux below 0.01 sigma) only adds noise for strong-line arc spectra.
            warnings.filterwarnings(
                "ignore", message="Spectrum is not below the threshold", category=AstropyUserWarning
            )
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
