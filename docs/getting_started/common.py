from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, Model

from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty, CCDData, VarianceUncertainty
import astropy.units as u
from photutils.datasets import apply_poisson_noise

from specreduce.utils.synth_data import make_2d_arc_image, make_2d_trace_image


def make_2d_spec_image(
    nx: int = 1000,
    ny: int = 300,
    wcs: WCS | None = None,
    extent: Sequence[int | float] = (6500, 9500),
    wave_unit: u.Unit = u.Angstrom,
    wave_air: bool = False,
    background: int | float = 5,
    line_fwhm: float = 5.0,
    linelists: list[str] = ("OH_GMOS"),
    airglow_amplitude: float = 1.0,
    spectrum_amplitude: float = 1.0,
    tilt_func: Model = models.Legendre1D(degree=0),
    trace_center: int | float | None = None,
    trace_order: int = 3,
    trace_coeffs: None | dict[str, int | float] = None,
    source_profile: Model = models.Moffat1D(amplitude=10, alpha=0.1),
    add_noise: bool = True,
) -> CCDData:
    """
    Generate a simulated 2D spectroscopic image.

    This function creates a two-dimensional synthetic spectroscopic image, combining
    arc (wavelength calibration) data and trace (spatial profile) data. The image simulates
    a realistic spectral profile with contributions from background, airglow, and a modeled
    source profile. Noise can optionally be added to the resulting image. It employs models
    for traces and tilts, and allows customization of wavelength range, amplitude scales,
    and noise addition.

    Parameters
    ----------
    nx : int, optional
        Number of columns (spatial dimension) in the generated image, by default 3000.
    ny : int, optional
        Number of rows (dispersion dimension) in the generated image, by default 1000.
    wcs : WCS or None, optional
        World Coordinate System (WCS) object for the image. Specifies spectral coordinates.
        If None, WCS is not assigned, by default None.
    extent : Sequence[int or float], optional
        Wavelength range for the generated image in units determined by `wave_unit`.
        Defined as a tuple (min, max), by default (6500, 9500).
    wave_unit : Unit, optional
        Unit of the generated wavelength axis, by default astropy.units.Angstrom.
    wave_air : bool, optional
        If True, use air wavelengths. If False, use vacuum wavelengths, by default False.
    background : int or float, optional
        Constant background level added to the image, by default 5.
    line_fwhm : float, optional
        Full-width half maximum (in pixels) for spectral lines in the image, by default 5.0.
    linelists : list of str, optional
        Names of line lists (e.g., emission lines) to simulate in the arc image,
        by default ["OH_GMOS"].
    airglow_amplitude : float, optional
        Scaling factor for the airglow contribution to the image, by default 1.0.
    spectrum_amplitude : float, optional
        Scaling factor for the primary spectrum contribution (trace image), by default 1.0.
    amplitude_scale : float, optional
        Global amplitude scaling factor applied to both arcs and traces, by default 1.0.
    tilt_func : Model, optional
        Astropy model representing the spectral tilt across the spatial axis. By default,
        a zero-degree Legendre polynomial (no tilt) is used.
    trace_center : int, float, or None, optional
        Central position of the trace in spatial pixels. If None, defaults to trace modeled
        by coefficients in `trace_coeffs`, by default None.
    trace_order : int, optional
        Polynomial order to model the trace profile spatial variation, by default 3.
    trace_coeffs : None or dict of str to int or float, optional
        Coefficients for modeling the trace position (spatial axis). If None, defaults to
        {"c0": 0, "c1": 50, "c2": 100}, by default None.
    source_profile : Model, optional
        Astropy model to simulate the source profile along the trace, by default Moffat1D with
        amplitude=10 and alpha=0.1.
    add_noise : bool, optional
        If True, adds Poisson noise to the generated spectral image, by default True.

    Returns
    -------
    CCDData
        A CCDData object containing the simulated 2D spectroscopic image. The data includes
        contributions from arc lines, traces, airglow, and background, with optional noise
        added. The WCS information, if provided, is preserved.
    """

    if trace_coeffs is None:
        trace_coeffs = {"c0": 0, "c1": 50, "c2": 100}

    arc_image = make_2d_arc_image(
        nx=nx,
        ny=ny,
        wcs=wcs,
        extent=extent,
        wave_unit=wave_unit,
        wave_air=wave_air,
        background=0,
        line_fwhm=line_fwhm,
        linelists=linelists,
        tilt_func=tilt_func,
        add_noise=False,
    )

    trace_image = make_2d_trace_image(
        nx=nx,
        ny=ny,
        background=0,
        trace_center=trace_center,
        trace_order=trace_order,
        trace_coeffs=trace_coeffs,
        profile=source_profile,
        add_noise=False,
    )

    wl = wcs.spectral.pixel_to_world(np.arange(nx)).to(u.nm).value
    signal = 0.8 + 0.2 * np.abs(np.sin((wl - 650) / 100 * 2 * np.pi)) ** 5

    n = lambda a: a / a.max()
    spec_image = (
        airglow_amplitude * n(arc_image.data)
        + spectrum_amplitude * n(trace_image.data) * signal
        + background
    )

    if add_noise:
        from photutils.datasets import apply_poisson_noise

        spec_image = apply_poisson_noise(spec_image)

    return CCDData(
        spec_image,
        unit=u.count,
        wcs=arc_image.wcs,
        uncertainty=StdDevUncertainty(np.sqrt(spec_image)),
    )


def make_science_and_arcs(ndisp: int = 1000, ncross: int = 300):
    refx = ndisp // 2
    wcs = WCS(
        header={
            "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
            "CUNIT1": "Angstrom",  # Dispersion units
            "CRPIX1": refx,  # Reference pixel [pix]
            "CRVAL1": 7410,  # Reference value [Angstrom]
            "CDELT1": 3 * 1.19,  # Linear dispersion [Angstrom/pix]
            "PV1_0": 5.0e5,  # Grating density [1/m]
            "PV1_1": 1,  # Diffraction order
            "PV1_2": 8.05,  # Incident angle [deg]
            "CTYPE2": "PIXEL",  # Spatial detector coordinates
            "CUNIT2": "pix",  # Spatial units
            "CRPIX2": 1,  # Reference pixel
            "CRVAL2": 0,  # Reference value
            "CDELT2": 1,  # Spatial units per pixel
        }
    )

    science = make_2d_spec_image(
        ndisp,
        ncross,
        add_noise=True,
        background=10,
        wcs=wcs,
        airglow_amplitude=20.0,
        spectrum_amplitude=80.0,
        trace_coeffs={"c0": 0, "c1": 30, "c2": 40},
        source_profile=models.Moffat1D(amplitude=1, alpha=0.3),
    )

    arcargs = dict(wcs=wcs, line_fwhm=3, background=0, add_noise=False)
    arcs = []
    for linelist in ["HeI", "NeI", "ArI"]:
        arc = make_2d_arc_image(ndisp, ncross, linelists=[linelist], **arcargs)
        arc.data = apply_poisson_noise(100*(arc.data / arc.data.max()) + 10) - 10
        arcs.append(arc)
    return science, arcs


def plot_2d_spectrum(spec, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    else:
        fig = ax.figure
    ax.imshow(spec, origin="lower", aspect="auto")
    if label is not None:
        ax.text(0.98, 0.9, label, va="top", ha="right", transform=ax.transAxes, c="w")
    plt.setp(ax, xlabel="Dispersion axis [pix]", ylabel="Cross-disp. axis [pix]")
    return fig, ax
