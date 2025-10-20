import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty
import astropy.units as u

from specreduce.utils.synth_data import make_2d_arc_image, make_2d_spec_image

def make_science_and_arc(ndisp: int = 1000, ncross: int = 300):
    refx = ndisp // 2
    wcs = WCS(header={
        'CTYPE1': 'AWAV-GRA',  # Grating dispersion function with air wavelengths
        'CUNIT1': 'Angstrom',  # Dispersion units
        'CRPIX1': refx,        # Reference pixel [pix]
        'CRVAL1': 7410,        # Reference value [Angstrom]
        'CDELT1': 3*1.19,      # Linear dispersion [Angstrom/pix]
        'PV1_0': 5.0e5,        # Grating density [1/m]
        'PV1_1': 1,            # Diffraction order
        'PV1_2': 8.05,         # Incident angle [deg]
        'CTYPE2': 'PIXEL',     # Spatial detector coordinates
        'CUNIT2': 'pix',       # Spatial units
        'CRPIX2': 1,           # Reference pixel
        'CRVAL2': 0,           # Reference value
        'CDELT2': 1            # Spatial units per pixel
    })

    science = make_2d_spec_image(ndisp, ncross, add_noise=False, background=0,
                                 wcs=wcs, amplitude_scale=0.25,
                                 trace_coeffs={'c0': 0, 'c1': 30, 'c2': 40},)
    arc = make_2d_arc_image(ndisp, ncross, linelists=['HeI', 'NeI', 'ArI'],  wcs=wcs, line_fwhm=3)
    return science, arc


def plot_2d_spectrum(spec, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    else:
        fig = ax.figure
    ax.imshow(spec, origin='lower', aspect='auto')
    plt.setp(ax, xlabel='Dispersion axis [pix]', ylabel='Cross-disp. axis [pix]')
    return fig, ax