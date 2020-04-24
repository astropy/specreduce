"""
Utilities for defining, loading, and handling spectroscopic calibration data
"""

import os
import pkg_resources
import warnings

import astropy.units as u
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from specutils import Spectrum1D

__all__ = ['get_reference_file_path']

"""
Make specreduce_data optional. If it's available, great and we can access its data via
pkg_resources. If not, we'll fall back to downloading and optionally caching it using
astropy.utils.data.
"""
LOCAL_DATA = True
try:
    import specreduce_data  # noqa
except ModuleNotFoundError:
    warnings.warn(
        "Can't import specreduce_data package. Falling back to downloading data...",
        AstropyUserWarning
    )
    LOCAL_DATA = False
    from astropy.utils.data import download_file

SUPPORTED_EXTINCTION_MODELS = [
    "kpno",
    "ctio",
    "apo",
    "lapalma",
    "mko",
    "mtham",
    "paranal"
]


def get_reference_file_path(path=None, cache=False, show_progress=False):
    """
    Basic function to take a path to a file and load it via pkg_resources if the specreduce_data
    package is available and load it via ghithub raw user content if not.

    Parameters
    ----------
    path : str or None (default: None)
        Filename of reference file relative to the reference_data directory within
        specreduce_data package.

    cache : bool (default: False)
        Set whether file is cached if file is downloaded.

    show_progress : bool (default: False)
        Set whether download progress bar is shown if file is downloaded.

    Returns
    -------
    file_path : str or None
        Path to reference data file or None if the path cannot be constructed or if the file
        itself is not valid.

    Examples
    --------
    >>> from specreduce.calibration_data import get_reference_file_path
    >>> kpno_extinction_file = get_reference_file_path("extinction/kpnoextinct.dat")
    """
    if path is None:
        return None

    if LOCAL_DATA:
        file_path = pkg_resources.resource_filename(
            "specreduce_data",
            os.path.join("reference_data", path)
        )
    else:
        repo_url = "https://raw.githubusercontent.com/astropy/specreduce-data"
        remote_url = f"{repo_url}/master/specreduce_data/reference_data/{path}"
        try:
            file_path = download_file(
                remote_url,
                cache=cache,
                show_progress=show_progress,
                pkgname='specreduce'
            )
        except Exception as e:
            msg = f"Downloading of {path} failed: {e}"
            warnings.warn(msg, AstropyUserWarning)
            return None

    # final sanity check to make sure file_path is actually a file.
    if os.path.isfile(file_path):
        return file_path
    else:
        warnings.warn(f"Able to construct {file_path}, but it is not a file.")
        return None


class AtmosphericExtinction(Spectrum1D):
    """
    Spectrum container for atmospheric extinction as a function of wavelength. If extinction
    and spectral_axis are provided, this will them to build a custom model. If they are not,
    the model parameter will be used to lookup and load a pre-defined atmospheric extinction
    model from the specreduce_data package.

    Parameters
    ----------
    model : str
        Name of atmospheric extinction model provided by specreduce_data package. Valid
        options are:
            kpno - Kitt Peak National Observatory
            ctio - Cerro Tololo International Observatory
            apo - Apache Point Observatory
            lapalma - Roque de los Muchachos Observatory, La Palma, Canary Islands
            mko - Mauna Kea Observatories
            mtham - Lick Observatory, Mt. Hamilton station
            paranal - European Southern Observatory, Cerro Paranal station

    extinction : `astropy.units.Quantity` or astropy.nddata.NDData`-like or None
        Optionally provided extinction data for this spectrum. Used along with spectral_axis
        to build custom atmospheric extinction model.

    spectral_axis : `astropy.units.Quantity` or `specutils.SpectralCoord` or None
        Optional Dispersion information with the same shape as the last (or only)
        dimension of flux, or one greater than the last dimension of flux
        if specifying bin edges. Used along with flux to build custom atmospheric
        extinction model.
    """
    def __init__(self, model="kpno", extinction=None, spectral_axis=None, **kwargs):
        if extinction is None and spectral_axis is None:
            if model not in SUPPORTED_EXTINCTION_MODELS:
                msg = (
                    f"Requested extinction model, {model}, not in list "
                    f"of available models: {SUPPORTED_EXTINCTION_MODELS}"
                )
                raise ValueError(msg)
            model_file = os.path.join("extinction", f"{model}extinct.dat")
            model_path = get_reference_file_path(path=model_file)
            t = Table.read(model_path, format="ascii", names=['wavelength', 'extinction'])

            # the specreduce_data models all provide wavelengths in angstroms
            spectral_axis = t['wavelength'].data * u.angstrom

            # the specreduce_data models all provide extinction in magnitudes at an airmass of 1
            extinction_mag = u.Magnitude(
                t['extinction'].data,
                u.MagUnit(u.dimensionless_unscaled)
            )
            # Spectrum1D wants this to be linear
            extinction = extinction_mag.to(u.dimensionless_unscaled)

        if extinction is not None:
            if not isinstance(extinction, u.Quantity):
                warnings.warn(
                    "Input extinction is not a Quanitity. Assuming linear and dimensionless...",
                    AstropyUserWarning
                )
                extinction *= u.dimensionless_unscaled

        if spectral_axis is None:
            msg = "Missing spectral axis for input extinction data."
            raise ValueError(msg)

        super(AtmosphericExtinction, self).__init__(
            flux=extinction,
            spectral_axis=spectral_axis,
            unit=u.dimensionless_unscaled,
            **kwargs
        )

    @property
    def extinction_mag(self):
        """
        This property returns the extinction in magnitudes
        """
        return self.flux.to(u.mag(u.dimensionless_unscaled))

    @property
    def extinction_frac(self):
        """
        This property returns the linear extinction as a fraction between 0 and 1
        """
        return self.flux


class AtmosphericIRExtinction(AtmosphericExtinction):
    """
    Spectrum container for atmospheric extinction in the IR as a function of wavelength.
    It is built from a pre-calculated table of values from 0.9 to 5.6 microns. The values
    were generated by the ATRAN model, https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi
    (Lord, S. D., 1992, NASA Technical Memorandum 103957). The extinction is given as a
    linear transmission fraction at an airmass of 1.
    """
    def __init__(self, **kwargs):
        data_file = os.path.join("extinction", "atm_trans_am1.0.dat")
        data_path = get_reference_file_path(path=data_file)
        t = Table.read(data_path, format="ascii", names=['wavelength', 'extinction'])

        # spectral axis is given in microns
        spectral_axis = t['wavelength'].data * u.um

        # extinction is given in a dimensionless transmission fraction
        extinction = t['extinction'].data * u.dimensionless_unscaled

        super(AtmosphericIRExtinction, self).__init__(
            extinction=extinction,
            spectral_axis=spectral_axis,
            **kwargs
        )
