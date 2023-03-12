"""
Utilities for defining, loading, and handling spectroscopic calibration data
"""

import os
import warnings

import numpy as np

import astropy.units as u
from astropy.table import Table, vstack, QTable
from astropy.utils.data import download_file
from astropy.utils.exceptions import AstropyUserWarning

import synphot
from specutils import Spectrum1D
from specutils.utils.wcs_utils import vac_to_air

__all__ = [
    'get_reference_file_path',
    'get_pypeit_data_path',
    'load_pypeit_calibration_lines',
    'load_MAST_calspec',
    'load_onedstds',
    'AtmosphericExtinction',
    'AtmosphericTransmission'
]

SUPPORTED_EXTINCTION_MODELS = [
    "kpno",
    "ctio",
    "apo",
    "lapalma",
    "mko",
    "mtham",
    "paranal"
]

SPECPHOT_DATASETS = [
    "bstdscal",
    "ctiocal",
    "ctionewcal",
    "eso",
    "gemini",
    "iidscal",
    "irscal",
    "oke1990",
    "redcal",
    "snfactory",
    "spec16cal",
    "spec50cal",
    "spechayescal"
]

PYPEIT_CALIBRATION_LINELISTS = [
    'Ne_IR_MOSFIRE',
    'ArII',
    'CdI',
    'OH_MOSFIRE_H',
    'OH_triplespec',
    'Ar_IR_MOSFIRE',
    'OH_GNIRS',
    'ThAr_XSHOOTER_VIS',
    'ThAr_MagE',
    'HgI',
    'NeI',
    'XeI',
    'OH_MODS',
    'ZnI',
    'OH_GMOS',
    'CuI',
    'ThAr_XSHOOTER_VIS_air',
    'ThAr_XSHOOTER_UVB',
    'OH_NIRES',
    'HeI',
    'FeI',
    'OH_MOSFIRE_J',
    'KrI',
    'Cd_DeVeny1200',
    'Ar_IR_GNIRS',
    'OH_MOSFIRE_Y',
    'ThAr',
    'FeII',
    'OH_XSHOOTER',
    'OH_FIRE_Echelle',
    'OH_MOSFIRE_K',
    'OH_R24000',
    'Hg_DeVeny1200',
    'ArI'
]


def get_reference_file_path(
        path=None,
        cache=True,
        repo_url="https://raw.githubusercontent.com/astropy/specreduce-data",
        repo_branch="main",
        repo_data_path="specreduce_data/reference_data",
        show_progress=False
):
    """
    Utility to load reference data via GitHub raw user content. By default the ``specreduce_data``
    repository at https://github.com/astropy/specreduce-data is used.

    Parameters
    ----------
    path : str or None (default: None)
        Filename of reference file relative to the reference_data directory within
        specified package.

    cache : bool (default: False)
        Set whether file is cached if file is downloaded.

    repo_url : str (default: https://raw.githubusercontent.com/astropy/specreduce-data)
        Base repository URL for the reference data.

    repo_branch : str (default: main)
        Branch of repository from which to fetch the reference data.

    repo_data_path : str (default: specreduce_data/reference_data/)
        Path within the repository where the reference data is located.

    show_progress : bool (default: False)
        Set whether download progress bar is shown if file is downloaded.

    Returns
    -------
    file_path : str or None
        Local path to reference data file or None if the path cannot be constructed or if the file
        itself is not valid.

    Examples
    --------
    >>> from specreduce.calibration_data import get_reference_file_path
    >>> kpno_extinction_file = get_reference_file_path("extinction/kpnoextinct.dat")
    """
    if path is None:
        return None

    remote_url = f"{repo_url}/{repo_branch}/{repo_data_path}/{path}"
    try:
        file_path = download_file(
            remote_url,
            cache=cache,
            show_progress=show_progress,
            pkgname='specreduce'
        )
    except Exception as e:
        msg = f"Downloading of {remote_url} failed: {e}"
        warnings.warn(msg, AstropyUserWarning)
        return None

    # final sanity check to make sure file_path is actually a file.
    if os.path.isfile(file_path):
        return file_path
    else:
        warnings.warn(f"Able to construct {file_path}, but it is not a file.")
        return None


def get_pypeit_data_path(
        path=None,
        cache=True,
        show_progress=False
):
    """
    Convenience utility to facilitate access to ``pypeit`` reference data. The data is accessed
    directly from the release branch on GitHub and downloaded/cached using `~astropy.utils.data.download_file`.

    Parameters
    ----------
    path : str or None (default: None)
        Filename of reference file relative to the reference_data directory within
        ``specreduce_data`` package.

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
    >>> from specreduce.calibration_data import get_pypeit_data_path
    >>> pypeit_he_linelist = get_pypeit_data_path("arc_lines/lists/HeI_lines.dat")
    """
    repo_url="https://raw.githubusercontent.com/pypeit/pypeit"
    repo_branch="release"
    repo_data_path="pypeit/data"

    return get_reference_file_path(
        path=path,
        cache=cache,
        repo_url=repo_url,
        repo_branch=repo_branch,
        repo_data_path=repo_data_path,
        show_progress=show_progress
    )


def load_pypeit_calibration_lines(lamps=None, wave_air=False, cache=True, show_progress=False):
    """
    Load reference calibration lines from ``pypeit`` linelists. The ``pypeit`` linelists are well-curated and have
    been tested across a wide range of spectrographs. The available linelists are defined by
    ``PYPEIT_CALIBRATION_LINELISTS``.

    Parameters
    ----------
    lamps : str or list-like (default: None)
        Lamp or list of lamps to include in output reference linelist. The parlance of "lamp" is retained
        here for consistency with its use in ``pypeit`` and elsewhere. In several of the supported cases the
        "lamp" is the sky itself (e.g. OH lines in the near-IR).

    wave_air : bool (default: False)
        If True, convert the vacuum wavelengths used by ``pypeit`` to air wavelengths using
        `~specutils.utils.wcs_utils.vac_to_air`.

    cache : bool (default: True)
        Toggle caching of downloaded data

    show_progress : bool (default: False)
        Show download progress bar

    Returns
    -------
    linelist: `~astropy.table.Table`
        Table containing the combined calibration line list. ``pypeit`` linelists have the following columns:
        * ``ion``: Ion or molecule generating the line.
        * ``wave``: Vacuum wavelength of the line in Angstroms.
        * ``NIST``: Flag denoting if NIST is the ultimate reference for the line's wavelength.
        * ``Instr``: ``pypeit``-specific instrument flag.
        * ``amplitude``: Amplitude of the line. Beware, this is somewhat arbitrary and not consistent between lists.
        * ``Source``: Source of the line information.
    """
    if lamps is None:
        return None

    linelist = None

    if isinstance(lamps, str):
        lamps = [lamps]

    if isinstance(lamps, (list, tuple, set, np.ndarray)):
        linelists = []
        for l in lamps:
            if l in PYPEIT_CALIBRATION_LINELISTS:
                list_path = f"arc_lines/lists/{l}_lines.dat"
                lines_file = get_pypeit_data_path(list_path, cache=cache, show_progress=show_progress)
                lines_tab = Table.read(lines_file, format='ascii.fixed_width', comment='#')
                if lines_tab is not None:
                    linelists.append(lines_tab)
            else:
                warnings.warn(f"{l} not in the list of supported calibration line lists: {PYPEIT_CALIBRATION_LINELISTS}.")
        if len(linelists) == 0:
            warnings.warn(f"No calibration lines loaded from {lamps}.")
            linelist = None
        else:
            linelist = vstack(linelists)
            # pypeit linelists use vacuum wavelengths in angstroms
            linelist['wave'] *= u.Angstrom
            if wave_air:
                linelist['wave'] = vac_to_air(linelist['wave'])
            linelist = QTable(linelist)
    else:
        raise ValueError(f"Invalid calibration lamps specification, {lamps}. Must be a string or list-like iterable.")

    return linelist


def load_MAST_calspec(filename, cache=True, show_progress=False):
    """
    Load a standard star spectrum from the ``calspec`` database at MAST. These spectra are provided in
    FITS format and are described in detail at:

    https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec  # noqa

    If ``remote`` is True, the spectrum will be downloaded from MAST. Set ``remote`` to False to load
    a local file.

    Parameters
    ----------
    filename : str
        FITS filename of a standard star spectrum, e.g. g191b2b_005.fits. If this is a local file, it will be loaded.
        If not, then a download from MAST will be attempted.
    cache : bool (default = True)
        Toggle whether downloaded data is cached or not.
    show_progress : bool (default = True)
        Toggle whether download progress bar is shown.

    Returns
    -------
    spectrum : `~specutils.Spectrum1D` or None
        If the spectrum can be loaded, return it as a `~specutils.Spectrum1D`.
        Otherwise return None. The spectral_axis units are Å and the flux units are milli-Janskys.
    """
    if os.path.isfile(filename):
        file_path = filename
    else:
        url = f"https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/{filename}"
        try:
            file_path = download_file(
                url,
                cache=cache,
                show_progress=show_progress,
                pkgname='specreduce'
            )
        except Exception as e:
            msg = f"Downloading of {url} failed: {e}"
            warnings.warn(msg, AstropyUserWarning)
            file_path = None

    if file_path is None:
        return None
    else:
        _, wave, flux = synphot.specio.read_fits_spec(file_path)

        # the calspec data stores flux in synphot's FLAM units. convert to flux units
        # supported directly by astropy.units. mJy is chosen since it's the JWST
        # standard and can easily be converted to/from AB magnitudes.
        flux_mjy = synphot.units.convert_flux(wave, flux, u.mJy)
        spectrum = Spectrum1D(spectral_axis=wave, flux=flux_mjy)
        return spectrum


def load_onedstds(dataset="snfactory", specfile="EG131.dat", cache=True, show_progress=False):
    """
    This is a convenience function for loading a standard star spectrum from the 'onedstds'
    dataset in the ``specreduce_data`` package. They will be downloaded from the
    repository on GitHub and cached by default.

    Parameters
    ----------
    dataset : str  (default = "snfactory")
        Standard star spectrum database. Valid options are described in :ref:`specphot_standards`.

    specfile : str (default = "EG131.dat")
        Filename of the standard star spectrum.

    cache : bool (default = True)
        Enable caching of downloaded data.

    show_progress : bool (default = False)
        Show download progress bar if data is downloaded.

    Returns
    -------
    spectrum : None or `~specutils.Spectrum1D`
        If the spectrum can be loaded, return it as a `~specutils.Spectrum1D`.
        Otherwise return None. The spectral_axis units are Å and the flux units are milli-Janskys.
    """
    if dataset not in SPECPHOT_DATASETS:
        msg = (f"Specfied dataset, {dataset}, not in list of supported datasets of "
               f"spectrophotometric standard stars: f{SPECPHOT_DATASETS}")
        warnings.warn(msg, AstropyUserWarning)
        return None

    spec_path = get_reference_file_path(
        path=os.path.join("onedstds", dataset, specfile),
        cache=cache,
        show_progress=show_progress
    )
    if spec_path is None:
        msg = f"Can't load {specfile} from {dataset}."
        warnings.warn(msg, AstropyUserWarning)
        return None

    t = Table.read(spec_path, format="ascii", names=['wavelength', 'ABmag', 'binsize'])

    # the specreduce_data standard star spectra all provide wavelengths in angstroms
    spectral_axis = t['wavelength'].data * u.angstrom

    # the specreduce_data standard star spectra all provide fluxes in AB mag
    flux = t['ABmag'].data * u.ABmag
    flux = flux.to(u.mJy)  # convert to linear flux units
    spectrum = Spectrum1D(spectral_axis=spectral_axis, flux=flux)
    return spectrum


class AtmosphericExtinction(Spectrum1D):
    """
    Spectrum container for atmospheric extinction in magnitudes as a function of wavelength.
    If extinction and spectral_axis are provided, this will use them to build a custom model.
    If they are not, the 'model' parameter will be used to lookup and load a pre-defined
    atmospheric extinction model from the ``specreduce_data`` package.

    Parameters
    ----------
    model : str
        Name of atmospheric extinction model provided by ``specreduce_data``. Valid
        options are:

        kpno - Kitt Peak National Observatory (default)
        ctio - Cerro Tololo International Observatory
        apo - Apache Point Observatory
        lapalma - Roque de los Muchachos Observatory, La Palma, Canary Islands
        mko - Mauna Kea Observatories
        mtham - Lick Observatory, Mt. Hamilton station
        paranal - European Southern Observatory, Cerro Paranal station

    extinction : `~astropy.units.LogUnit`, `~astropy.units.Magnitude`,
    `~astropy.units.dimensionless_unscaled`, 1D list-like, or None
        Optionally provided extinction data for this spectrum. Used along with spectral_axis
        to build custom atmospheric extinction model. If no units are provided, assumed to
        be given in magnitudes.

    spectral_axis : `~astropy.units.Quantity` or `~astropy.coordinates.SpectralCoord` or None
        Optional Dispersion information with the same shape as the last (or only)
        dimension of flux, or one greater than the last dimension of flux
        if specifying bin edges. Used along with flux to build custom atmospheric
        extinction model.

    Properties
    ----------
    extinction_mag : `~astropy.units.Magnitude`
        Extinction expressed in dimensionless magnitudes

    transmission : `~astropy.units.dimensionless_unscaled`
        Extinction expressed as fractional transmission

    """
    def __init__(self, model="kpno", extinction=None, spectral_axis=None,
                 cache=True, show_progress=False, **kwargs):
        if extinction is not None:
            if not isinstance(extinction, u.Quantity):
                warnings.warn(
                    "Input extinction is not a Quanitity. Assuming it is given in magnitudes...",
                    AstropyUserWarning
                )
                extinction = u.Magnitude(
                    extinction,
                    u.MagUnit(u.dimensionless_unscaled)
                ).to(u.dimensionless_unscaled)  # Spectrum1D wants this to be linear
            if isinstance(extinction, (u.LogUnit, u.Magnitude)) or extinction.unit == u.mag:
                # if in log or magnitudes, recast into Magnitude with dimensionless physical units
                extinction = u.Magnitude(
                    extinction.value,
                    u.MagUnit(u.dimensionless_unscaled)
                ).to(u.dimensionless_unscaled)
            if extinction.unit != u.dimensionless_unscaled:
                # if we're given something linear that's not dimensionless_unscaled,
                # it's an error
                msg = "Input extinction must have unscaled dimensionless units."
                raise ValueError(msg)

        if extinction is None and spectral_axis is None:
            if model not in SUPPORTED_EXTINCTION_MODELS:
                msg = (
                    f"Requested extinction model, {model}, not in list "
                    f"of available models: {SUPPORTED_EXTINCTION_MODELS}"
                )
                raise ValueError(msg)
            model_file = os.path.join("extinction", f"{model}extinct.dat")
            model_path = get_reference_file_path(
                path=model_file,
                cache=cache,
                show_progress=show_progress
            )
            t = Table.read(model_path, format="ascii", names=['wavelength', 'extinction'])

            # the specreduce_data models all provide wavelengths in angstroms
            spectral_axis = t['wavelength'].data * u.angstrom

            # the specreduce_data models all provide extinction in magnitudes at an airmass of 1
            extinction = u.Magnitude(
                t['extinction'].data,
                u.MagUnit(u.dimensionless_unscaled)
            ).to(u.dimensionless_unscaled)

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
    def transmission(self):
        """
        This property returns the transmission as a fraction between 0 and 1
        """
        return self.flux


class AtmosphericTransmission(AtmosphericExtinction):
    """
    Spectrum container for atmospheric transmission as a function of wavelength.

    Parameters
    ----------
    data_file : str or `~pathlib.Path` or None
        Name to file containing atmospheric transmission data. Data is assumed to have
        two columns, wavelength and transmission (unscaled dimensionless). If
        this isn't provided, a model is built from a pre-calculated table of values
        from 0.9 to 5.6 microns. The values were generated by the ATRAN model,
        https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi (Lord, S. D., 1992, NASA
        Technical Memorandum 103957). The extinction is given as a linear transmission
        fraction at an airmass of 1 and 1 mm of precipitable water.

    wave_unit : `~astropy.units.Unit` (default = u.um)
        Units for spectral axis.
    """
    def __init__(self, data_file=None, wave_unit=u.um, **kwargs):
        if data_file is None:
            data_path = os.path.join("extinction", "atm_trans_am1.0.dat")
            data_file = get_reference_file_path(path=data_path)

        t = Table.read(data_file, format="ascii", names=['wavelength', 'extinction'])

        # spectral axis is given in microns
        spectral_axis = t['wavelength'].data * wave_unit

        # extinction is given in a dimensionless transmission fraction
        extinction = t['extinction'].data * u.dimensionless_unscaled

        super(AtmosphericTransmission, self).__init__(
            extinction=extinction,
            spectral_axis=spectral_axis,
            **kwargs
        )
