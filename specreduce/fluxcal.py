import os

import numpy as np
from astropy import units as u
from astropy.constants import c as cc
from astropy.table import Table
from scipy.interpolate import UnivariateSpline

from specreduce.compat import Spectrum
from specreduce.core import SpecreduceOperation


__all__ = ['FluxCalibration']


class FluxCalibration(SpecreduceOperation):
    """
    Carries out routine flux calibration operations.

    Parameters
    ----------
    object_spectrum : a Spectrum object
        The observed object spectrum to apply the sensfunc to,
        with the wavelength of the data points in Angstroms
        as the ``spectral_axis``, and the magnitudes of the
        data as the ``flux``.
    airmass : float
        The value of the airmass. Note: NOT the header keyword.
    zeropoint : float, optional
        Conversion factor for mag->flux. (Default is 48.60).

    """

    def __call__(self, object_spectrum, airmass=1.00, zeropoint=48.60):
        self.object_spectrum = object_spectrum
        self.airmass = airmass
        self.zeropoint = zeropoint

    def mag2flux(self, spec_in=None):
        """
        Convert magnitudes to flux units. This is important for dealing with standards
        and files from IRAF, which are stored in AB mag units. To be clear, this converts
        to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF.

        Parameters
        ----------
        spec_in : a Spectrum object, optional
            An input spectrum with wavelength of the data points in Angstroms
            as the ``spectral_axis`` and magnitudes of the data as the ``flux``.

        Returns
        -------
        spec_out : specutils.Spectrum
            Containing both ``flux`` and ``spectral_axis`` data in which
            the ``flux`` has been properly converted from mag->flux.

        """
        if spec_in is None:
            spec_in = self.object_spectrum

        lamb = spec_in.spectral_axis
        mag = spec_in.flux

        flux = (10.0**((mag + self.zeropt) / (-2.5))) * (cc.to('AA/s').value / lamb ** 2.0)
        flux = flux * u.erg / u.s / u.angstrom / (u.cm * u.cm)

        spec_out = Spectrum(spectral_axis=lamb, flux=flux)

        return spec_out

    @staticmethod
    def obs_extinction(obs_file):
        """
        Load the observatory-specific airmass extinction file from the supplied library

        Parameters
        ----------
        obs_file : str, {'apoextinct.dat', 'ctioextinct.dat', 'kpnoextinct.dat', 'ormextinct.dat'}
            The observatory-specific airmass extinction file. If not known for your
            observatory, use one of the provided files (e.g. `kpnoextinct.dat`).
            Following IRAF standard, extinction files have 2-column format
            wavelength (Angstroms), Extinction (Mag per Airmass).

        Returns
        -------
        Xfile: `~astropy.table.Table`
            Table with the observatory extinction data.

        """
        if len(obs_file) == 0:
            raise ValueError('Must select an observatory extinction file.')

        dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'datasets', 'extinction')

        if not os.path.isfile(os.path.join(dir, obs_file)):
            msg = "No valid standard star found at: " + os.path.join(dir, obs_file)
            raise ValueError(msg)

        # To read in the airmass extinction curve
        Xfile = Table.read(os.path.join(dir, obs_file), format='ascii', names=('wave', 'X'))
        Xfile['wave'].unit = 'AA'

        return Xfile

    def airmass_cor(self, Xfile):
        """
        Correct the spectrum based on the airmass.
        Requires observatory extinction file.

        Parameters
        ----------
        Xfile : `~astropy.table.Table`
            The extinction table from `obs_extinction`, with columns ('wave', 'X')
            that have standard units of: (angstroms, mag/airmass).

        Returns
        -------
        airmass_cor_spec : specutils.Spectrum
            The airmass-corrected Spectrum object.

        """
        object_spectrum = self.mag2flux()
        airmass = self.airmass

        obj_wave, obj_flux = object_spectrum.spectral_axis, object_spectrum.flux

        # linear interpol airmass extinction onto observed wavelengths
        new_X = np.interp(obj_wave.value, Xfile['wave'], Xfile['X'])

        # air_cor in units of mag/airmass, convert to flux/airmass
        airmass_ext = 10.0**(0.4 * airmass * new_X)

        airmass_cor_spec = Spectrum(flux=obj_flux * airmass_ext, spectral_axis=obj_wave)

        return airmass_cor_spec

    def onedstd(self, stdstar):
        """
        Load the onedstd from the supplied library.

        Parameters
        ----------
        stdstar : str
            Name of the standard star file in the specreduce/datasets/onedstds
            directory to be used for the flux calibration. The user must provide the
            subdirectory and file name.

            For example::

                standard_sensfunc(obj_wave, obj_flux, stdstar='spec50cal/bd284211.dat', mode='spline')

            If no std is supplied, or an improper path is given, raises a ValueError.

        Returns
        -------
        standard : `~astropy.table.Table`
            A table with the onedstd data.
        """  # noqa: E501
        std_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'datasets', 'onedstds')

        if not os.path.isfile(os.path.join(std_dir, stdstar)):
            msg = "No valid standard star found at: " + os.path.join(std_dir, stdstar)
            raise ValueError(msg)

        standard = Table.read(os.path.join(std_dir, stdstar),
                              format='ascii', names=('wave', 'mag', 'width'))
        standard['wave'].unit = u.angstrom
        standard['width'].unit = u.angstrom

        # Standard star spectrum is stored in magnitude units (IRAF conventions)
        std_flux = self.mag2flux(spec_in=Spectrum(flux=standard['mag'],
                                                  spectral_axis=standard['wave']))
        std_flux = std_flux.flux
        standard['mag'].unit = u.mag
        standard.add_column(std_flux, name='flux')

        return standard

    def standard_sensfunc(self, standard, mode='linear', polydeg=9,
                          badlines=[6563, 4861, 4341], display=False):
        """
        Compute the standard star sensitivity function.

        Parameters
        ----------
        standard : `~astropy.table.Table`
            Output from ``onedstd``, has columns ('wave', 'width', 'mag', 'flux').
        mode : str, optional
            Can be "linear", "spline", or "poly". (Default is linear).
        polydeg : float, optional
            If mode='poly', this is the order of the polynomial to fit through.
            (Default is 9.)
        display : bool, optional
            If True, plot the sensfunc. (Default is False.)
            This requires ``matplotlib`` to be installed.
        badlines : array-like list
            A list of values (lines) to mask-out of when generating sensfunc.

        Returns
        -------
        sensfunc_spec : specutils.Spectrum
            The sensitivity function in the covered wavelength range
            for the given standard star.

        """
        spec = self.mag2flux()

        obj_wave, obj_flux = spec.spectral_axis, spec.flux

        # Automatically exclude some lines b/c resolution dependent response
        badlines = np.array(badlines, dtype='float')  # Balmer lines

        # Down-sample (ds) the observed flux to the standard's bins
        obj_flux_ds = np.array([], dtype=np.float)
        obj_wave_ds = np.array([], dtype=np.float)
        std_flux_ds = np.array([], dtype=np.float)
        for i in range(len(standard['flux'])):
            rng = np.where((obj_wave.value >= standard['wave'][i] - standard['width'][i] / 2.0) &
                           (obj_wave.value < standard['wave'][i] + standard['width'][i] / 2.0))[0]

            IsH = np.where((badlines >= standard['wave'][i] - standard['width'][i] / 2.0) &
                           (badlines < standard['wave'][i] + standard['width'][i] / 2.0))[0]

            # Does this bin contain observed spectra, and no Balmer lines?
            if (len(rng) > 1) and (len(IsH) == 0):
                obj_flux_ds = np.append(obj_flux_ds, np.nanmean(obj_flux.value[rng]))
                obj_wave_ds = np.append(obj_wave_ds, standard['wave'][i])
                std_flux_ds = np.append(std_flux_ds, standard['flux'][i])

        # the ratio between the standard star catalog flux and observed flux
        ratio = np.abs(std_flux_ds / obj_flux_ds)

        # The actual fit the log of this sensfunc ratio
        # Since IRAF does the 2.5*log(ratio), everything would be in mag units
        LogSensfunc = np.log10(ratio)

        # If invalid interpolation mode selected, make it spline
        if mode.lower() not in ('linear', 'spline', 'poly'):
            mode = 'spline'
            import warnings
            warnings.warn("WARNING: invalid mode set. Changing to default mode 'spline'")

        # Interpolate the calibration (sensfunc) on to observed wavelength grid
        if mode.lower() == 'linear':
            sensfunc2 = np.interp(obj_wave.value, obj_wave_ds, LogSensfunc)
        elif mode.lower() == 'spline':
            spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2, s=0.0025)
            sensfunc2 = spl(obj_wave.value)
        elif mode.lower() == 'poly':
            fit = np.polyfit(obj_wave_ds, LogSensfunc, polydeg)
            sensfunc2 = np.polyval(fit, obj_wave.value)

        sensfunc_out = (10 ** sensfunc2) * standard['flux'].unit / obj_flux.unit

        sensfunc_spec = Spectrum(spectral_axis=obj_wave, flux=sensfunc_out)

        if display is True:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(obj_wave, obj_flux * sensfunc_out, c="C0",
                     label="Observed x sensfunc", alpha=0.5)

            # plt.scatter(standard['wave'], std_flux, color='C1', alpha=0.75, label="stdstar")
            plt.scatter(obj_wave_ds, std_flux_ds, color='C1', alpha=0.75)

            plt.xlabel("Wavelength")
            plt.ylabel("Flux")

            plt.xlim(np.nanmin(obj_wave.value), np.nanmax(obj_wave.value))
            plt.ylim(np.nanmin(obj_flux.value * sensfunc_out.value) * 0.98,
                     np.nanmax(obj_flux.value * sensfunc_out.value) * 1.02)
            # plt.legend()
            plt.show()

        return sensfunc_spec

    def apply_sensfunc(self, sensfunc):
        """
        Apply the derived sensitivity function, converts observed units (e.g. ADU/s)
        to physical units (e.g., erg/s/cm2/A).
        Sensitivity function is first linearly interpolated onto the wavelength scale
        of the observed data, and then directly multiplied.

        Parameters
        ----------
        sensfunc : `~astropy.table.Table`
            The output of ``standard_sensfunc``, table has columns ('wave', 'S').

        Returns
        -------
        fluxcal_spec : specutils.Spectrum
            The sensfunc corrected ``Spectrum`` object.

        """
        spec = self.mag2flux()

        obj_wave, obj_flux = spec.spectral_axis, spec.flux

        # Sort, in case the sensfunc wavelength axis is backwards
        ss = np.argsort(obj_wave.value)

        # Interpolate the sensfunc onto the observed wavelength axis
        sensfunc2 = np.interp(obj_wave.value, sensfunc['wave'][ss], sensfunc['S'][ss])

        object_spectrum = obj_flux * (sensfunc2 * sensfunc['S'].unit)

        fluxcal_spec = Spectrum(spectral_axis=obj_wave, flux=object_spectrum)

        return fluxcal_spec
