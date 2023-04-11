.. _wavelength_calibration:

Wavelength Calibration
======================

Wavelength calibration is currently supported for 1D spectra. Given a list of spectral
lines with known wavelengths and estimated pixel positions on an input calibration
spectrum, you can currently use ``specreduce`` to:

#. Fit an ``astropy`` model to the wavelength/pixel pairs to generate a spectral WCS
   solution for the dispersion.
#. Apply the generated spectral WCS to other `~specutils.Spectrum1D` objects.

1D Wavelength Calibration
-------------------------

The `~specreduce.wavelength_calibration.WavelengthCalibration1D` class can be used
to fit a dispersion model to a list of line positions and wavelengths. Future development
will implement catalogs of known lamp spectra for use in matching observed lines. In the
example below, the line positions (``pixel_centers``) have already been extracted from
``lamp_spectrum``::

    import astropy.units as u
	 from specreduce import WavelengthCalibration1D
	 pixel_centers = [10, 22, 31, 43]
	 wavelengths = [5340, 5410, 5476, 5543]*u.AA
	 test_cal = WavelengthCalibration1D(lamp_spectrum, line_pixels=pixel_centers,
										line_wavelengths=wavelengths)
    calibrated_spectrum = test_cal.apply_to_spectrum(science_spectrum)

The example above uses the default model (`~astropy.modeling.functional_models.Linear1D`)
to fit the input spectral lines, and then applies the calculated WCS solution to a second
spectrum (``science_spectrum``). Any other 1D ``astropy`` model can be provided as the
input ``model`` parameter to the `~specreduce.wavelength_calibration.WavelengthCalibration1D`.
In the above example, the model fit and WCS construction is all done as part of the
``apply_to_spectrum()`` call, but you could also access the `~gwcs.wcs.WCS` object itself
by calling::

    test_cal.wcs

The calculated WCS is a cached property that will be cleared if the ``line_list``, ``model``,
or ``input_spectrum`` properties are updated, since these will alter the calculated dispersion
fit.

You can also provide the input pixel locations and wavelengths of the lines as an
`~astropy.table.QTable` with (at minimum) columns ``pixel_center`` and ``wavelength``,
using the ``matched_line_list`` input argument::

    from astropy.table import QTable
    pixels = [10, 20, 30, 40]*u.pix
    wavelength = [5340, 5410, 5476, 5543]*u.AA
    line_list = QTable([pixels, wavelength], names=["pixel_center", "wavelength"])
    test_cal = WavelengthCalibration1D(lamp_spectrum, matched_line_list=line_list)