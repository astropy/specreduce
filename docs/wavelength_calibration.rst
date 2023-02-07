.. _wavelength_calibration:

Wavelength Calibration
======================

Wavelength calibration is currently supported for 1D spectra. Given a list of spectral
lines with known wavelengths and estimated pixel positions on an input calibration
spectrum, you can currently use ``specreduce`` to:

#. Refine the pixel position estimates of the lines based on features in the input
   calibration spectrum.
#. Fit an ``astropy`` model to the wavelength/pixel pairs to generate a spectral WCS
   solution for the dispersion.
#. Apply the generated spectral WCS to other `~specutils.Spectrum1D` objects.

Calibration Lines
-----------------

``specreduce`` provides a `~specreduce.wavelength_calibration.CalibrationLine` class for
encoding the information about each spectral line to be used in the dispersion model
solution fitting. The minimum information required is the wavelength and estimated pixel
location on the calibration spectrum of each line. By default, no refinement of the pixel
position is done; to enable this feature you must provide a refinement method and any
associated keywords (currently ``range`` defines the number of pixels on either side
of the estimated location to use for all available refinement options). For example,
to define a line and use ``specreduce`` to update the pixel location the the centroid
of a gaussian fit to the line, you would do the following::

	import astropy.units as u
	from specreduce import CalibrationLine
	test_line = CalibrationLine(calibration_spectrum, 6562.81*u.AA, 500,
							   refinement_method="gaussian",
							   refinement_kwargs={"range": 10})
	refined_line = test_line.with_refined_pixel()

Note that in this example and in the example below, ``calibration_spectrum`` is a
`~specutils.Spectrum1D` spectrum to be used to refine the line locations (for example
a lamp spectrum).

1D Wavelength Calibration
-------------------------

Of course, refining the location of a line is only really useful when done for multiple
lines to be fit as part of the dispersion model. This can be accomplished with the
`~specreduce.wavelength_calibration.WavelengthCalibration1D` class as follows below.
Note that the lines can be input as simple (wavelength, pixel) pairs rather than as
`~specreduce.wavelength_calibration.CalibrationLine` objects - in this case the
``default_refinement_method`` and ``default_refinement_kwargs`` will be applied to
each input line::

	import astropy.units as u
	from specreduce import WavelengthCalibration1D
	test_cal = WavelengthCalibration1D(calibration_spectrum,
									   [(5000*u.AA, 0), (5100*u.AA, 10),
									    (5198*u.AA, 20), (5305*u.AA, 30)],
                                       default_refinement_method="gaussian")
    calibrated_spectrum = test_cal.apply_to_spectrum(science_spectrum)

The example above uses the default model (`~astropy.modeling.functional_models.Linear1D`)
to fit the input spectral lines, and then applies the calculated WCS solution to a second
spectrum (``science_spectrum``). Any other ``astropy`` model can be provided as the
input ``model`` parameter to the `~specreduce.wavelength_calibration.WavelengthCalibration1D`.
In the above example, the model fit and WCS construction is all done as part of the
``apply_to_spectrum()`` call, but you could also access the `~gwcs.wcs.WCS` object itself
by calling::

	test_cal.wcs

The calculated WCS is a cached property that will be cleared if the ``lines``, ``model``,
or ``input_spectrum`` properties are updated, since these will alter the calculated dispersion
fit.