
1.10.0 (unreleased)
-------------------

New Features
^^^^^^^^^^^^

- Added ``WavelengthSolution1D.to_asdf()`` and ``WavelengthSolution1D.from_asdf()``
  for serializing a wavelength solution losslessly into an ASDF file. Only the
  coordinate transformation is stored, as a GWCS object whose bounding box carries
  the pixel bounds, so the restored solution reproduces the original exactly. [#317]

- Added a ``WavelengthSolution1D.wcs()`` method that exports the wavelength
  solution as a standard FITS ``astropy.wcs.WCS`` object by fitting the
  WCS Paper III grating dispersion function (``WAVE-GRI`` or ``AWAV-GRA``)
  to the solution. The reference pixel keywords (CRPIX, CRVAL, CDELT) are
  set exactly from the solution model and only the grating PV terms are
  fitted, and a warning is emitted if the approximation deviates from the
  exact solution by more than a given number of pixels. [#NNN]

- The ``wave_air`` flag given to ``WavelengthCalibration1D`` is now stored
  in the resulting ``WavelengthSolution1D``, where it selects between air
  (``AWAV-GRA``) and vacuum (``WAVE-GRI``) spectral axis types in FITS WCS
  export. [#NNN]

1.9.0 (2026-05-06)
------------------

New Features
^^^^^^^^^^^^

- Added new ``specreduce.tilt_correction`` module with a ``TiltCorrection`` class
  for detecting and fitting 2D spectral tilt from arc lamp images. [#303]

- Added new ``specreduce.tilt_solution`` module with a ``TiltSolution`` class
  that stores the fitted tilt model and provides bidirectional coordinate
  transforms between detector and tilt-corrected frames, flux-conserving 2D
  resampling, GWCS export via a ``gwcs`` property, and reconstruction from a
  GWCS object via ``TiltSolution.from_gwcs()``. [#303]

- Added a new specreduce-specific Glossary page to the documentation. [#304]

Other changes
^^^^^^^^^^^^^

- Deprecated the legacy ``docs/terms.rst`` terminology document. It will be
  removed in a future release; users should refer to the new Glossary
  instead. [#304]

1.8.0 (2026-02-24)
------------------

New Features
^^^^^^^^^^^^

- Added uncertainty propagation to ``specreduce.extract.BoxcarExtract`` and
  ``specreduce.extract.HorneExtract``. The extracted spectra have now proper uncertainties.
  [#295, #296]

- Added uncertainty propagation to ``specreduce.background.Background``. The
  ``bkg_image()``, ``bkg_spectrum()``, ``sub_image()``, and ``sub_spectrum()`` methods
  now return spectra with proper uncertainties. When input image has uncertainty, it is
  propagated using variance formulas appropriate for the chosen statistic. When no
  uncertainty is provided, it is estimated from the flux values in the background region. [#297]

- Added optional ``sigma`` parameter to ``specreduce.background.Background`` for sigma
  clipping outlier rejection in background estimation. Default is 5.0; set to ``None``
  to disable. [#297]

API Changes
^^^^^^^^^^^

- Removed ``specreduce.compat`` module and migrated all internal code to use
  ``specutils.Spectrum`` directly. This breaks compatibility with specutils 1.x.
  Users must update to specutils ≥2.0. [#299]

- Bumped minimum dependency versions: specutils ≥2.0, astropy ≥6.0,
  scipy ≥1.14, photutils ≥1.11. These increases are required for
  specutils 2.0 compatibility. [#299]

Other changes
^^^^^^^^^^^^^

- Changed to use ``sphinx_astropy.conf.v2`` and revised the documentation. [#275]

1.7.0 (2025-11-13)
------------------

New Features
^^^^^^^^^^^^

- Added a new ``specreduce.wavecal1d.WavelengthCalibration1D`` class for one-dimensional wavelength
  calibration. The old ``specreduce.wavelength_calibration.WavelengthCalibration1D`` is
  deprecated and will be removed in v. 2.0.
- Added a ``disp_bounds`` argument to ``tracing.FitTrace``. The argument allows for adjusting the
  dispersion-axis window from which the trace peaks are estimated.

1.6.0 (2025-06-18)
------------------

Bug Fixes
^^^^^^^^^
- When all-zero bin encountered in fit_trace with peak_method=gaussian, the bin peak
  will be set to NaN in this case to work better with DogBoxLSQFitter. [#257]
- Reverted the changes to ``background.Background.bgk_spectrum`` introduced in 1.5.0 [#266].

Other changes
^^^^^^^^^^^^^

- Compatibility with specutils 2.0. [#260]
- Set Python 3.11 as the minimum supported Python version and added test support
  for Python 3.13. [#271]
- Changed the ``statistic`` parameter in ``utils.measure_cross_dispersion_profile`` to accept
  either ``median`` or ``average`` instead of ``median`` or ``mean``. [#258]


1.5.1 (2025-03-08)
------------------

Bug Fixes
^^^^^^^^^

- Changed Horne extraction to behave as before when using an interpolated spatial profile
  and not explicitly setting `bkgrd_prof` to `None`. The changed default behavior in 1.5.0
  caused problems in codes using specreduce. [#256]

1.5.0 (2025-03-06)
------------------

New Features
^^^^^^^^^^^^

- Added the ``mask_treatment`` parameter to Background, Trace, and Boxcar Extract
  operations to handle non-finite data and boolean masks. Available options are
  ``apply``, ``ignore``, ``propagate``, ``zero_fill``, ``nan_fill``, ``apply_mask_only``,
  or ``apply_nan_only``. [#216, #254]

- Modified ``background.Background.bgk_spectrum`` to allow the user to select the statistic
  used for background estimation between ``median`` or ``average``. [#253]

- Modified ``extract.BoxcarExtract`` to ignore non-finite pixels when ``mask_treatment`` is set
  to ``apply``; otherwise, non-finite values are propagated. Boxcar extraction is
  now carried out as a weighed sum over the window. When no non-finite values are
  present, the extracted spectra remain unchanged from the previous behaviour.

Bug Fixes
^^^^^^^^^

- Fixed Astropy v7.0 incompatibility bug in ``tracing.FitTrace``: changed to use
  ``astropy.modeling.fitting.DogBoxLSQFitter`` when fitting a Gaussian peak model instead of
  ``astropy.modeling.fitting.LevMarLSQFitter`` that may be deprecated in the future. Also
  changed to use ``fitting.LMLSQFitter`` instead of ``fitting.LevMarLSQFitter`` when fitting
  a generic nonlinear trace model. [#229]

Other changes
^^^^^^^^^^^^^
- Changed ``tracing.FitTrace`` to use ``astropy.modeling.fitting.LinearLSQFitter``
  if the trace model is linear.

1.4.1 (2024-06-20)
------------------

Bug Fixes
^^^^^^^^^
- Fix bug where Background one sided / two sided was not correctly assigning units to data. [#221]


1.4.0 (2024-05-29)
------------------

New Features
^^^^^^^^^^^^

- Added 'interpolated_profile' option for HorneExtract. If The ``interpolated_profile`` option
  is used, the image will be sampled in various wavelength bins (set by
  ``n_bins_interpolated_profile``), averaged in those bins, and samples are then
  interpolated between (linear by default, interpolation degree can be set with
  the ``interp_degree_interpolated_profile`` parameter) to generate a continuously varying
  spatial profile that can be evaluated at any wavelength. [#173]

- Added a function to measure a cross-dispersion profile. A profile can be
  obtained at a single pixel/wavelength, or an average profile can be obtained
  from a range/set of wavelengths. [#214]

API Changes
^^^^^^^^^^^

- Fit residuals exposed for wavelength calibration in ``WavelengthCalibration1D.fit_residuals``. [#446]

Bug Fixes
^^^^^^^^^

- Output 1D spectra from Background no longer include NaNs. Output 1D
  spectra from BoxcarExtract no longer include NaNs when none are present
  in the extraction window. NaNs in the window will still propagate to
  BoxcarExtract's extracted 1D spectrum. [#159]

- Backgrounds using median statistic properly ignore zero-weighted pixels.
  [#159]

- HorneExtract now accepts 'None' as a vaild option for ``bkgrd_prof``. [#171]

- Fix in FitTrace to set fully-masked column bin peaks to NaN. Previously, for
  peak_method='max' these were set to 0.0, and for peak_method='centroid' they
  were set to the number of rows in the image, biasing the final fit to all bin
  peaks. Previously for Gaussian, the entire fit failed. [#205, #206]

- Fixed input of `traces` in `Background`. Added a condition to 'FlatTrace' that
  trace position must be a positive number. [#211]

Other changes
^^^^^^^^^^^^^

- The following packages are now optional dependencies because they are not
  required for core functionality: ``matplotlib``, ``photutils``, ``synphot``.
  To install them anyway, use the ``[all]`` specifier when you install specreduce; e.g.:
  ``pip install specreduce[all]`` [#202]

1.3.0 (2022-12-05)
------------------

New Features
^^^^^^^^^^^^

- The new FitTrace class (see "API Changes" below) introduces the
  ability to take a polynomial trace of an image [#128]

API Changes
^^^^^^^^^^^

- Renamed KosmosTrace as FitTrace, a conglomerate class for traces that
  are fit to images instead of predetermined [#128]

- The default number of bins for FitTrace is now its associated image's
  number of dispersion pixels instead of 20. Its default peak_method is
  now 'max' [#128]

- All operations now accept Spectrum1D and Quantity-type images. All
  accepted image types are now processed internally as Spectrum1D objects
  [#144, #154]

- All operations' ``image`` attributes are now coerced Spectrum1D
  objects [#144, #154]

- HorneExtract can now handle non-flat traces [#148]

Bug Fixes
^^^^^^^^^

- Fixed passing a single ``Trace`` object to ``Background`` [#146]

- Moved away from creating image masks with numpy's ``mask_invalid()``
  function after change to upstream API. This will make specreduce
  be compatible with numpy 1.24 or later. [#155]


1.2.0 (2022-10-04)
------------------

New Features
^^^^^^^^^^^^

- ``Background`` has new methods for exposing the 1D spectrum of the
  background or background-subtracted regions [#143]

Bug Fixes
^^^^^^^^^

- Improved errors/warnings when background region extends beyond bounds
  of image [#127]

- Fixed boxcar weighting bug that often resulted in peak pixels having
  weight above 1 and erroneously triggered overlapping background errors
  [#125]

- Fixed boxcar weighting to handle zero width and edge of image cases
  [#141]


1.1.0 (2022-08-18)
------------------

New Features
^^^^^^^^^^^^

- ``peak_method`` as an optional argument to ``KosmosTrace`` [#115]

API Changes
^^^^^^^^^^^

- ``HorneExtract`` no longer requires ``mask`` and ``unit`` arguments [#105]

- ``BoxcarExtract`` and ``HorneExtract`` now accept parameters (and
  require the image and trace) at initialization, and allow overriding any
  input parameters when calling [#117]

Bug Fixes
^^^^^^^^^

- Corrected the default mask created in
  ``HorneExtract``/``OptimalExtract`` when a user doesn't specify one and
  gives their image as a numpy array [#118]


1.0.0 (2022-03-29)
------------------

New Features
^^^^^^^^^^^^

- Added ``Trace`` classes

- Added basic synthetic data routines

- Added ``BoxcarExtract``

- Added ``HorneExtract``, a.k.a. ``OptimalExtract``

- Added basic ``Background`` subtraction

Bug Fixes
^^^^^^^^^

- Update ``codecov-action`` to ``v2``

- Change default branch from ``master`` to ``main``

- Test fixes; bump CI to python 3.8 and 3.9 and deprecate support for
  3.7
