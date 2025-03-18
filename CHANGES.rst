1.5.2 (unreleased)
------------------

New Features
^^^^^^^^^^^^

API Changes
^^^^^^^^^^^

- ``bkg_statistic`` keyword in ``Background.bkg_spectrum()`` now takes callable
  instead of string. Its default is also changed from ``"sum"`` (``np.nansum``)
  to ``np.nanmedian`` to better suit background calculation. [#253]

Bug Fixes
^^^^^^^^^

- When all-zero bin encountered in fit_trace with peak_method=gaussian, the bin peak
  will be set to NaN in this caseto work better with DogBoxLSQFitter. [#257]

Other changes
^^^^^^^^^^^^^

- Compatibility with specutils 2.0. [#260]

1.5.1 (2024-03-08)
------------------

Bug Fixes
^^^^^^^^^

- Changed Horne extraction to behave as before when using an interpolated spatial profile
  and not explicitly setting `bkgrd_prof` to `None`. The changed default behavior in 1.5.0
  caused problems in codes using specreduce.

1.5.0 (2024-03-06)
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

API Changes
^^^^^^^^^^^

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
