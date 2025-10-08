.. _wavelength_calibration:

Wavelength Calibration
======================

In spectroscopy, the raw data from a detector typically records the intensity of light
at different pixel positions. For scientific analysis, we need to know the physical
wavelength (or frequency, or energy) corresponding to each pixel. **Wavelength
calibration** is the process of determining this mapping, creating a model that
converts pixel coordinates to wavelength values.

This is often achieved by observing a calibration source with well-known emission
lines (e.g., an arc lamp). By identifying the pixel positions of these lines, we can fit
a mathematical model describing the spectrograph’s dispersion.

The ``specreduce`` library provides two complementary classes for 1D calibration:

- :class:`~specreduce.wavecal1d.WavelengthCalibration1D`: a high-level calibration workflow
  to detect and match arc lines and fit a dispersion model.
- `~specreduce.wavesol1d.WavelengthSolution1D`: a lightweight wavelength-solution
  container that holds the pixel→wavelength transform with its inverse, a WCS solution, and
  a method for flux-conserving resampling.

Tools for calibrating 2D spectra will be added in later versions.

1D Wavelength Calibration
-------------------------

The :class:`~specreduce.wavecal1d.WavelengthCalibration1D` class encapsulates the data
and methods needed to compute a 1D wavelength solution, and the
:class:`~specreduce.wavesol1d.WavelengthSolution1D` class provides the tools to use the
computed solution. A typical workflow involves:

1. **Initialization**: Create an instance of the class with either arc spectra or
   pre-identified observed line positions, and provide one or more line lists of known
   wavelengths. Multiple arcs and multiple catalogs are supported.
2. **Line Detection (if using arc spectra)**: Identify line positions in the arc spectra.
3. **Fitting the Wavelength Solution**: Use either direct fitting with known pixel–
   wavelength pairs (:meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_lines`)
   or automated global optimization
   (:meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_dispersion`).
4. **Refinement and Inspection**: Optionally refine the solution and check fit quality
   using RMS statistics and plots.
5. **Applying the Solution**: Access the fitted solution as a
   :class:`~specreduce.wavesol1d.WavelengthSolution1D` to resample spectra, or attach the
   solution to a `specutils.Spectrum` as a `~gwcs.wcs.WCS` object.

Quickstart
----------

1. Initialization
*****************

You instantiate the :class:`~specreduce.wavecal1d.WavelengthCalibration1D` by providing basic
information about your setup and data. You must provide *either* a list of arc spectra (or a
single arc spectrum) *or* a list of known observed line positions:

*   **Using an Arc Spectrum**: Provide the arc spectrum as a `specutils.Spectrum`
    object via the ``arc_spectra`` argument. You also need to provide a ``line_lists`` argument,
    which can be a list of known catalog wavelengths or the name(s) of standard line lists
    recognized by `specreduce` (e.g., ``"HeI"``). You can query the available line lists by using
    the `~specreduce.calibration_data.get_available_line_catalogs` function.

    .. code-block:: python

        wc = WavelengthCalibration1D(arc_spectra=arc_he, line_lists=["HeI"])

*   **Using multiple Arc Spectra**:

    .. code-block:: python

        wc = WavelengthCalibration1D(arc_spectra=[arc_he, arc_hg_ar],
                                     line_lists=[["HeI"], ["HgI", "ArI"]])

*   **Using Observed Line Positions**: If you have already identified the pixel centroids of
    lines in your calibration spectrum, you can provide them directly via the ``obs_lines``
    argument (as a list of NumPy arrays). In this case, you *must* also provide the detector's pixel
    boundaries using ``pix_bounds`` (a tuple like ``(min_pixel, max_pixel)``) and a reference
    pixel, ``ref_pixel``,  which serves as the anchor point for the polynomial fit. You also
    need to provide the ``line_lists`` containing the potential matching catalog wavelengths.

    .. code-block:: python

        obs_he = np.array([105.3, 210.8, 455.1, 512.5, 680.2])
        pixel_bounds = (0, 1024)

        wc = WavelengthCalibration1D(ref_pixel=512,
                                     obs_lines=obs_he,
                                     line_lists=["HeI"],
                                     pix_bounds=pixel_bounds)

*   **Using Observed Line Positions From Multiple Arcs**:

    .. code-block:: python

        obs_he = np.array([105.3, 210.8, 455.1, 512.5, 680.2])
        obs_hg_ar = np.array([234.2, 534.1, 768.2, 879.6])
        pixel_bounds = (0, 1024)

        wc = WavelengthCalibration1D(ref_pixel=512,
                                     obs_lines=[obs_he, obs_hg_ar],
                                     line_lists=[["HeI"], ["HgI", "ArI"]],
                                     pix_bounds=pixel_bounds)

2. Finding Observed Lines
*************************

If you initialized the class with ``arc_spectra``, you need to detect the lines in it. Use the
:meth:`~specreduce.wavecal1d.WavelengthCalibration1D.find_lines` method:

.. code-block:: python

    wc.find_lines(fwhm=3.5, noise_factor=5)

This populates the `~specreduce.wavecal1d.WavelengthCalibration1D.observed_line_locations`
attribute.

.. code-block:: python

    print(wc.observed_line_locations)

3. Matching and Fitting the Solution
************************************

The core of the calibration process is fitting a model that maps pixels to wavelengths.

*   **Global Fitting for Automated Pipelines**: If you have
    `~specreduce.wavecal1d.WavelengthCalibration1D.observed_lines` (either found automatically or
    provided initially) and
    `~specreduce.wavecal1d.WavelengthCalibration1D.catalog_lines` (from ``line_lists``), but don't
    know the exact pixel-wavelength pairs, you can use
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_dispersion`.  This method applies the
    `differential evolution optimization algorithm <https://en.wikipedia
    .org/wiki/Differential_evolution>`_
    to find the polynomial coefficients that minimize the distance between observed line
    positions (transformed to wavelength space) and the nearest catalog lines.

    The fit is anchored around the reference pixel (``ref_pixel``), which defines the centre
    of the polynomial model. If not explicitly set at initialization, it defaults to the middle
    of the detector when arc spectra are supplied. You must provide estimated bounds for both
    the wavelength and the dispersion (dλ/dx) at ``ref_pixel``.

    The polynomial degree is set with the ``degree`` argument. A higher degree can capture
    more complex dispersion behaviour but risks overfitting if too few calibration lines are
    available.

    The ``higher_order_limits`` argument optionally constrains the absolute values of the
    higher-order polynomial coefficients, reducing the risk of unrealistic solutions.

    The ``popsize`` argument controls the population size in the differential evolution search.
    Larger values generally improve robustness at the expense of longer computation times.


    .. code-block:: python

        ws = wc.fit_dispersion(wavelength_bounds=(7450, 7550),
                              dispersion_bounds=(1.8, 2.2),
                              higher_order_limits=[0.001, 1e-5],
                              degree=4,
                              popsize=30,
                              refine_fit=True)

    The fitted `~specreduce.wavesol1d.WavelengthSolution1D` is returned by the method and also
    stored in `~specreduce.wavecal1d.WavelengthCalibration1D.solution`. Setting ``refine_fit=True``
    automatically performs a least-squares refinement after the global fit finds an initial
    solution and matches lines.

*   **Fitting Known Pairs for an Interactive Workflow**: If you have already established explicit
    pairs of observed pixel centers and their corresponding known wavelengths, you can use
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_lines` to perform a direct
    least-squares fit.

    .. code-block:: python

        ws = wc.fit_lines(pixels=[105.3, 512.5, 780.1],
                         wavelengths=[6965.43, 7503.87, 7723.76],
                         degree=2,
                         refine_fit=True)

    When ``refine_fit=True`` is set, the method automatically identifies matching pairs between
    observed and catalog lines, then performs a least-squares refinement using **all matching lines**.
    This goes beyond the subset of lines provided to :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_lines`,
    resulting in a more complete wavelength calibration.

After fitting (either way), the pixel-to-wavelength
(`~specreduce.wavecal1d.WavelengthCalibration1D.pix_to_wav`) and wavelength-to-pixel
(`~specreduce.wavecal1d.WavelengthCalibration1D.wav_to_pix`) model transforms are calculated.

4. Inspecting the Fit
*********************

Several tools help assess the quality of the wavelength solution:

*   **RMS Error**: Calculate the root-mean-square error of the fit in wavelength or pixel units
    using :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.rms`.

    .. code-block:: python

        rms_wave = wc.rms(space='wavelength')
        rms_pix = wc.rms(space='pixel')
        print(f"Fit RMS (wavelength): {rms_wave}")
        print(f"Fit RMS (pixel): {rms_pix}")

*   **Plotting**: Visualize the fit and residuals:

    *   :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.plot_fit`: Shows the observed line
        positions mapped to the wavelength axis, overlaid with the catalog lines and the fitted
        solution. Also shows the fit residuals (observed - fitted wavelength) vs. pixel.
    *   :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.plot_residuals`: Plots residuals vs.
        pixel or vs. wavelength.
    *   :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.plot_observed_lines`: Plots the
        identified observed line positions (in pixels or mapped to wavelengths). Can optionally
        overlay the arc spectrum.
    *   :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.plot_catalog_lines`: Plots the catalog
        line positions (in wavelengths or mapped to pixels).

5. Using the Wavelength Solution
********************************

The fitted wavelength solution is stored as a :class:`~specreduce.wavesol1d.WavelengthSolution1D`
instance that you can use to transform and resample spectra. The fitting methods return the
solution, but it is also stored in `~specreduce.wavecal1d.WavelengthCalibration1D.solution`.

*   **Convert Coordinates**: Use :meth:`~specreduce.wavesol1d.WavelengthSolution1D.pix_to_wav` and
    :meth:`~specreduce.wavesol1d.WavelengthSolution1D.wav_to_pix` to convert between pixel and
    wavelength coordinates.

    .. code-block:: python

        pixels = np.array([100, 500, 900])
        wavelengths = ws.pix_to_wav(pixels)
        print(wavelengths)

*   **Get WCS Object**: Access the `~gwcs.wcs.WCS` object representing the solution via the
    :attr:`~specreduce.wavesol1d.WavelengthSolution1D.gwcs` attribute. This is particularly
    useful for attaching the calibration to a :class:`~specutils.Spectrum` object.

*   **Rebin Spectrum**: Resample a spectrum onto a new wavelength grid using
    :meth:`~specreduce.wavesol1d.WavelengthSolution1D.resample`. The rebinning is
    flux-conserving, meaning the integrated flux in the output spectrum matches the integrated flux
    in the input spectrum.

    .. code-block:: python

        resampled_arc = ws.resample(arc_spectrum, nbins=1000)
        print(resampled_arc.spectral_axis)

    By default, :meth:`~specreduce.wavesol1d.WavelengthSolution1D.resample` produces a spectrum on
    a linear wavelength grid. Alternatively, you can pass an arbitrary 1D array of wavelength bin
    edges via ``bin_edges`` to define a custom grid.

    .. code-block:: python

        resampled_arc = ws.resample(arc_spectrum, bin_edges=np.geomspace(4500, 7500, 1000))
        print(resampled_arc.spectral_axis)