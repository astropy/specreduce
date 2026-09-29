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

For 2D spectra, where the spectral lines are tilted or curved across the detector,
`~specreduce.wavesol2d.WavelengthSolution2D` combines a 1D wavelength solution with a
:ref:`tilt solution <tilt_correction>` into a solution that maps every detector pixel to a
wavelength (see :ref:`wavelength_calibration_2d`).

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
    stored in ``WavelengthCalibration1D.solution``. Setting ``refine_fit=True``
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
    resulting in an improved wavelength solution.

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

The fitted polynomial wavelength solution is stored as a
:class:`~specreduce.wavesol1d.WavelengthSolution1D` instance that you can use to transform and
resample spectra. The fitting methods return the solution, but it is also stored in the
``solution`` attribute of :class:`~specreduce.wavecal1d.WavelengthCalibration1D`.

*   **Convert Coordinates**: Use :meth:`~specreduce.wavesol1d.WavelengthSolution1D.pix_to_wav` and
    :meth:`~specreduce.wavesol1d.WavelengthSolution1D.wav_to_pix` to convert between pixel and
    wavelength coordinates.

    .. code-block:: python

        pixels = np.array([100, 500, 900])
        wavelengths = ws.pix_to_wav(pixels)
        print(wavelengths)

*   **Export a FITS WCS**: Use :meth:`~specreduce.wavesol1d.WavelengthSolution1D.wcs` to get a
    standard `~astropy.wcs.WCS` object that approximates the polynomial solution with the FITS WCS
    Paper III grism dispersion function. The returned WCS is a numerical fit to the
    polynomial solution and can be serialized into a FITS header readable by any FITS-compliant
    software, or attached to a :class:`~specutils.Spectrum` object. A warning
    is emitted if the fit deviates from the polynomial solution by more than ``max_residual``
    pixels.

    .. code-block:: python

        header = ws.wcs().to_header()
        print(header)

*   **Export a GWCS Object**: Access the `~gwcs.wcs.WCS` object representing the polynomial
    wavelength solution via the :attr:`~specreduce.wavesol1d.WavelengthSolution1D.gwcs` attribute.
    This is particularly useful for attaching the calibration to a :class:`~specutils.Spectrum`
    object. Unlike `~specreduce.wavesol1d.WavelengthSolution1D.wcs`, which returns a grism
    dispersion function fitted to the polynomial solution, the GWCS object carries the
    exact polynomial model.

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

.. _wavelength_calibration_2d:

2D Wavelength Calibration
-------------------------

In a 2D spectral image, the optical distortions that tilt and curve the spectral lines make the
wavelength depend on both detector coordinates, so a single 1D solution is exact only along one
row. The :class:`~specreduce.wavesol2d.WavelengthSolution2D` class removes this limitation by
chaining two solutions:

- a :class:`~specreduce.tilt_solution.TiltSolution` that maps the detector to a rectified
  space where the wavelength is constant along the cross-dispersion axis (see
  :ref:`tilt_correction`), and
- a :class:`~specreduce.wavesol1d.WavelengthSolution1D` that maps the rectified
  dispersion-axis coordinate to a wavelength.

The combined solution maps any detector pixel ``(disp, cdisp)`` directly to a wavelength. A
typical workflow involves:

1. **Tilt Correction**: Fit the tilt of the arc lines with
   :class:`~specreduce.tilt_correction.TiltCorrection`.
2. **1D Calibration**: Calibrate an arc spectrum in the rectified space with
   :class:`~specreduce.wavecal1d.WavelengthCalibration1D`.
3. **Combining the Solutions**: Create a :class:`~specreduce.wavesol2d.WavelengthSolution2D`
   from the tilt and 1D solutions.
4. **Applying the Solution**: Convert coordinates, resample 2D images onto a wavelength grid,
   and save the solution for later use.

1. Fitting the Tilt Solution
****************************

Fit the tilt solution from the arc frames as described in :ref:`tilt_correction`. The
reference row set by ``cdisp_ref_pixel`` (or derived from a trace) is where the rectified and
detector coordinates coincide.

.. code-block:: python

    from specreduce.tilt_correction import TiltCorrection

    tc = TiltCorrection(arc_frames=arc, cdisp_ref_pixel=150)
    tc.find_arc_lines(fwhm=3.5)
    tc.fit(degree=3)
    ts = tc.solution

2. Calibrating the Rectified Arc
********************************

The 1D wavelength solution must be defined in the rectified space. Along the reference row the
rectified and detector coordinates coincide, so the simplest option is to calibrate the arc
spectrum of the reference row. A higher signal-to-noise arc spectrum is obtained by
tilt-correcting the arc frame and summing several rows around the reference row: the lines
are straight in the rectified image, so summing does not broaden them.

.. code-block:: python

    import numpy as np
    from specutils import Spectrum
    from specreduce.wavecal1d import WavelengthCalibration1D

    arc_rect = ts.resample(arc)
    flux = arc_rect.data[140:161].sum(axis=0)
    arc_1d = Spectrum((flux - np.median(flux)) * arc_rect.unit)

    wc = WavelengthCalibration1D(arc_spectra=arc_1d, line_lists=[["HeI", "NeI"]])
    wc.find_lines(fwhm=3.5)
    ws = wc.fit_dispersion(wavelength_bounds=(5600, 5900),
                           dispersion_bounds=(2.3, 2.7),
                           degree=3)

The example assumes the dispersion axis runs along the image columns (``disp_axis=1``), and
subtracts the median as a simple background estimate because the line detection expects a
background-subtracted arc spectrum.

.. note::

    Resample the arc with the default grid of
    :meth:`~specreduce.tilt_solution.TiltSolution.resample`, which has one bin per detector
    pixel starting from zero. The pixel coordinates of the 1D solution are then the rectified
    coordinates. A custom ``nbins``, ``bounds``, or ``bin_edges`` grid would scale or shift
    the pixel axis of the 1D solution relative to the rectified space.

3. Combining the Solutions
**************************

Create the 2D solution from the tilt and 1D solutions:

.. code-block:: python

    from specreduce.wavesol2d import WavelengthSolution2D

    ws2d = WavelengthSolution2D(ts, ws)

The detector image shape is taken from the tilt solution. If the tilt solution does not know
it, pass it with the ``image_shape`` argument, since the inverse tilt transformation is fitted
numerically over the image. The components can also be given as the GWCS objects exported by
the :attr:`~specreduce.tilt_solution.TiltSolution.gwcs` and
:attr:`~specreduce.wavesol1d.WavelengthSolution1D.gwcs` properties:

.. code-block:: python

    ws2d = WavelengthSolution2D(ts.gwcs, ws.gwcs, image_shape=(ny, nx))

A bare GWCS object cannot carry the air-wavelength flag, so a solution built from one gives
vacuum wavelengths. The component solutions stay accessible through the
:attr:`~specreduce.wavesol2d.WavelengthSolution2D.tilt` and
:attr:`~specreduce.wavesol2d.WavelengthSolution2D.wavelength` properties, and the wavelength
unit and air-wavelength flag follow the 1D solution.

4. Using the 2D Solution
************************

*   **Convert Coordinates**: Use :meth:`~specreduce.wavesol2d.WavelengthSolution2D.pix_to_wav`
    to map detector pixels to wavelengths, and
    :meth:`~specreduce.wavesol2d.WavelengthSolution2D.wav_to_pix` to find the detector
    dispersion-axis coordinate of a wavelength at given cross-dispersion coordinates. The
    inverse mapping returns ``nan`` for wavelengths outside the solution.

    .. code-block:: python

        wav = ws2d.pix_to_wav(disp=np.array([100, 500, 900]), cdisp=np.array([20, 150, 280]))
        disp = ws2d.wav_to_pix(5800, cdisp=np.array([20, 150, 280]))

*   **Underlying Model**: The detector-pixel-to-wavelength transformation is available as a
    single Astropy compound model with two inputs and one output via the
    :attr:`~specreduce.wavesol2d.WavelengthSolution2D.p2w` property.

*   **Export a GWCS Object**: The :attr:`~specreduce.wavesol2d.WavelengthSolution2D.gwcs`
    property gives a `~gwcs.wcs.WCS` object mapping the 2D detector pixel frame to a 1D spectral
    frame.

    .. code-block:: python

        wav = ws2d.gwcs(500, 150)

*   **Rebin a 2D Image**: Use :meth:`~specreduce.wavesol2d.WavelengthSolution2D.resample` to
    bin a 2D detector image onto a common wavelength grid. Each cross-dispersion row is binned
    independently, directly from the detector pixels to the wavelength bins, so the flux is
    redistributed only once instead of in separate tilt-correction and wavelength-resampling
    steps. The result is a 2D `~specutils.Spectrum` in units of flux per wavelength unit, with
    the propagated uncertainty and a mask that flags every bin receiving flux from a masked
    pixel. The binning conserves the integrated flux of each row as long as the wavelength
    grid covers the whole detector.

    .. code-block:: python

        science_wl = ws2d.resample(science_frame)

    By default, the wavelength grid has one bin per detector column and spans the wavelengths
    covered by the reference row. You can control the grid with the ``nbins``, ``wlbounds``,
    and ``bin_edges`` arguments:

    .. code-block:: python

        # Custom number of bins and wavelength bounds
        science_wl = ws2d.resample(science_frame, nbins=800, wlbounds=(4600, 6900))

        # Explicit bin edges
        science_wl = ws2d.resample(science_frame, bin_edges=np.geomspace(4600, 6900, 801))

    The ``resample`` method accepts the same ``mask_treatment`` options as the other
    ``specreduce`` operations. Because the cross-dispersion axis is unchanged, the resampled
    image can be traced and extracted like any other 2D spectrum, and the extracted 1D
    spectrum inherits the wavelength grid:

    .. code-block:: python

        from specreduce.extract import BoxcarExtract
        from specreduce.tracing import FlatTrace

        trace = FlatTrace(science_wl, 150)
        spectrum = BoxcarExtract(science_wl, trace, width=10).spectrum

*   **Save and Load the Solution**: Use
    :meth:`~specreduce.wavesol2d.WavelengthSolution2D.to_asdf` and
    :meth:`~specreduce.wavesol2d.WavelengthSolution2D.from_asdf` to store the solution in an
    ASDF file and read it back. All transformations are stored as analytic models, so the
    restored solution is identical to the original. The file also holds the chained
    detector-pixel-to-wavelength GWCS object, so any ASDF-aware tool can evaluate the solution
    without ``specreduce``.

    .. code-block:: python

        ws2d.to_asdf("wavesol2d.asdf")
        ws2d = WavelengthSolution2D.from_asdf("wavesol2d.asdf")
