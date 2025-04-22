.. _wavelength_calibration:

Wavelength Calibration
======================

In spectroscopy, the raw data from a detector typically records the intensity of light received
at different detector positions (pixels). However, for scientific analysis, we need to know the
physical wavelength (or frequency, or energy) corresponding to each pixel. **Wavelength
calibration** is the process of determining this relationship, creating a mapping function or
model  that converts pixel coordinates to wavelength values.

This is often achieved by observing a calibration source with well-known emission or absorption
lines at specific wavelengths (e.g., an arc lamp spectrum). By identifying the pixel positions of
these known spectral features, we can fit a mathematical model that describes the dispersion of
the spectrograph.

The ``specreduce`` library provides the `~specreduce.wavecal1d.WavelengthCalibration1D` class
to  facilitate this process for one-dimensional spectra. Tools for calibrating 2D spectra will
be introduced soon.

1D Wavelength Calibration
-------------------------

The `~specreduce.wavecal1d.WavelengthCalibration1D` class encapsulates the data and methods
needed to perform  1D wavelength calibration. The class supports multiple workflows with varying
levels of user interaction. It can be used:

*  manually,
*  as part of an interactive pipeline, or
*  as part of a fully automated pipeline.

The typical workflow involves these steps:

1.  **Initialization**: Create an instance of the class, providing either an observed arc lamp
    spectrum or pre-identified observed line  positions, along with a catalog of known line
    wavelengths. `~specreduce.wavecal1d.WavelengthCalibration1D` supports the use of multiple
    arc spectra, and the initialization can also be done using a list of arc spectra (or a
    list of line arrays identified from multiple arc spectra) and a list of catalogs for each arc
    spectra.
2.  **Line Identification (Optional)**: If an arc spectrum was provided, identify the pixel
    locations of emission lines within it.
3.  **Matching and Fitting**: Determine the correspondence between observed line pixels and
    catalog wavelengths, and fit a model (a polynomial) to represent the
    pixel-to-wavelength transformation. This can be done manually by providing matched pairs or
    automatically using global optimization techniques.
4.  **Inspection**: Evaluate the quality of the fit using residuals and diagnostic plots.
5.  **Applying the Solution**: Use the fitted model (often accessed as a WCS object) to
    calibrate science spectra or resample spectra onto a linear wavelength grid.

These steps are detailed in the following tutorials.

.. toctree::
   :maxdepth: 1

   wavecal1d_example_01.ipynb
   wavecal1d_example_02.ipynb
   wavecal1d_example_03.ipynb

Detailed Steps
--------------

**1. Initialization**

You instantiate the :class:`~specreduce.wavecal1d.WavelengthCalibration1D` by providing basic
information about your setup and data. A reference pixel (``ref_pixel``) is required, which serves
as the anchor point for the polynomial fit.

You must provide *either* a list of arc spectra (or a single arc spectrum) *or* a list of known
observed line positions:

*   **Using an Arc Spectrum**: Provide the arc spectrum as a `specutils.Spectrum`
    object via the ``arc_spectra`` argument. You also need to provide a ``line_lists`` argument,
    which can be a list of known catalog wavelengths (e.g., from an `astropy.table.QTable` or a
    NumPy array with units) or the name(s) of standard line lists recognized by `specreduce` (e.g.,
    ``"ArI"``).

    .. code-block:: python

        import astropy.units as u
        import numpy as np
        from specreduce.compat import Spectrum
        from specreduce.wavecal1d import WavelengthCalibration1D

        # Example arc spectrum (replace with your actual data)
        arc_flux = np.random.rand(1024) * u.DN
        arc_pixels = np.arange(1024) * u.pix # Dummy axis
        arc_spectrum = Spectrum(flux=arc_flux, spectral_axis=arc_pixels)

        # Known ArI line wavelengths
        known_ari_lines = [6965.43, 7067.22, 7383.98, 7503.87, 7635.11] * u.AA

        # Define reference pixel (e.g., center of the detector)
        ref_pix = 512

        ws = WavelengthCalibration1D(ref_pix, arc_spectra=arc_spectrum, line_lists=known_ari_lines)

*   **Using Observed Line Positions**: If you have already identified the pixel centroids of
    lines in your calibration spectrum, you can provide them directly via the ``obs_lines``
    argument (as a list or array). In this case, you *must* also provide the detector's pixel
    boundaries using ``pix_bounds`` (a tuple like ``(min_pixel, max_pixel)``). You still need to
    provide the ``line_lists`` containing the potential matching catalog wavelengths.

    .. code-block:: python

        # Assume observed line pixel centers were found previously
        observed_pixels = np.array([105.3, 210.8, 455.1, 512.5, 680.2])

        # Pixel range of the detector
        pixel_bounds = (0, 1024)

        ws = WavelengthCalibration1D(ref_pix,
                                  obs_lines=observed_pixels,
                                  line_lists=known_ari_lines,
                                  pix_bounds=pixel_bounds)

You can also specify the ``degree`` of the polynomial to be used for the fit (defaults to 3).

**2. Finding Observed Lines**

If you initialized the class with ``arc_spectra``, you need to detect the lines in it. Use the
:meth:`~specreduce.wavecal1d.WavelengthCalibration1D.find_lines` method:

.. code-block:: python

    # Find lines with an estimated FWHM and noise factor
    ws.find_lines(fwhm=3.5, noise_factor=5)

    # Access the found lines (pixel positions)
    observed_lines = ws.observed_lines
    print(observed_lines)

This populates the `~specreduce.wavecal1d.WavelengthCalibration1D.observed_lines` attribute.

**3. Matching and Fitting the Solution**

The core of the process is fitting the model that maps pixels to wavelengths.

*   **Global Fitting (``fit_global``)**: If you have
    `~specreduce.wavecal1d.WavelengthCalibration1D.observed_lines` (either found automatically or
    provided initially) and
    `~specreduce.wavecal1d.WavelengthCalibration1D.catalog_lines` (from ``line_lists``), but don't
    know the exact pixel-wavelength pairs, you can use
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_global`. This method uses a global
    optimization algorithm (differential evolution) to find the best-fit polynomial parameters by
    minimizing the distance between predicted line wavelengths and the nearest catalog lines. You
    need to provide estimated bounds for the wavelength and dispersion at the ``ref_pixel``.

    .. code-block:: python

        # Estimate wavelength and dispersion around the reference pixel
        # (e.g., Wavelength around 7500 AA, Dispersion ~2 AA/pix)
        wavelength_bounds = (7450, 7550)
        dispersion_bounds = (1.8, 2.2)

        ws.fit_global(wavelength_bounds, dispersion_bounds, popsize=30, refine_fit=True)

    Setting ``refine_fit=True`` automatically runs a least-squares refinement after the global
    fit finds an initial solution and matches lines.

*   **Fitting Known Pairs (``fit_lines``)**: If you have already established explicit pairs of
    observed pixel centers and their corresponding known wavelengths, you can use
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.fit_lines` to perform a direct
    least-squares fit.

    .. code-block:: python

        # Assume these are matched pairs
        matched_pixels = np.array([105.3, 512.5, 780.1])
        matched_wavelengths = np.array([6965.43, 7503.87, 7723.76]) * u.AA

        # Create a temporary WS object or manually set attributes if needed
        # (Ensure pix_bounds is set if ws was initialized without arc_spectra)
        # ws.pix_bounds = (0, 1024) # If not set earlier

        ws.fit_lines(pixels=matched_pixels, wavelengths=matched_wavelengths)


After fitting (either way), the pixel-to-wavelength
(`~specreduce.wavecal1d.WavelengthCalibration1D.pix_to_wav`) and wavelength-to-pixel
(`~specreduce.wavecal1d.WavelengthCalibration1D.wav_to_pix`) model transforms are calculated.

**4. Inspecting the Fit**

Several tools help assess the quality of the wavelength solution:

*   **RMS Error**: Calculate the root-mean-square error of the fit in wavelength or pixel units
    using :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.rms`.

    .. code-block:: python

        rms_wave = ws.rms(space='wavelength')
        rms_pix = ws.rms(space='pixel')
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

    .. code-block:: python

        import matplotlib.pyplot as plt

        fig_fit = ws.plot_fit()
        fig_resid = ws.plot_residuals(space='wavelength')
        plt.show()


**5. Using the Solution**

Once satisfied with the fit, you can use the wavelength solution:

*   **Convert Coordinates**: Use :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.pix_to_wav` and
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.wav_to_pix` to convert between pixel and
    wavelength coordinates.

    .. code-block:: python

        pixels = np.array([100, 500, 900])
        wavelengths = ws.pix_to_wav(pixels)
        print(wavelengths)

*   **Get WCS Object**: Access the `gwcs.WCS` object representing the solution via the
    :attr:`~specreduce.wavecal1d.WavelengthCalibration1D.wcs` attribute. This is particularly useful
    for attaching the calibration to a :class:`~specutils.Spectrum` object.

    .. code-block:: python

        # Assuming 'science_spectrum' is a Spectrum1D object
        # science_spectrum.wcs = ws.wcs

*   **Resample Spectrum**: Resample a spectrum (like your science target or the original arc lamp)
    onto a new, potentially linearized, wavelength grid using
    :meth:`~specreduce.wavecal1d.WavelengthCalibration1D.resample`.

    .. code-block:: python

        # Resample the original arc spectrum onto a grid of 1000 points
        resampled_arc = ws.resample(arc_spectrum, nbins=1000)

        # The resampled spectrum now has a linear wavelength axis
        print(resampled_arc.spectral_axis)


See Also
========

For hands-on examples, please refer to the wavelength calibration example notebooks provided with
``specreduce``. For detailed information on each method and its parameters, consult the API
documentation for :class:`~specreduce.wavecal1d.WavelengthCalibration1D`.

.. _wavecal1d_doc: