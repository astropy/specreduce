Tracing
=======

The `specreduce.tracing` module defines the spatial position (trace) of a spectrum across a 2D
detector image. In spectroscopic data reduction, tracing is a critical step that identifies and maps 
where the spectrum falls on each column and row of the detector. This mapping enables accurate 
extraction of the one-dimensional spectrum from the two-dimensional image data. The trace effectively 
accounts for any curvature or tilt in how the spectrum is projected onto the detector, which can 
occur due to optical effects in the spectrograph or mechanical flexure during observations.

The trace position can be determined either semi-automatically or manually depending on factors like:

* Data quality and signal-to-noise ratio
* Spectral characteristics and features
* Presence of contaminating sources or artifacts
* Wavelength coverage and dispersion direction

The module provides three main trace types to handle different observational scenarios:

* `~specreduce.tracing.ArrayTrace` - Uses a pre-defined array of positions for maximum flexibility. 
  Ideal for complex or unusual trace shapes that are difficult to model mathematically.

* `~specreduce.tracing.FlatTrace` - Assumes the spectrum follows a straight horizontal/vertical line
  across the detector. Best for well-aligned spectrographs with minimal optical distortion.

* `~specreduce.tracing.FitTrace` - Fits a polynomial function to automatically detected spectrum 
  positions. Suitable for typical spectra with smooth, continuous trace profiles.

Each trace class requires the 2D spectral image as input, along with trace-specific parameters
that control how the trace is determined and fitted to the data.

Flat trace
----------

.. code-block:: python

    # FlatTrace example - specify the fixed row/column position
    trace = specreduce.tracing.FlatTrace(image, position=15)

ArrayTrace
----------

.. code-block:: python

    # ArrayTrace example - manual position array
    positions = [14, 15, 15, 16]  # pixel positions along dispersion axis
    trace = specreduce.tracing.ArrayTrace(image, positions)

FitTrace
--------

.. code-block:: python

    # FitTrace example - automatic detection with polynomial fitting
    trace = specreduce.tracing.FitTrace(image, 
                                      order=3,          # polynomial order
                                      window=10)        # pixel window for centroid


Best Practices
-------------

When selecting and configuring a trace method, consider these guidelines:

* For bright, well-defined spectra:
    - `FitTrace` with default parameters usually works well
    - Larger ``window`` values can improve centroid accuracy
    - Higher polynomial orders can better follow any curvature

* For noisy or faint spectra:
    - Reduce the ``window`` parameter to minimize impact of background noise
    - Lower the polynomial ``order`` to prevent overfitting
    - Consider using `FlatTrace` for very faint spectra
    - Mask cosmic rays or bad pixels before tracing
    - Pre-process images to improve signal-to-noise if needed

* For unusual or complex traces:
    - Use `ArrayTrace` with manually determined positions
    - Consider breaking the trace into segments
    - Validate trace positions visually before extraction