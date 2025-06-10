Tracing
=======

The `specreduce.tracing` module defines the spatial position (trace) of a spectrum across a 2D
detector image. Tracing is a necessary step that identifies where the spectrum falls on each
column and row of the detector, enabling accurate extraction of the 1D spectrum. The trace can be
determined either semi-automatically or manually depending on the data quality and spectral
characteristics.

The module provides three main trace types for different scenarios:

* `~specreduce.tracing.ArrayTrace` - Uses a pre-defined array of positions for maximum flexibility
* `~specreduce.tracing.FlatTrace` - Assumes spectrum follows a straight horizontal/vertical line
* `~specreduce.tracing.FitTrace` - Fits a polynomial function to automatically detected spectrum positions

Each trace class requires the 2D spectral image as input, along with trace-specific parameters.

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


.. note::

    When using `~specreduce.tracing.FitTrace` with noisy or faint spectra:
    
    * Reduce the ``window`` parameter to minimize impact of background noise
    * Lower the polynomial ``order`` to prevent overfitting
    * Consider using `~specreduce.tracing.FlatTrace` for very faint spectra
    * Mask cosmic rays or bad pixels before tracing