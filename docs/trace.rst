Tracing
=======

The `specreduce.tracing` module defines the spatial position (trace) of a spectrum across a 2D
detector image. In spectroscopic data reduction, tracing is a critical step that identifies and maps 
where the spectrum falls on each column and row of the detector. The trace effectively
accounts for any curvature or tilt in how the spectrum is projected onto the detector, which can 
occur due to optical effects in the spectrograph or mechanical flexure during observations.

The trace position can be determined either semi-automatically or manually depending on:

* Data quality and signal-to-noise ratio
* Spectral characteristics and features
* Presence of contaminating sources or artifacts

The `~specreduce.tracing` module provides three main trace types to handle different scenarios:
`~specreduce.tracing.FlatTrace`, `~specreduce.tracing.FitTrace`, and
`~specreduce.tracing.ArrayTrace`. Each trace class requires the 2D spectral image as input, along
with trace-specific parameters that control how the trace is determined and fitted to the data.

Flat trace
----------

`~specreduce.tracing.FlatTrace` assumes that the spectrum follows a straight line across the
detector, and is best for well-aligned spectrographs with minimal optical distortion. To
initialize a `~specreduce.tracing.FlatTrace`, we need to specify the fixed cross-dispersion
pixel position:

.. code-block:: python

    trace = specreduce.tracing.FlatTrace(image, position=6)


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fw = 10
    nd, ncd = 31, 13
    xd, xcd = np.arange(nd), np.arange(ncd)

    spectrum = np.zeros((ncd, nd))
    spectrum[:,:] = norm(6.0, 1.5).pdf(xcd)[:, None]

    plt.rc('font', size=13)
    fig, ax = plt.subplots(figsize=(fw, fw*(ncd/nd)), constrained_layout=True)
    ax.imshow(spectrum, origin='lower')
    ax.plot((0, nd-1), (6, 6), c='k')
    ax.set_xticks(xd+0.5, minor=True)
    ax.set_yticks(xcd+0.5, minor=True)
    ax.grid(alpha=0.25, lw=1, which='minor')
    plt.setp(ax, xlabel='Dispersion axis', ylabel='Cross-dispersion axis')
    fig.show()

FitTrace
--------

`~specreduce.tracing.FitTrace` fits a polynomial function to automatically detected spectrum
positions, and is suitable for typical spectra with smooth, continuous trace profiles. The trace
model can be chosen from `~astropy.modeling.polynomial.Chebyshev1D`,
`~astropy.modeling.polynomial.Legendre1D`, `~astropy.modeling.polynomial.Polynomial1D`,
or `~astropy.modeling.spline.Spline1D`, and the fitting can be optimized by binning the spectrum
along the dispersion axis.

.. code-block:: python

    trace = specreduce.tracing.FitTrace(image, bins=10, trace_model=Polynomial1D(3))

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    plt.rc('font', size=13)

    fw = 10
    nd, ncd = 31, 13
    xd, xcd = np.arange(nd), np.arange(ncd)

    tr = np.poly1d([-0.01, 0.2, 7.0])
    spectrum = np.zeros((ncd, nd))
    for i,x in enumerate(xd):
        spectrum[:,i] = norm(tr(x), 1.0).pdf(xcd)

    fig, ax = plt.subplots(figsize=(fw, fw*(ncd/nd)), constrained_layout=True)
    ax.imshow(spectrum, origin='lower')
    ax.plot(xd, tr(xd), 'k')
    ax.set_xticks(xd+0.5, minor=True)
    ax.set_yticks(xcd+0.5, minor=True)
    ax.grid(alpha=0.25, lw=1, which='minor')
    plt.setp(ax, xlabel='Dispersion axis', ylabel='Cross-dispersion axis')
    fig.show()

ArrayTrace
----------

`~specreduce.tracing.ArrayTrace` uses a pre-defined array of positions for maximum flexibility,
and is ideal for complex or unusual trace shapes that are difficult to model mathematically.
`~specreduce.tracing.ArrayTrace` initialization requires an array of cross-dispersion pixel
positions. The size of the array must match the number of dispersion-axis pixels in the image.

.. code-block:: python

    trace = specreduce.tracing.ArrayTrace(image, positions)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    plt.rc('font', size=13)

    fw = 10
    nd, ncd = 31, 13
    xd, xcd = np.arange(nd), np.arange(ncd)

    tr = np.full_like(xd, 6)
    tr[:6] = 4
    tr[15:23] = 8

    spectrum = np.zeros((ncd, nd))

    for i,x in enumerate(xd):
        spectrum[:,i] = norm(tr[i], 1.0).pdf(xcd)

    plt.rc('font', size=13)
    fig, ax = plt.subplots(figsize=(fw, fw*(ncd/nd)), constrained_layout=True)
    ax.imshow(spectrum, origin='lower')
    ax.plot(xd, tr, 'k')
    ax.set_xticks(xd+0.5, minor=True)
    ax.set_yticks(xcd+0.5, minor=True)
    ax.grid(alpha=0.25, lw=1, which='minor')
    plt.setp(ax, xlabel='Dispersion axis', ylabel='Cross-dispersion axis')
    fig.show()

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