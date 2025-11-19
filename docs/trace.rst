.. _tracing:

Tracing
=======

The `specreduce.tracing` module provides three ``Trace`` classes that are used to define the
spatial position (trace) of a spectrum across a 2D detector image: `~specreduce.tracing.FlatTrace`,
`~specreduce.tracing.FitTrace`, and `~specreduce.tracing.ArrayTrace`. Each trace class requires
the 2D spectral image as input, along with trace-specific parameters that control how the trace
is determined.

FlatTrace
---------

`~specreduce.tracing.FlatTrace` assumes that the spectrum follows a straight line across the
detector, and is best for well-aligned spectrographs with minimal optical distortion. To
initialize a `~specreduce.tracing.FlatTrace`, specify the fixed cross-dispersion pixel
position with the ``trace_pos`` argument:

.. code-block:: python

    trace = specreduce.tracing.FlatTrace(image, trace_pos=6)


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

The method works by (optionally) binning the 2D spectrum along the dispersion axis, finding
the PSF peak position along the cross-dispersion for each bin, and then fitting a 1D polynomial to
the cross-dispersion and dispersion axis positions. Binning is optional, and the native image
resolution is used by default. Binning can significantly increase the reliability of the fitting
with low SNR spectra, and always increases the speed.

The peak detection method can be chosen from ``max``, ``centroid``, and ``gaussian``. Of these
methods, ``max`` is the fastest but yields an integer pixel precision.  Both ``centroid`` and
``gaussian`` can be used when sub-pixel precision is required, and ``gaussian``, while being the
slowest method of the three, is the best option if the data is significantly contaminated by
non-finite values.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from numpy.random import seed, normal
    from astropy.modeling.models import Gaussian1D
    from astropy.modeling.fitting import DogBoxLSQFitter
    plt.rc('font', size=13)
    seed(5)

    fw = 10
    nd, ncd = 31, 13
    xd, xcd = np.arange(nd), np.arange(ncd)

    psf = norm(5.4, 1.5).pdf(xcd) + normal(0, 0.01, ncd)
    fitter = DogBoxLSQFitter()
    m = fitter(Gaussian1D(), xcd, psf)

    fig, ax = plt.subplots(figsize=(fw, fw*(ncd/nd)), constrained_layout=True)
    ax.step(xcd, psf, where='mid', c='k')
    ax.axvline(xcd[np.argmax(psf)], label='max')
    ax.axvline(np.average(xcd, weights=psf), ls='--', label='centroid')
    ax.axvline(m.mean.value, ls=':', label='gaussian')
    ax.plot(xcd, m(xcd), ls=':')
    ax.legend()
    plt.setp(ax, yticks=[], ylabel='Flux', xlabel='Cross-dispersion axis [pix]', xlim=(0, ncd-1))
    fig.show()

ArrayTrace
----------

`~specreduce.tracing.ArrayTrace` uses a pre-defined array of positions for maximum flexibility
and is ideal for complex or unusual trace shapes that are difficult to model mathematically.
To initialize `~specreduce.tracing.ArrayTrace`, provide a 1D array of cross-dispersion pixel
positions via the ``trace`` argument. The size of this array must match the number of
pixels along the dispersion axis of the image.

.. code-block:: python

    trace = specreduce.tracing.ArrayTrace(image, trace=positions)

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
