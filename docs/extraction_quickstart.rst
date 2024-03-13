.. _extraction_quickstart:

===============================
Spectral Extraction Quick Start
===============================

Specreduce provides flexible functionality for extracting a 1D spectrum from a
2D spectral image, including steps for determining the trace of a spectrum,
background subtraction, and extraction.


Tracing
=======

The `specreduce.tracing` module defines the trace of a spectrum on the 2D image.
These traces can either be determined semi-automatically or manually, and are
provided as the inputs for the remaining steps of the extraction process.
Supported trace types include:

* `~specreduce.tracing.ArrayTrace`
* `~specreduce.tracing.FlatTrace`
* `~specreduce.tracing.FitTrace`


Each of these trace classes takes the 2D spectral image as input, as well as
additional information needed to define or determine the trace (see the API docs
above for required parameters for each of the available trace classes)::

  trace = specreduce.tracing.FlatTrace(image, 15)

.. note::
  The fit for `~specreduce.tracing.FitTrace` may be adversely affected by noise where the spectrum
  is faint. Narrowing the window parameter or lowering the order of the fitting function may
  improve the result for noisy data.


Background
==========

The `specreduce.background` module generates and subtracts a background image from
the input 2D spectral image.  The `~specreduce.background.Background` object is
defined by one or more windows, and can be generated with:

* `~specreduce.background.Background`
* `Background.one_sided <specreduce.background.Background.one_sided>`
* `Background.two_sided <specreduce.background.Background.two_sided>`

The center of the window can either be passed as a float/integer or as a trace::

  bg = specreduce.tracing.Background.one_sided(image, trace, separation=5, width=2)


or, equivalently::

  bg = specreduce.tracing.Background.one_sided(image, 15, separation=5, width=2)


The background image can be accessed via `~specreduce.background.Background.bkg_image`
and the background-subtracted image via `~specreduce.background.Background.sub_image`
(or ``image - bg``).

The background and trace steps can be done iteratively, to refine an automated
trace using the background-subtracted image as input.

Extraction
==========

The `specreduce.extract` module extracts a 1D spectrum from an input 2D spectrum
(likely a background-extracted spectrum from the previous step) and a defined
window, using one of the following implemented methods:

* `~specreduce.extract.BoxcarExtract`
* `~specreduce.extract.HorneExtract`

Each of these takes the input image and trace as inputs (see the API above for
other required and optional parameters)::

  extract = specreduce.extract.BoxcarExtract(image-bg, trace, width=3)

or::

  extract = specreduce.extract.HorneExtract(image-bg, trace)

For the Horne algorithm, the variance array is required. If the input image is
an ``astropy.NDData`` object with ``image.uncertainty`` provided,
then this will be used. Otherwise, the ``variance`` parameter must be set.::

  extract = specreduce.extract.HorneExtract(image-bg, trace, variance=var_array)

An optional mask array for the image may be supplied to HorneExtract as well. 
This follows the same convention and can either be attacted to ``image`` if it
is and ``astropy.NDData`` object, or supplied as a keyword argrument. Note that
any wavelengths columns containing any masked values will be omitted from the
extraction.

The previous examples in this section show how to initialize the BoxcarExtract
or HorneExtract objects with their required parameters. To extract the 1D
spectrum::

  spectrum = extract.spectrum

The ``extract`` object contains all the set options.  The extracted 1D spectrum
can be accessed via the ``spectrum`` property or by calling (e.g ``extract()``)
the ``extract`` object (which also allows temporarily overriding any values)::

  spectrum2 = extract(width=6)

or, for example to override the original ``trace_object``::
  spectrum2 = extract(trace_object=new_trace)

Spatial profile options
-----------------------
The Horne algorithm provides two options for fitting the spatial profile to the
cross_dispersion direction of the source: a Gaussian fit (default),
or an empirical 'interpolated_profile' option.

If the default Gaussian option is used, an optional background model may be
supplied as well (default is a 2D Polynomial) to account
for residual background in the spatial profile. This option is not supported for
``interpolated_profile``.


If  the ``interpolated_profile`` option is used, the image will be sampled in various
wavelength bins (set by ``n_bins_interpolated_profile``), averaged in those bins, and
samples are then interpolated between (linear by default, interpolation degree can
be set with ``interp_degree_interpolated_profile``, which defaults to linear in
x and y) to generate an empirical interpolated spatial profile. Since this option
has two optional parameters to control the fit, the input can either be a string
to indicate that ``interpolated_profile`` should be used for the spatial profile
and to use the defaults for bins and interpolation degree, or to override these
defaults a dictionary can be passed in.

For example, to use the ``interpolated_profile`` option with default bins and
interpolation degree::

  interp_profile_extraction = extract(spatial_profile='interpolated_profile')

Or, to override the default of 10 samples and use 20 samples::

  interp_profile_extraction = extract(spatial_profile={'name': 'interpolated_profile',
                                    'n_bins_interpolated_profile': 20)

Or, to do a cubic interpolation instead of the default linear::

    interp_profile_extraction = extract(spatial_profile={'name': 'interpolated_profile',
                                    'interp_degree_interpolated_profile': 3)

As usual, parameters can either be set when instantiating the HorneExtraxt object,
or supplied/overridden when calling the extraction method on that object.

Example Workflow
================

This will produce a 1D spectrum, with flux in units of the 2D spectrum. The
wavelength units will be pixels. Wavelength and flux calibration steps are not
included here.

Putting all these steps together, a simple extraction process might look
something like::

    from specreduce.tracing import FlatTrace
    from specreduce.background import Background
    from specreduce.extract import BoxcarExtract

    trace = FlatTrace(image, 15)
    bg = Background.two_sided(image, trace, separation=5, width=2)
    extract = BoxcarExtract(image-bg, trace, width=3)
    spectrum = extract.spectrum
