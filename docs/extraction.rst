.. _extraction_quickstart:

Spectrum Extraction
===================

The `specreduce.extract` module extracts a 1D spectrum from an input 2D spectrum
(likely a background-extracted spectrum from the previous step) and a defined
window, using one of the following implemented methods:

* `~specreduce.extract.BoxcarExtract`
* `~specreduce.extract.HorneExtract`

Each of these takes the input image and trace as inputs (see the :ref:`api_index` for
other required and optional parameters)

.. code-block:: python

  extract = specreduce.extract.BoxcarExtract(image-bg, trace, width=3)

or

.. code-block:: python

  extract = specreduce.extract.HorneExtract(image-bg, trace)

For the Horne algorithm, the variance array is required. If the input image is
an `~astropy.nddata.NDData` object with ``image.uncertainty`` provided,
then this will be used. Otherwise, the ``variance`` parameter must be set.

.. code-block:: python

  extract = specreduce.extract.HorneExtract(image-bg, trace, variance=var_array)

An optional mask array for the image may be supplied to HorneExtract as well.
This follows the same convention and can either be attached to ``image`` if it
is an `~astropy.nddata.NDData` object, or supplied as a keyword argument.

The extraction methods automatically detect non-finite pixels in the input
image and combine them with the user-supplied mask to prevent them from biasing the
extraction. In the boxcar extraction, the treatment of these pixels is controlled by
the ``mask_treatment`` option. When set to ``exclude`` (the default), non-finite
pixels within the extraction window are excluded from the extraction, and the extracted
flux is scaled according to the effective number of unmasked pixels. When using other
options (``filter`` or ``omit``), the non-finite values may be propagated or treated
differently as documented in the API.

The previous examples in this section show how to initialize the BoxcarExtract
or HorneExtract objects with their required parameters. To extract the 1D
spectrum

.. code-block:: python

  spectrum = extract.spectrum

The ``extract`` object contains all the set options.  The extracted 1D spectrum
can be accessed via the ``spectrum`` property or by calling (e.g ``extract()``)
the ``extract`` object (which also allows temporarily overriding any values)

.. code-block:: python

  spectrum2 = extract(width=6)

or, for example to override the original ``trace_object``

.. code-block:: python

  spectrum2 = extract(trace_object=new_trace)

Spatial profile options
-----------------------
The Horne algorithm provides two options for fitting the spatial profile to the
cross dispersion direction of the source: a Gaussian fit (default),
or an empirical ``interpolated_profile`` option.

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
interpolation degree

.. code-block:: python

    interp_profile_extraction = extract(spatial_profile='interpolated_profile')

Or, to override the default of 10 samples and use 20 samples

.. code-block:: python

    interp_profile_extraction = extract(
        spatial_profile={
            'name': 'interpolated_profile',
            'n_bins_interpolated_profile': 20
        }
    )

Or, to do a cubic interpolation instead of the default linear

.. code-block:: python

    interp_profile_extraction = extract(
        spatial_profile={
            "name": "interpolated_profile",
            "interp_degree_interpolated_profile": 3,
        }
    )

As usual, parameters can either be set when instantiating the HorneExtraxt object,
or supplied/overridden when calling the extraction method on that object.