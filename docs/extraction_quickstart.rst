.. _extraction_quickstart:

Spectral Extraction Quick Start
===============================

Specreduce provides flexible functionality for extracting a 1D spectrum from a 2D spectral image,
including steps for determining the trace of a spectrum, background subtraction, and extraction.


Tracing
-------

The `specreduce.tracing` module defines the trace of a spectrum on the 2D image.  These
traces can either be determined semi-automatically or manually, and are provided as the inputs for
the remaining steps of the extraction process.  Supported trace types include:

* `~specreduce.tracing.ArrayTrace`
* `~specreduce.tracing.FlatTrace`
* `~specreduce.tracing.KosmosTrace`


Each of these trace classes takes the 2D spectral image as input, as well as additional information
needed to define or determine the trace (see the API docs above for required parameters for each
of the available trace classes)::

  trace = specreduce.tracing.FlatTrace(image, 15)


Background
----------

The `specreduce.background` module generates and subtracts a background image from
the input 2D spectral image.  The `~specreduce.background.Background` object is defined by one
or more windows, and can be generated with:

* `~specreduce.background.Background`
* `Background.one_sided <specreduce.background.Background.one_sided>`
* `Background.two_sided <specreduce.background.Background.two_sided>`

The center of the window can either be passed as a float/integer or as a trace::

  bg = specreduce.tracing.Background.one_sided(image, trace, separation=5, width=2)


or, equivalently::

  bg = specreduce.tracing.Background.one_sided(image, 15, separation=5, width=2)


The background image can be accessed via `~specreduce.background.Background.bkg_image` and the 
background-subtracted image via `~specreduce.background.Background.sub_image` (or ``image - bg``).

The background and trace steps can be done iteratively, to refine an automated trace using the 
background-subtracted image as input.

Extraction
----------

The `specreduce.extract` module extracts a 1D spectrum from an input 2D spectrum (likely a
background-extracted spectrum from the previous step) and a defined window, using one of the 
implemented methods:

* `~specreduce.extract.BoxcarExtract`
* `~specreduce.extract.HorneExtract`

Each of these takes the input image and trace as inputs (see the API above for other required
and optional parameters)::

    extract = specreduce.extract.BoxcarExtract(image-bg, trace, width=3)
    spectrum = extract.spectrum

The returned ``extract`` object contains all the set options.  The extracted 1D spectrum can be
accessed via the ``spectrum`` property or by calling the ``extract`` object (which also allows
temporarily overriding any values)::

  spectrum2 = extract(width=6)

Example Workflow
----------------

This will produce a 1D spectrum, with flux in units of the 2D spectrum. The wavelength units will
be pixels. Wavelength and flux calibration steps are not included here.

Putting all these steps together, a simple extraction process might look something like::

    from specreduce.trace import FlatTrace
    from specreduce.background import Background
    from specreduce.extract import BoxcarExtract

    trace = FlatTrace(image, 15)
    bg = Background.two_sided(image, trace, separation=5, width=2)
    extract = BoxcarExtract(image-bg, trace, width=3)
    spectrum = extract.spectrum
