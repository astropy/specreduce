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
above for required parameters for each of the available trace classes)

.. code-block:: python

  trace = specreduce.tracing.FlatTrace(image, 15)

.. note::
  The fit for `~specreduce.tracing.FitTrace` may be adversely affected by noise where the spectrum
  is faint. Narrowing the window parameter or lowering the order of the fitting function may
  improve the result for noisy data.