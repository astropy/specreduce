Tilt Correction
===============

In astronomical spectroscopy, tilt correction is a calibration step that addresses optical
distortions and misalignments in spectroscopic instruments. These distortions cause wavelength
to vary along the cross-dispersion (spatial) axis, resulting in spectral features appearing
tilted or curved across the detector rather than being perfectly aligned with detector columns.

Tilt correction is performed by modeling a two-dimensional tilt function that describes how
wavelength positions shift across the spatial axis. This function can be determined empirically
from arc lamp calibration spectra by measuring how the centroids of emission lines vary along
the cross-dispersion axis.

Once characterized, the tilt function enables transformation of two-dimensional spectroscopic
images so that wavelengths become aligned along straight lines parallel to the detector axes (a
process known as 2D rectification). This alignment is essential for achieving accurate
wavelength calibration and performing robust sky subtraction.

In the `specreduce` package, the tilt function is represented as a 2D polynomial using an
``~astropy.modeling.models.Polynomial2D`` instance of a specified degree. The
`~specreduce.tilt_correction.TiltCorrection` class implements this correction through several steps:

1. Identifying emission lines in one or more arc lamp calibration spectra for a given number of
   cross-dispersion sample positions
2. Fitting a 2D polynomial model to characterize the geometric distortion
3. Computing a transformation that maps the tilted features to straight lines
4. Applying this transformation to rectify the observed frames


Tutorials
---------

The following tutorial provides hands-on examples demonstrating the usage of the
`~specreduce.tilt_correction.TiltCorrection` class.

.. toctree::
   :maxdepth: 1

   osiris_example.ipynb
