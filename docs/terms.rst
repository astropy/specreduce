

Terminology
-----------

As part of a workshop at NOIRLab, 13-16 Nov 2024, the attendees
extensively discussed terminology in an effort to generate a common
understanding of, and the nuances in, many of the terms bandied about
when discussing spectroscopy data and its reduction and analysis.  The
following is a living document that stemmed from that original
discussion.

.. note::

    Below, relative terms that can have multiple meanings are
    highlighted in italics.


2D Spectrum/Image
=================

- Image = 2D Image
- Optical/IR, different for other wavelengths
- Something that is not necessarily pure “raw” data, but data that
  exists prior to extraction
- Refers to e.g. the 2D CCD images (optionally pre-processed) from which
  1D spectra (flux vs. wavelength) are extracted

.. KBW: I missed this discussion, so I don't know how we want to capture
   "Alt" definitions here...  Seems a bit awkward.

Alt Spectrum 2D
===============

- Spectrum as a function of slit position?
- Fiber dispersed
- Lots of alt Spectra 2D
- Multiple spectral orders?
- Can get up to something like 5D for solar data: wavelength x space x
  space x time x polarization
- Is there an IVOA definition that differs from both of the above?

1D Spectrum
===========

- Flux versus spectral axis (wavelength/frequency). Could be calibrated but not necessarily.

*Preprocessing*
===============

- (As related to spectral reduction): Things like flat-fielding and bias
  subtraction that are done on a full CCD image before extractions are
  performed.
- Alt term: instrument signature removal
- Meaning of this depends on context (“What do others do before I start?”)
- Typically done on raw/2d image
- Work done before mine.

*Post-processing*
=================

- Typically done on the extracted spectra?

Extraction
==========

- The process of converting raw spectrum data on 2D image into flux versus
  spectral axis or pixel (i.e. Spectrum1D), not necessarily flux or spectral
  calibration.

Rectified ND spectrum
=====================

- Non-dispersion axes means something like ra/dec, or polarization? 
- Nearly always (maybe always?) Implies some amount of resampling 
- There is one dimension that is ENTIRELY a spectral axis, all others are not
  specified by calling it a “Rectified ND Spectrum”, although they often do have
  some kind of meaning

Calibrated 2D image
===================

- Not-resampled. 
- Calibrated from raw pixel to wavelength.

.. KBW: Not really sure this is terminology, but I've left it as written in the
   workshop notes.

Facilitating Data sharing, cross-matching, etc., standards via IVOA standards
=============================================================================
- Data models (but see below)
- COM
- ObsCore
- SSA
- Not widely adopted, but could be!

*Workflow*
==========

- More holistic than a pipeline. Often a  superset of pipeline with additional
  steps to facilitate data ingest, job orchestration, data
  collection/coordination, data archiving, etc.  
- Also analysis.

*Pipeline*
==========

- Organized code for **automatically** processing raw data into calibrated spectra and 
    - Optionally derived quantities like redshifts, line fits, ...
- More generically, the linking of several steps/jobs/codes/methods where
  outputs of one feed as inputs to another.
- “Spectral reduction pipeline”? vs. “analysis pipeline” vs. “XYZ pipeline”
- SDSS and JWST pipelines are different, Dragons has three pipelines. DESI has
  different pipelines depending on who you talk to.

Classification
==============
- Identifying the type of object a spectrum represents: star, galaxy, QSO, ...
- Result of Model fitting (auto or by-eye).

*Redshifting*
=============

- Determining the redshift of a spectrum, which may or may not be independent of
  classification.
- Sometimes also means “changing the redshift of the object to its redshift”,
  e.g. converting the “wavelength/frequency” axis to rest-frame
- Related issues on GitHub:
    - `#258 <https://github.com/astropy/specutils/issues/258>`__
    - `#455 <https://github.com/astropy/specutils/issues/455>`__
    - `#820 <https://github.com/astropy/specutils/issues/820>`__

Heliocentric / barycentric correct
==================================

- Converting a spectrum to some rest frame in order to measure a radial velocity
  for a nearby stellar source.
- Heliocentric is “the frame where the sun is at rest”
- Barycentric is “the frame where the barycenter of the solar system is at rest”
- (Some of this is very very precisely defined at the GR level)

*Archive*
=========

- A physical or virtual location from which processed data can be accessed. This
  could include both PI/collaboration access and public access

    - Or unprocessed.
    - And metadata needed to reduce raw data.

- Raw data bundle (science+all cals needed to reduce it) 
- Ideally would Supporting FAIR principles (Findable, Accessible, Interoperable,
  Reusable)
- Noun or verb.

Data Assembly
=============

- A bundle of data prepared for collaboration access (to write papers, etc.)
  that will eventually become a data release.
- Used internally by DESI, but deprecated.
- It becomes the data release at the DR date
- Sloan synonym: “Internal Product Launch”

Data release
============

- A bundle of data specifically intended to be public
- Can be raw or not raw
- Somehow “pinned” data raw/reduced/analyzed with a particular version of
  pipelines.
- Aspires to be frozen.
- Can be either a noun or a verb

Open Development 
================

- Developing software in a way that the community can see both how it has been
  developed and why it was developed that way. 
- Usually, but not absolutely necessarily, implies the community is also free to
  contribute.
- Not necessarily open source.  
- Repos are publicly visible, including issue tracker.

Flux calibration
================

- Converting a 1D/2D spectrum from “counts” to astrophysical units of flux
  density

*Telluric Correction*
=====================

- Removing the telluric (atmospheric) absorption bands from spectra
- Removing the *multiplicative* component of the sky - absorption
- But there was some disagreement over whether this includes sky

*Sky subtraction*
=================

- Removing the *additive* component of the sky/emission so all “photons” come
  from the source
- But there was some disagreement over whether this is overlapping with a
  Telluric correction

IFU (Integral-Field Unit)
=========================

- Covers a “contiguous” 2d field on the sky  with spatial information along both
  axes
- Fibers or similar are tightly bundled and contiguously cover a region on the
  sky. Or an image slicer. Or a microlens array.
- May or may not be multi-object.
- IFS (Integral field Spectrograph) and IFU are sometimes distinguished where
  IFS is the whole instrument but IFU is the head-unit that does the IF part

MOS (Multi-Object Spectroscopy)
===============================

- Could be a fiber or a slit 
- Multiple objects observed in the same exposure

*Flux*
======

- Energy per time per area
- Also used as a shorthand for “the not spectral unit part of a 1D spectrum”
  (would that be the “dependent variable”?)
- Oftentimes used to mean “flux density”
- `Spectrum1D
  <https://specutils.readthedocs.io/en/stable/api/specutils.Spectrum1D.html#specutils.Spectrum1D>`__
  uses the attribute 'flux'. Should this be renamed to 'flux_density'?

    - The intent in specutils was to not agonize over this but just accept that
      it's a shorthand astronomers use, and there wasn't a better name (“y”,
      “data”, etc)

Flux Density
============

- Flux per unit wavelength/energy/wavenumber, usually(?) in astrophysical units,
  e.g., W/m^2/nm

*Row-stacked spectra*
=====================

- Collection of 1D spectra in a 2D array (image?), one spectrum per row.
- Shared spectral axis.
- This is the format of specutils.Spectrum1D when it's a “vector” spectrum1D

*Data cube*
===========

- Spectral 3D matrix with 2 spatial dimensions and one spectral one. Product of
  IFU data with contiguous sky coverage

    - Doesn't even have to be spectral, although in the spectral context it
      usually is

- Not always 3D (data hypercuboid??).
- “Multi dimensional data blob”

*Spectral data cube*
====================

- At least one axis is a spectral axis but who knows about the rest!
- Hypercube.

*[Spectral] Data format*
========================

- “Format” can mean data structure (i.e., in-memory, possibly bound to a
  particular language, though it doesn't have to be - see Apache Arrow)
- “Format” can also mean a file format
- “Format” can also be something even more technical like “how the bytes in a
  struct are packed“

Data Structures
===============

- Python Data structures, which are Python classes.

    - NDData/NDCube/SpectrumCollection, Spectrum1D etc.  
    - CCDData. Subclass of NDData
    - AstroData - from DRAGONS (collection of NDData-like objects, mapped to a
      file, plus metadata abstraction etc.)
    - Lots of classes to represent spectra
    - Link to issue about renaming Spectrum1D class in specutils.
    - arrays

*Data Model*
============

- In the SDSS/DESI sphere, this has a meaning that is known to differ from other
  uses of the term. In SDSS/DESI this means a documentation product that
  describes all of the files in a data release, both file formats and how they
  are organized into a hierarchy of directories on disk. For example,
  see the `desidatamodel <https://desidatamodel.readthedocs.io/en/stable/>`__.

- IVOA data model is a formalized thing that follows a specific XML schema

    - Data model is abstract, implementation could potentially be different.
    - In the IVOA it is not yet allowed to be anything other than XML although
      there's a lot of interest in changing that

- Which is different from SQL data models

    - The word 'schema' is sometimes used here, but that is also ambiguous even
      within SQL flavors itself.

*Spectroscopic search - Data discovery*
=======================================

- Search for spectra from any/particular instrument based on position or other
  known properties of the sources. If available, all the spectra will be listed.
- Example tool to do this: SPARCL (How-To Jupyter notebook available here)

SSA = Simple Spectral Access [VO protocol]
==========================================

- "Uniform interface to remotely discover and access one-dimensional spectra."
  See `here <https://www.ivoa.net/documents/SSA/#:~:text=The%20Simple%20Spectral%20Access%20(SSA,(DAL)%20of%20the%20IVOA>`__.
- Not commonly (used in the US?). (Example of use Data Central)

Reduction (of spectroscopic data)
=================================

- Getting data from raw-off-the-instrument (or nearly so) to the point where
  analysis can be done.
- The process of turning 2D spectral images to 1D spectra. Can be wavelength
  calibrated, sky subtracted, flux calibrated, but intermediate products are
  "reduced" compared to earlier steps of the process.
- MAYBE: Can potentially be done automatically without a human-in-the-loop?
- Required products vs optional products
- Removing instrument signature.
- "Reduction" is in the sense of reducing complexity, but it is often an
  inflation of bytes (in radio it is a literal reduction, in optical usually
  not)
- Astronomy specific word.
- Spectroscopic reduction: the process of going from raw data to science-ready
  spectra

Analysis (of spectroscopic data) 
================================

- Taking scientific measurements or achieving scientific results from
  already-reduced spectroscopic data
- Analysis does not depend on the instrument.Rem
- MAYBE: cannot be done automatically, requires a human to make some sort of
  judgment
- Optional.
- Spectroscopic analysis: the process of going from science-ready spectra to
  science

Sky
===

- Model or observed sky background (really a foreground!) which was usually
  subtracted from the observed spectrum

*Stacking*
==========

- Combination of multiple spectra in a prescribed way to increase quality (e.g.,
  S/N)
- in some contexts like numpy arrays and astropy.table.vstack, it can refer to
  combining multiple objects / tables into a single object without coadding data.

*Coadding*
==========

- combining multiple spectra of the same object into a single spectrum, e.g. to
  improve signal-to-noise or combine spectra across multiple wavelength ranges.
- Alternative term for “stacking” to disambiguate meanings

Spectral fitting
================

- Modeling an observed spectrum with templates (possibly physically motivated)
  and/or mathematical functions 

Digital Twins
=============

- Realistic fake data that potentially adapts to new states of the system over
  time.

Trace fitting
=============

- on a 2D CCD image from a multi-object spectrometer, typically wavelengths span
  in 1 direction and fibers/objects in the other direction.  “trace fitting” is
  mapping the y vs. x of where the spectra actually go on the CCD image.

- Different for slit-based and fiber-based spectrographs

    - slit-based: trace the *edges* of the slit spectrum along the spectral
      direction
    - fiber-based: trace the *center* of the fiber spectrum spatial point-spread
      function

- Spectral tracing

Wavelength calibration
======================

- calibrating what true wavelength is represented by the observed photons on a
  detector, e.g. what wavelength is row y of a detector?
- Possibly a 2d process
- The process of adding (the spectral part of) a WCS

Visual Inspection (of spectra)
==============================

- Humans looking at spectra and making decisions about what the “truth” is.
- Can include identifying the presence/absence of features (qualitative) or
  assessing a quantitative fit (e.g., best-fit redshift value)

*Spectral resolution*
=====================

- Changes in spectral dispersion power as a function of wavelength due to
  instrument 

- Resolving power vs Resolution/Dispersion

    - Resolving power is the ability to distinguish close features
    - Dispersion is the change in wavelength/energy per pixel

API
===

- Application Programming Interface

    - What makes a good API for spectroscopic software?
    - What is needed for different aspects of spectroscopic software (e.g.,
      reduction vs. archive access)?

*Spectral class*
================

- E.g., Spectrum1D
- In SDSS, 'class' is short for 'classification'.
- DESI uses SPECTYPE for spectral type (QSO, GALAXY, STAR)

Package
=======

- A software tool or collection of tools developed in the same “space”
- Has a specific meaning in a Python context that’s more specific, but can be
  used more generally for multiple languages

Spectral data visualization
===========================

- Tools and procedures to display reduced spectral data

Spectral decomposition 
======================

- a form of spectral fitting that identifies separate components that appear in
  a spectrum; e.g, quasar + galaxy.
- Goes with spectral fitting.

Processing Steps
================

- For DESI (largely inherited from SDSS usage):

    - pre-processing (of CCD images, bias, dark, pixel-flat fielding)
    - extraction (getting counts vs. wavelength from 2D images)
    - sky subtraction (subtracting the additive non-signal sky component)
    - flux calibration (includes both instrument throughput and telluric
      absorption multiplicative corrections)
    - classification and redshift fitting (is it a galaxy, star, or quasar; at
      what redshift?)



Mentioned but not defined
=========================

- WCS & Database archive
- Cloud archiving
- Modular functions which can be used by other pipelines.
- Interactive Dashboard 

