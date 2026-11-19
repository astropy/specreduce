"""
This is a conceptual set of classes meant to encode the basics of a longslit
workflow as swappable blocks.  Adapted from a whiteboard diagram from
the 2019 Astropy Coordination Meeting (included in the PR adding this file).

Two things that this file is *not* meant to imply:

* The specific use or not of classes as opposed to other approaches
  for defining interfaces.
* Details of how the various blocks connect together.  There are a variety of
  "pipeline" packages in the wider community, and we shouldn't be in the
  business of duplicating that if we can reuse it.  So these blocks just assume
  the simplistic idiom of "call the block, then manually pass it's result to the
  next one", without anything about how you compose blocks together

"""
from dataclasses import dataclass
from typing import Union
from pathlib import Path


@dataclass
class CCDProc:
    """
    Note this class may not need to exist - we could just tell the users "do
    whatever you need to do with ccdproc to get something that's been
    overscan, bias, dark, and pixel-flat subtracted, but that results in a
    """
    ccdproc_subtract_background : bool
    #... other ccdproc options ...


    def __call__(self, rawin: Union([CCDImage,NDData])) -> CCDImage:
        raise NotImplementedError()
        return ccdproc_corrected_image


@dataclass
class CCDProc:
    """
    Note this class may not need to exist - we could just tell the users "do
    whatever you need to do with ccdproc to get something that's been
    overscan, bias, dark, and pixel-flat subtracted, but that results in a
    """
    ccdproc_subtract_background : bool
    #... other ccdproc options ...


    def __call__(self, rawin: Union([CCDImage,NDData])) -> CCDImage:
        raise NotImplementedError()
        return ccdproc_corrected_image

@dataclass
class AutoIdentify2D:
    line_list : dict  # maps a name of a line to a *spectral axis unit* astropy quantity


    def __call__(self, arcimage: CCDImage) -> WavelengthSolution:
        raise NotImplementedError()
        return solution

@dataclass
class AutoIdentify1D:
    template_spectrum : Union([Path, str])  # should be a path to whatever is used as the template,
    #,,, other autoidentify parameters

    def __call__(self, arcimage: CCDImage) -> WavelengthSolution:
        raise NotImplementedError()
        return solution

@dataclass
class InteractiveIdentify1D:
    """
    This is left generally undefined in detail since it requires some form of
    GUI interaction.  In principle it may not even best be thought of as a "call
    and get result"  But the key point is that the user starts with the arc as a
    CCDImage and ends with a wavelength solution.
    """
    def __call__(self, arcimage: CCDImage) -> WavelengthSolution:
        raise NotImplementedError()
        return solution

class WavelengthSolution:
    def __init__(self, tbd):
        raise NotImplementedError()

    def get_wcs(self):
        raise NotImplementedError()

    def apply_wcs(self, spectrum : Union([Spectrum1D, SpectrumCollection])):
        spectrum.wcs = self.get_wcs()


@dataclass
class BoxcarExtract:
    width : float  # pixels
    trace : Union([astropy.modeling.models.Model1D, float])  # model maps x to y, float is syntactic sugar for a constant model
    background_width: Union([float, None]) # None for no background subtraction
    background_fit: astropy.modeling.models.Model1D = astropy.modeling.models.Linear1D # the line-by-line background to subtract

    def __call__(self, spec2d: SpectrumCollection) -> Spectrum1D # note spec2d should already have the wavelength solution in it (see the workflow for how this could be acheived)
        raise NotImplementedError()
        return extracted_spectrum


def basic_workflow():
    rawframe = CCDImage.open(fn)

    ccdproc = CCDProc() # whatever needs to happen here
    processed_sci = ccdproc(rawframe)

    ccdproc = CCDProc() # whatever needs to happen here
    processed_arc = ccdproc(rawdata[idx_arc]

    id2d = AutoIdentify2D(my_line_list_that_i_like)
    wlsoln = id2d(processed_arc)

    wlcal_spec = wlsoln.apply(SpectrumCollection([for row in processed_sci]))

    bex = BoxcarExtract(width=10, trace=123*u.pixel, background_width=5)

    spec1d = bex(wlcal_spec)  # 1d spectrum!
