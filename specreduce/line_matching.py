import numpy as np

from specreduce.calibration_data import get_available_line_catalogs


class LineMatch:
    """
    This is the base class for the supported line matching techniques. This is
    effectively what other spectral reduction packages call 'reidentify'. It
    uses a list of pixel positions of lines in a calibration spectrum,
    a catalog of known line wavelengths, a spectral WCS, and a matching
    tolerance in pixels to create matched lists of lines and their pixel/wavelegnth
    positions.
    """
    pass


class AutomaticLineMatch(LineMatch):
    """
    This is a reimplemenation and improvement upon the ``pypeit`` "Holy Grail" technique
    to fully automate the line matching process using no prior information other than
    pixel positions and a catalog of known line wavelengths.
    """
    pass


class TemplateLineMatch(LineMatch):
    """
    This implements using a template calibration spectrum to estimate the initial WCS
    to use in the line matching process. It is based on the ``pypeit`` "reid_arxiv"
    technique and supports using ``pypeit`` templates.
    """
    pass
