import specutils
from astropy.utils import minversion

__all__ = []

SPECUTILS_LT_2 = not minversion(specutils, "2.0.dev")

if SPECUTILS_LT_2:
    from specutils import Spectrum1D as Spectrum
else:
    from specutils import Spectrum  # noqa: F401
