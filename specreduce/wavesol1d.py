from functools import cached_property
from typing import Callable

import astropy.units as u
import gwcs
import numpy as np
from astropy.modeling import models, Model, CompoundModel
from astropy.nddata import VarianceUncertainty
from gwcs import coordinate_frames
from numpy.ma import MaskedArray
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

from specreduce.compat import Spectrum


__all__ = ["WavelengthSolution1D"]


def _diff_poly1d(m: models.Polynomial1D) -> models.Polynomial1D:
    """Compute the derivative of a Polynomial1D model.

    Computes the derivative of a Polynomial1D model and returns a new Polynomial1D
    model representing the derivative. The coefficients of the input model are
    used to calculate the coefficients of the derivative model. For a Polynomial1D
    of degree n, the derivative is a Polynomial1D of degree n-1.

    Parameters
    ----------
    m
        A Polynomial1D model for which the derivative is to be computed.

    Returns
    -------
    A new Polynomial1D model representing the derivative of the input Polynomial1D model.
    """
    coeffs = {f"c{i-1}": i * getattr(m, f"c{i}").value for i in range(1, m.degree + 1)}
    return models.Polynomial1D(m.degree - 1, **coeffs)


class WavelengthSolution1D:
    def __init__(
        self,
        p2w: None | CompoundModel,
        bounds_pix: tuple[int, int],
        unit: u.Unit,
    ) -> None:
        """Class defining a one-dimensional wavelength solution.

        This class manages the mapping between pixel positions and wavelength values in a 1D spectrum,
        supporting both forward and reverse transformations. It provides methods for resampling
        spectra in the pixel-to-wavelength space while conserving flux, and integrates with GWCS for
        coordinate transformations.

        Initializes an object with pixel-to-wavelength transformation, pixel bounds, and
        measurement unit. Also, converts the unit to its LaTeX string representation.

        Parameters
        ----------
        p2w
            The pixel-to-wavelength transformation model. If None, no transformation
            will be set.
        bounds_pix
            The lower and upper pixel bounds defining the range of the spectrum.
        unit
            The wavelength unit.
        """
        self.unit = unit
        self._unit_str = unit.to_string("latex")
        self.bounds_pix: tuple[int, int] = bounds_pix
        self.bounds_wav: tuple[float, float] | None = None
        self._p2w: None | CompoundModel = None
        self.p2w = p2w

    @property
    def p2w(self) -> None | CompoundModel:
        """Pixel-to-wavelength transformation."""
        return self._p2w

    @p2w.setter
    def p2w(self, m: CompoundModel) -> None:
        self._p2w = m
        self.ref_pixel = m[0].offset.value if m is not None else None

        if "p2w_dldx" in self.__dict__:
            del self.p2w_dldx
        if "w2p" in self.__dict__:
            del self.w2p
        if "gwcs" in self.__dict__:
            del self.gwcs

    @cached_property
    def p2w_dldx(self) -> CompoundModel:
        """Derivative of the pixel-to-wavelength transformation."""
        return models.Shift(self._p2w.offset_0) | _diff_poly1d(self._p2w[1])

    @cached_property
    def w2p(self) -> Callable:
        """Wavelength-to-pixel transformation."""
        p = np.arange(self.bounds_pix[0] - 2, self.bounds_pix[1] + 2)
        self.bounds_wav = self.p2w(self.bounds_pix)
        return interp1d(self.p2w(p), p, bounds_error=False, fill_value=np.nan)

    def pix_to_wav(self, pix: float | ArrayLike) -> float | NDArray | MaskedArray:
        """Map pixel values into wavelength values.

        Parameters
        ----------
        pix
            The pixel value(s) to be transformed into wavelength value(s).

        Returns
        -------
        Transformed wavelength value(s) corresponding to the input pixel value(s).
        """
        if isinstance(pix, MaskedArray):
            wav = self.p2w(pix.data)
            return np.ma.masked_array(wav, mask=pix.mask)
        else:
            return self.p2w(pix)

    def wav_to_pix(self, wav: float | ArrayLike) -> float | NDArray | MaskedArray:
        """Map wavelength values into pixel values.

        Parameters
        ----------
        wav
            The wavelength value(s) to be converted into pixel value(s).

        Returns
        -------
        The corresponding pixel value(s) for the input wavelength(s).
        """
        if isinstance(wav, MaskedArray):
            pix = self.w2p(wav.data)
            return np.ma.masked_array(pix, mask=wav.mask)
        else:
            return self.w2p(wav)

    @cached_property
    def gwcs(self) -> gwcs.wcs.WCS:
        """GWCS object defining the mapping between pixel and spectral coordinate frames."""
        pixel_frame = coordinate_frames.CoordinateFrame(
            1, "SPECTRAL", (0,), axes_names=["x"], unit=[u.pix]
        )
        spectral_frame = coordinate_frames.SpectralFrame(
            axes_names=("wavelength",), unit=[self.unit]
        )
        pipeline = [(pixel_frame, self._p2w), (spectral_frame, None)]
        return gwcs.wcs.WCS(pipeline)

    def resample(
        self,
        spectrum: "Spectrum",
        nbins: int | None = None,
        wlbounds: tuple[float, float] | None = None,
        bin_edges: ArrayLike | None = None,
    ) -> Spectrum:
        """Bin the given pixel-space 1D spectrum to a wavelength space conserving the flux.

        This method bins a pixel-space spectrum to a wavelength space using the computed
        pixel-to-wavelength and wavelength-to-pixel transformations and their derivatives with
        respect to the spectral axis. The binning is exact and conserves the integrated flux.

        Parameters
        ----------
        spectrum
            A Spectrum instance containing the flux to be resampled over the wavelength
            space.

        nbins
            The number of bins for resampling. If not provided, it defaults to the size of the
            input spectrum.

        wlbounds
            A tuple specifying the starting and ending wavelengths for resampling. If not
            provided, the wavelength bounds are inferred from the object's methods and the
            entire flux array is used.

        bin_edges
            Explicit bin edges in the wavelength space. Should be an 1D array-like [e_0, e_1,
            ..., e_n] with n = nbins + 1. The bins are created as [[e_0, e_1], [e_1, e_2], ...,
            [e_n-1, n]]. If provided, ``nbins`` and ``wlbounds`` are ignored.

        Returns
        -------
            1D spectrum binned to the specified wavelength bins.
        """
        if nbins is not None and nbins < 0:
            raise ValueError("Number of bins must be positive.")

        if self._p2w is None:
            raise ValueError("Wavelength solution not set.")

        flux = spectrum.flux.value
        pixels = spectrum.spectral_axis.value

        if spectrum.uncertainty is not None:
            ucty = spectrum.uncertainty.represent_as(VarianceUncertainty).array
            ucty_type = type(spectrum.uncertainty)
        else:
            ucty = np.zeros_like(flux)
            ucty_type = VarianceUncertainty
        npix = flux.size
        nbins = npix if nbins is None else nbins
        if wlbounds is None:
            l1, l2 = self.p2w(pixels[[0, -1]] + np.array([-0.5, 0.5]))
        else:
            l1, l2 = wlbounds

        if bin_edges is not None:
            bin_edges_wav = np.asarray(bin_edges)
            nbins = bin_edges_wav.size - 1
        else:
            bin_edges_wav = np.linspace(l1, l2, num=nbins + 1)

        bin_edges_pix = np.clip(self.w2p(bin_edges_wav) + 0.5, 0, npix - 1e-12)
        bin_edge_ix = np.floor(bin_edges_pix).astype(int)
        bin_edge_w = bin_edges_pix - bin_edge_ix
        bin_centers_wav = 0.5 * (bin_edges_wav[:-1] + bin_edges_wav[1:])
        flux_wl = np.zeros(nbins)
        ucty_wl = np.zeros(nbins)
        weights = np.zeros(npix)

        dldx = np.diff(self.p2w(np.arange(pixels[0], pixels[-1] + 2) - 0.5))

        for i in range(nbins):
            i1, i2 = bin_edge_ix[i : i + 2]
            weights[:] = 0.0
            if i1 != i2:
                weights[i1 + 1 : i2] = 1.0
                weights[i1] = 1 - bin_edge_w[i]
                weights[i2] = bin_edge_w[i + 1]
                sl = slice(i1, i2 + 1)
                w = weights[sl]
                flux_wl[i] = (w * flux[sl] * dldx[sl]).sum()
                ucty_wl[i] = (w**2 * ucty[sl] * dldx[sl]).sum()
            else:
                fracw = bin_edges_pix[i + 1] - bin_edges_pix[i]
                flux_wl[i] = fracw * flux[i1] * dldx[i1]
                ucty_wl[i] = fracw**2 * ucty[i1] * dldx[i1]

        bin_widths_wav = np.diff(bin_edges_wav)
        flux_wl = flux_wl / bin_widths_wav * spectrum.flux.unit / self.unit
        ucty_wl = VarianceUncertainty(ucty_wl / bin_widths_wav**2).represent_as(ucty_type)
        return Spectrum(flux_wl, bin_centers_wav * self.unit, uncertainty=ucty_wl)
