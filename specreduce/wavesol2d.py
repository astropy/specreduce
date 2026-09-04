from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Callable

import asdf
import astropy.units as u
import gwcs
import numpy as np
from astropy.modeling import CompoundModel
from astropy.nddata import VarianceUncertainty
from gwcs import coordinate_frames
from numpy.ma import MaskedArray
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from specutils import Spectrum

from specreduce.core import parse_image, MaskingOption
from specreduce.tilt_solution import TiltSolution
from specreduce.wavesol1d import WavelengthSolution1D

__all__ = ["WavelengthSolution2D"]


class WavelengthSolution2D:
    def __init__(
        self,
        tilt: TiltSolution | gwcs.wcs.WCS,
        wavelength: WavelengthSolution1D | gwcs.wcs.WCS,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        """Class defining a two-dimensional wavelength solution.

        Chains a tilt solution, which maps detector pixels to a rectified space where the
        wavelength is constant along the cross-dispersion axis, with a one-dimensional
        wavelength solution defined in that rectified space. The result maps any detector
        pixel ``(disp, cdisp)`` directly to a wavelength, supports the inverse mapping from a
        wavelength and a cross-dispersion coordinate back to a detector column, and can
        resample a 2D detector image onto a common wavelength grid while conserving flux.

        Parameters
        ----------
        tilt
            The tilt solution, given either as a `~specreduce.tilt_solution.TiltSolution` or as
            the GWCS object exported by its ``gwcs`` property. The cross-dispersion coordinate
            is assumed to pass through the tilt transformation unchanged.
        wavelength
            The one-dimensional wavelength solution in the rectified space, given either as a
            `~specreduce.wavesol1d.WavelengthSolution1D` or as the GWCS object exported by its
            ``gwcs`` property. A bare GWCS cannot carry the air-wavelength flag, so a solution
            built from one is taken to give vacuum wavelengths.
        image_shape
            The detector image shape as ``(ny, nx)``. Required when the tilt solution does not
            know it already, because the inverse tilt transformation is fitted numerically over
            the image. The shape also gives the pixel bounds of a wavelength solution passed as
            a GWCS without a bounding box.

        Raises
        ------
        ValueError
            If the image shape is not known from either the tilt solution or the argument.
        """
        if isinstance(tilt, gwcs.wcs.WCS):
            tilt = TiltSolution.from_gwcs(tilt, image_shape=image_shape)
        if tilt.image_shape is None:
            if image_shape is None:
                raise ValueError(
                    "The image shape must be given either by the tilt solution or as an "
                    "argument."
                )
            tilt = TiltSolution(tilt.c2d, disp_axis=tilt.disp_axis, image_shape=image_shape)

        if isinstance(wavelength, gwcs.wcs.WCS):
            bounds_pix = None if wavelength.bounding_box is not None else (0, tilt.image_shape[1])
            wavelength = WavelengthSolution1D.from_gwcs(wavelength, bounds_pix=bounds_pix)

        self._tilt: TiltSolution = tilt
        self._wavelength: WavelengthSolution1D = wavelength

    # --- Components -------------------------------------------------------------------------

    @property
    def tilt(self) -> TiltSolution:
        """The tilt solution mapping between the rectified and detector spaces."""
        return self._tilt

    @tilt.setter
    def tilt(self, value: TiltSolution) -> None:
        self._tilt = value
        self._clear_caches()

    @property
    def wavelength(self) -> WavelengthSolution1D:
        """The one-dimensional wavelength solution in the rectified space."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: WavelengthSolution1D) -> None:
        self._wavelength = value
        self._clear_caches()

    def _clear_caches(self) -> None:
        for name in ("p2w", "_w2p_table", "w2p", "gwcs"):
            self.__dict__.pop(name, None)

    @property
    def unit(self) -> u.Unit:
        """The wavelength unit."""
        return self._wavelength.unit

    @property
    def wave_air(self) -> bool:
        """Whether the solution gives air rather than vacuum wavelengths."""
        return self._wavelength.wave_air

    @property
    def bounds_pix(self) -> tuple[int, int]:
        """The pixel bounds of the wavelength solution along the rectified dispersion axis."""
        return self._wavelength.bounds_pix

    @property
    def disp_axis(self) -> int:
        """The index of the image's dispersion axis."""
        return self._tilt.disp_axis

    @property
    def image_shape(self) -> tuple[int, int]:
        """The detector image shape as ``(ny, nx)``."""
        return self._tilt.image_shape

    # --- Transformations --------------------------------------------------------------------

    @cached_property
    def p2w(self) -> CompoundModel:
        """Detector-pixel-to-wavelength transformation, ``(disp, cdisp) -> wavelength``.

        The inverse tilt transformation from detector to rectified space chained with the
        one-dimensional pixel-to-wavelength transformation, as a single astropy compound
        model with two inputs and one output.
        """
        return self._tilt.d2c | self._wavelength.p2w

    @cached_property
    def _w2p_table(self) -> tuple[NDArray, NDArray]:
        """Tabulated wavelength-to-rectified-pixel mapping covering the whole detector.

        The one-dimensional solution's own inverse spans only its pixel bounds, but the tilt
        can carry detector pixels far outside them in the rectified space, so the table is
        built over the rectified range reached by any detector pixel, with a small margin.
        """
        ny, nx = self.image_shape
        y = np.arange(ny, dtype=float)
        x_rec = self._tilt.d2c(
            np.concatenate([np.full(ny, -0.5), np.full(ny, nx - 0.5)]), np.tile(y, 2)
        )
        p = np.arange(np.floor(x_rec.min()) - 2, np.ceil(x_rec.max()) + 3)
        lam = self._wavelength.p2w(p)
        order = np.argsort(lam)
        return lam[order], p[order]

    @cached_property
    def w2p(self) -> Callable:
        """Wavelength-to-rectified-pixel transformation, `nan` outside the solution."""
        lam, p = self._w2p_table
        return interp1d(lam, p, bounds_error=False, fill_value=np.nan)

    def pix_to_wav(
        self, disp: float | ArrayLike, cdisp: float | ArrayLike
    ) -> float | NDArray | MaskedArray:
        """Map detector pixel coordinates into wavelength values.

        Parameters
        ----------
        disp
            The dispersion-axis pixel coordinate(s) in detector space.
        cdisp
            The cross-dispersion-axis pixel coordinate(s) in detector space.

        Returns
        -------
        The wavelength value(s) corresponding to the input detector pixel(s). If either input
        is a masked array, the result is a masked array with the combined mask.
        """
        if isinstance(disp, MaskedArray) or isinstance(cdisp, MaskedArray):
            wav = self.p2w(np.ma.getdata(disp), np.ma.getdata(cdisp))
            mask = np.ma.getmaskarray(disp) | np.ma.getmaskarray(cdisp)
            return np.ma.masked_array(wav, mask=mask)
        else:
            return self.p2w(disp, cdisp)

    def wav_to_pix(
        self, wav: float | ArrayLike, cdisp: float | ArrayLike
    ) -> float | NDArray | MaskedArray:
        """Map wavelength values and cross-dispersion coordinates into detector columns.

        Parameters
        ----------
        wav
            The wavelength value(s) to be converted.
        cdisp
            The cross-dispersion-axis pixel coordinate(s) in detector space.

        Returns
        -------
        The dispersion-axis pixel coordinate(s) in detector space, `nan` for wavelengths
        outside the solution. If either input is a masked array, the result is a masked array
        with the combined mask.
        """
        if isinstance(wav, MaskedArray) or isinstance(cdisp, MaskedArray):
            pix = self._tilt.c2d(self.w2p(np.ma.getdata(wav)), np.ma.getdata(cdisp))
            mask = np.ma.getmaskarray(wav) | np.ma.getmaskarray(cdisp)
            return np.ma.masked_array(pix, mask=mask)
        else:
            return self._tilt.c2d(self.w2p(wav), cdisp)

    @cached_property
    def gwcs(self) -> gwcs.wcs.WCS:
        """GWCS object mapping the 2D detector pixel frame to a 1D spectral frame."""
        detector_frame = coordinate_frames.CoordinateFrame(
            2,
            ("PIXEL", "PIXEL"),
            (0, 1),
            axes_names=("disp", "cdisp"),
            unit=[u.pix, u.pix],
            name="detector",
        )
        spectral_frame = coordinate_frames.SpectralFrame(
            axes_names=("wavelength",), unit=[self.unit]
        )
        pipeline = [(detector_frame, self.p2w), (spectral_frame, None)]
        return gwcs.wcs.WCS(pipeline)

    # --- Serialization ----------------------------------------------------------------------

    def to_asdf(self, path: str | Path, **kwargs) -> None:
        """Serialize the wavelength solution into an ASDF file.

        Stores the solution under the 'wavelength_solution_2d' key of the ASDF tree. The two
        component solutions are written as GWCS objects under 'tilt' and 'wavelength', which is
        what :meth:`from_asdf` reads back, and the chained detector-pixel-to-wavelength mapping
        is written once more under 'gwcs' so that any ASDF-aware tool can evaluate the solution
        without specreduce. All transformations are stored as analytic models, so the solution
        read back is identical to the one written.

        Parameters
        ----------
        path
            File to write to, given either as a path or as a writable file object.
        **kwargs
            Additional keyword arguments passed to `asdf.AsdfFile.write_to`.

        Notes
        -----
        The image shape and the pixel bounds travel as the bounding boxes of the exported
        GWCS objects. The bounding boxes are set on the exported objects only: the live
        :attr:`gwcs` property is deliberately left unbounded, because a bounded GWCS returns
        NaN outside its box instead of extrapolating.

        The air-wavelength flag is stored as a separate 'wave_air' entry because it states
        what the wavelengths mean rather than how they are computed, and neither GWCS
        spectral frames nor their physical types can express it.
        """
        ny, nx = self.image_shape
        image_box = ((0, nx), (0, ny))

        w_tilt = deepcopy(self._tilt.gwcs)
        w_tilt.bounding_box = image_box
        w_wave = deepcopy(self._wavelength.gwcs)
        w_wave.bounding_box = (self.bounds_pix,)
        w_2d = deepcopy(self.gwcs)
        w_2d.bounding_box = image_box

        tree = {
            "wavelength_solution_2d": {
                "gwcs": w_2d,
                "tilt": w_tilt,
                "wavelength": w_wave,
                "wave_air": bool(self.wave_air),
                "disp_axis": int(self.disp_axis),
            }
        }
        asdf.AsdfFile(tree).write_to(path, **kwargs)

    @classmethod
    def from_asdf(cls, path: str | Path) -> "WavelengthSolution2D":
        """Create a wavelength solution from an ASDF file written by :meth:`to_asdf`.

        Parameters
        ----------
        path
            File to read from, given either as a path or as a readable file object.

        Returns
        -------
        The wavelength solution stored in the file.

        Raises
        ------
        ValueError
            If the file holds no wavelength solution written by :meth:`to_asdf`.
        """
        with asdf.open(path, lazy_load=False) as af:
            try:
                node = af["wavelength_solution_2d"]
                w_tilt = node["tilt"]
                w_wave = node["wavelength"]
                wave_air = node["wave_air"]
                disp_axis = node["disp_axis"]
            except (KeyError, TypeError):
                raise ValueError(
                    "The ASDF file contains no wavelength solution written by "
                    "WavelengthSolution2D.to_asdf."
                )
            tilt = TiltSolution.from_gwcs(w_tilt, disp_axis=int(disp_axis))
            wavelength = WavelengthSolution1D.from_gwcs(w_wave, wave_air=bool(wave_air))
        return cls(tilt, wavelength)

    # --- Resampling -------------------------------------------------------------------------

    def resample(
        self,
        image,
        nbins: int | None = None,
        wlbounds: tuple[float, float] | None = None,
        bin_edges: ArrayLike | None = None,
        mask_treatment: MaskingOption = "apply",
    ) -> Spectrum:
        """Bin a 2D detector image onto a common wavelength grid conserving the flux.

        Each cross-dispersion row is binned independently: the wavelength bin edges are mapped
        through the inverse wavelength solution and the forward tilt transformation into that
        row's detector-space bin edges, and the flux of the detector pixels they enclose is
        summed with fractional weights at the edges. The binning is exact and conserves the
        integrated flux as long as the wavelength grid covers the whole detector.

        Parameters
        ----------
        image
            The 2D detector image, in any of the formats accepted by
            `~specreduce.core.parse_image`. The dispersion and cross-dispersion axis alignment
            should be the same as in the arc frames the tilt solution was fitted to.
        nbins
            The number of wavelength bins. If not provided, it defaults to the number of
            detector pixels along the dispersion axis.
        wlbounds
            The starting and ending wavelengths of the grid. If not provided, the grid spans
            the wavelengths covered by the reference row of the tilt solution, where the
            rectified and detector coordinates coincide.
        bin_edges
            Explicit bin edges in the wavelength space, as a 1D array-like ``[e_0, e_1, ...,
            e_n]`` with ``n = nbins + 1``. If provided, ``nbins`` and ``wlbounds`` are ignored.
        mask_treatment
            Specifies how to handle masked or non-finite values in the input image. See
            `~specreduce.core.parse_image` for the accepted values.

        Returns
        -------
        A 2D `~specutils.Spectrum` holding the image binned onto the wavelength grid, in units
        of flux per wavelength unit, with the bin centres as its spectral axis, the propagated
        variance as its uncertainty (represented as the input uncertainty type), and a mask
        flagging every bin that received flux from a masked detector pixel. The axis order
        follows the input image, and the spectral axis index is set accordingly.

        Notes
        -----
        The flux convention matches `~specreduce.wavesol1d.WavelengthSolution1D.resample`:
        each detector pixel contributes its value multiplied by its own wavelength width, and
        the sum in each bin is divided by the bin width, so that the integral of the output
        over wavelength equals the sum over the input of flux times pixel wavelength width. The
        pixel wavelength widths are evaluated per row from the chained transformation, which
        accounts for the change in dispersion introduced by the tilt without any further
        normalization.
        """
        if nbins is not None and nbins < 0:
            raise ValueError("Number of bins must be non-zero and positive.")

        im = parse_image(image, disp_axis=self.disp_axis, mask_treatment=mask_treatment)
        flux = im.flux.value
        ny, nx = flux.shape
        mask = im.mask if im.mask is not None else np.zeros((ny, nx), dtype=bool)

        if im.uncertainty is not None:
            var = im.uncertainty.represent_as(VarianceUncertainty).array
            ucty_type = type(im.uncertainty)
        else:
            var = np.zeros_like(flux)
            ucty_type = VarianceUncertainty

        # Wavelength grid.
        nbins = nx if nbins is None else nbins
        if bin_edges is not None:
            bin_edges_wav = np.asarray(bin_edges, dtype=float)
            nbins = bin_edges_wav.size - 1
        else:
            if wlbounds is None:
                l1, l2 = self._wavelength.p2w(np.array([-0.5, nx - 0.5]))
            else:
                l1, l2 = wlbounds
            bin_edges_wav = np.linspace(l1, l2, num=nbins + 1)
        bin_centers_wav = 0.5 * (bin_edges_wav[:-1] + bin_edges_wav[1:])
        bin_widths_wav = np.diff(bin_edges_wav)

        # Bin edges in detector space, one set per row. Wavelengths outside the tabulated
        # inverse saturate at its ends, which land outside the detector and are clipped
        # into zero-width bins.
        ypix = np.arange(ny, dtype=float)
        lam, p = self._w2p_table
        edges_rec = np.interp(bin_edges_wav, lam, p)
        bin_edges_det = np.clip(self._tilt.c2d(*np.meshgrid(edges_rec, ypix)) + 0.5, 0, nx - 1e-12)
        bin_edge_ix = np.floor(bin_edges_det).astype(int)
        bin_edge_w = bin_edges_det - bin_edge_ix

        # Wavelength width of every detector pixel, from the chained transformation.
        xe, ye = np.meshgrid(np.arange(nx + 1) - 0.5, ypix)
        dldx = np.diff(self.p2w(xe, ye), axis=1)
        fd = flux * dldx
        vd = var * dldx**2

        flux_wl = np.zeros((ny, nbins))
        var_wl = np.zeros((ny, nbins))
        mask_wl = np.zeros((ny, nbins), dtype=bool)
        ixs = np.tile(np.arange(nx), (ny, 1))
        ys = np.arange(ny)

        for i in range(nbins):
            i1, i2 = bin_edge_ix[:, i : i + 2].T
            # Rows where the whole bin falls inside a single detector pixel.
            m = i1 == i2
            if m.any():
                fracw = bin_edges_det[:, i + 1] - bin_edges_det[:, i]
                flux_wl[m, i] = (fracw * fd[ys, i1])[m]
                var_wl[m, i] = (fracw**2 * vd[ys, i1])[m]
                mask_wl[m, i] = ((fracw > 0) & mask[ys, i1])[m]
            # Rows where the bin spans several detector pixels: fractional weights at the
            # edge pixels, unit weights in between.
            if not m.all():
                imin, imax = i1.min(), i2.max() + 1
                ixc = ixs[:, imin:imax]
                w = np.zeros((ny, imax - imin))
                w[(ixc > i1[:, None]) & (ixc < i2[:, None])] = 1.0
                w[ys, i1 - imin] = 1.0 - bin_edge_w[:, i]
                w[ys, i2 - imin] = bin_edge_w[:, i + 1]
                flux_wl[~m, i] = (fd[~m, imin:imax] * w[~m]).sum(1)
                var_wl[~m, i] = (vd[~m, imin:imax] * w[~m] ** 2).sum(1)
                mask_wl[~m, i] = ((w[~m] > 0) & mask[~m, imin:imax]).any(1)

        flux_wl = flux_wl / bin_widths_wav * im.unit / self.unit
        ucty_wl = VarianceUncertainty(var_wl / bin_widths_wav**2).represent_as(ucty_type)

        spectral_axis_index = 1
        if self.disp_axis == 0:
            flux_wl = flux_wl.T
            ucty_wl = ucty_wl.__class__(ucty_wl.array.T)
            mask_wl = mask_wl.T
            spectral_axis_index = 0

        return Spectrum(
            flux_wl,
            spectral_axis=bin_centers_wav * self.unit,
            uncertainty=ucty_wl,
            mask=mask_wl,
            spectral_axis_index=spectral_axis_index,
        )
