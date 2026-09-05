import warnings
from functools import cached_property
from pathlib import Path
from typing import Callable
from copy import deepcopy

import asdf
import astropy.units as u
import gwcs
import numpy as np
from astropy.modeling import models, CompoundModel
from astropy.nddata import VarianceUncertainty
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS, InvalidTransformError
from gwcs import coordinate_frames
from numpy.ma import MaskedArray
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.interpolate import interp1d

from specutils import Spectrum


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


def _make_grism_wcs(
    p: ArrayLike, crpix: float, crval: float, cdelt: float, ctype: str, cunit: str
) -> WCS:
    """Create a 1D FITS WCS using the WCS Paper III grism dispersion function.

    The grism dispersion function and its PV parameters are defined in Greisen et al.
    (2006, A&A 446, 747), Sect. 5.1 and Table 6.

    Parameters
    ----------
    p
        Fitted grism parameters (g, alpha, nrp, theta): the effective grating ruling
        density [1/m], the incidence angle [deg], the derivative of the prism refractive
        index with respect to wavelength at the reference wavelength [1/m], and the
        camera angle [deg].
    crpix
        Reference pixel in the one-based FITS convention.
    crval
        Wavelength at the reference pixel.
    cdelt
        Linear dispersion at the reference pixel.
    ctype
        Spectral axis type, either 'AWAV-GRA' (air) or 'WAVE-GRI' (vacuum).
    cunit
        Wavelength unit string.

    Returns
    -------
    A 1D `~astropy.wcs.WCS` with a grism dispersion spectral axis.
    """
    g, alpha, nrp, theta = p
    w = WCS(naxis=1)
    w.wcs.ctype = [ctype]
    w.wcs.cunit = [cunit]
    w.wcs.crpix = [crpix]
    w.wcs.crval = [crval]
    w.wcs.cdelt = [cdelt]
    w.wcs.set_pv(
        [
            (1, 0, g),  # G: effective ruling density [1/m], absorbs m and cos(eps)
            (1, 1, 1.0),  # m: interference order, fixed (enters only in the product G*m)
            (1, 2, alpha),  # alpha: incidence angle [deg], absorbs n_r
            (1, 3, 1.0),  # n_r: prism refractive index, fixed (enters only via n_r*sin(alpha))
            (1, 4, nrp),  # n'_r: dn/dlambda at the reference wavelength [1/m]
            (1, 5, 0.0),  # eps: grating tilt, fixed (enters only via G*m/cos(eps))
            (1, 6, theta),  # theta: camera angle [deg]
        ]
    )
    return w


GRISM_SENTINEL = 1e8
"""Residual value standing in for unphysical grism parameter combinations."""


def _fit_grism_dispersion(
    x_fits: NDArray,
    lam: NDArray,
    crpix: float,
    crval: float,
    cdelt: float,
    ctype: str,
    cunit: str,
    m_to_unit: float,
) -> optimize.OptimizeResult:
    """Fit the WCS Paper III grism dispersion function to an exact wavelength solution.

    Fits the four free grism parameters (g, alpha, nrp, theta) with a single bounded
    least-squares run. The reference pixel keywords (CRPIX, CRVAL, CDELT) are exact
    statements about the wavelength solution and pin the value and slope of the dispersion
    function at the reference pixel, which makes the problem unimodal enough for a local
    optimizer. The initial ruling density is the Littrow-configuration value at the
    reference wavelength, which places the starting point inside the arcsin domain of the
    grism equation (Greisen et al. 2006, Eq. 68) for any reference wavelength, and pixels
    for which wcslib returns non-finite values yield large finite residuals that keep the
    optimizer within the valid parameter space.

    Parameters
    ----------
    x_fits
        Pixel coordinates as a (npix, 1) array in the one-based FITS convention.
    lam
        Exact wavelengths at ``x_fits`` in the solution unit.
    crpix
        Fixed reference pixel in the one-based FITS convention.
    crval
        Fixed wavelength at the reference pixel.
    cdelt
        Fixed linear dispersion at the reference pixel.
    ctype
        Spectral axis type, either 'AWAV-GRA' (air) or 'WAVE-GRI' (vacuum).
    cunit
        Wavelength unit string in the FITS format.
    m_to_unit
        Conversion factor from metres to the solution unit, needed because wcslib returns
        metres for a spectral axis regardless of the CUNIT.

    Returns
    -------
    The `~scipy.optimize.OptimizeResult` of the fit, where ``x`` holds the optimized
    grism parameters as in `_make_grism_wcs` and ``fun`` the residual wavelengths in the
    solution unit.

    Raises
    ------
    RuntimeError
        If the grism dispersion function cannot be evaluated anywhere along the
        spectral axis at the fitted parameters.
    """

    def residuals(p: NDArray) -> NDArray:
        try:
            w = _make_grism_wcs(p, crpix, crval, cdelt, ctype, cunit)
            res = w.wcs_pix2world(x_fits, 1)[:, 0] * m_to_unit - lam
        except InvalidTransformError:
            return np.full(lam.size, GRISM_SENTINEL)
        return np.where(np.isfinite(res), res, GRISM_SENTINEL)

    lam_ref_m = crval / m_to_unit
    g0 = float(np.clip(2.0 * np.sin(np.radians(10.0)) / lam_ref_m, 1e3, 1e7))
    p0 = [g0, 10.0, 0.0, 0.0]  # g, alpha, nrp, theta

    # Set the WCS once outside the fit so that parameter-independent configuration errors
    # (such as an invalid CUNIT) propagate instead of being fenced off as unphysical
    # parameter combinations. Out-of-domain grism parameters do not raise here: wcslib
    # signals them with non-finite coordinates handled in `residuals`.
    _make_grism_wcs(p0, crpix, crval, cdelt, ctype, cunit).wcs.set()

    bounds = ([1e3, -89.9, -1e7, -89.9], [1e7, 89.9, 1e7, 89.9])
    fit = optimize.least_squares(
        residuals, p0, bounds=bounds, method="trf", x_scale=[g0, 10.0, 1e4, 10.0]
    )
    if np.all(fit.fun == GRISM_SENTINEL):
        raise RuntimeError(
            "The grism dispersion function could not be evaluated anywhere along the "
            "spectral axis: the wavelength solution is outside the parameter space "
            "reachable by the 'WAVE-GRI'/'AWAV-GRA' dispersion model. The 'gwcs' property "
            "provides a lossless representation."
        )
    return fit


class WavelengthSolution1D:
    def __init__(
        self,
        p2w: None | CompoundModel,
        bounds_pix: tuple[int, int],
        unit: u.Unit,
        wave_air: bool = False,
    ) -> None:
        """Class defining a one-dimensional wavelength solution.

        This class manages the mapping between pixel positions and wavelength values in a 1D
        spectrum, supporting both forward and reverse transformations. It provides methods for
        resampling spectra in the pixel-to-wavelength space while conserving flux, and integrates
        with GWCS for coordinate transformations.

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
        wave_air
            Whether the solution maps pixels to air rather than vacuum wavelengths; by
            default `False`, meaning vacuum wavelengths.
        """
        self.unit = unit
        self.wave_air = wave_air
        self._unit_str = unit.to_string("latex")
        self.bounds_pix: tuple[int, int] = bounds_pix
        self.bounds_wav: tuple[float, float] | None = None
        self._wcs_cache: dict[bool, tuple[tuple, float]] = {}
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
        self._wcs_cache.clear()

    @cached_property
    def p2w_dldx(self) -> CompoundModel:
        """Partial derivative of the pixel-to-wavelength transformation, (d lambda) / (d pix)."""
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

    def wcs(self, wave_air: bool | None = None, max_residual: float = 1.0) -> WCS:
        """Fit and return a FITS WCS approximating the wavelength solution.

        Fits the FITS WCS Paper III grism dispersion function (Greisen et al. 2006,
        A&A 446, 747, Sect. 5.1) to the exact pixel-to-wavelength model and returns the
        result as a standard `~astropy.wcs.WCS` object that can be serialized into a FITS
        header and evaluated by any FITS-compliant reader. The spectral axis type is
        'AWAV-GRA' for air wavelengths or 'WAVE-GRI' for vacuum wavelengths.

        Note on the naming: despite appearances, neither algorithm code refers to a
        grating versus a grism. Both 'GRA' and 'GRI' denote the same Paper III grism
        dispersion algorithm, and the trailing letter only identifies the medium the
        wavelengths are measured in: 'GRA' is the grism dispersion relation for air
        wavelengths and 'GRI' the one for vacuum wavelengths. The algorithm code is
        therefore always paired with the matching coordinate type: 'AWAV' (air
        wavelength) with 'GRA', and 'WAVE' (vacuum wavelength) with 'GRI'.

        The fit is cached per air/vacuum axis type and invalidated whenever the
        pixel-to-wavelength transformation changes. Each call returns an independent copy
        of the cached WCS, so the caller is free to modify the result.

        Parameters
        ----------
        wave_air
            Whether the spectral axis represents air rather than vacuum wavelengths,
            selecting between the 'AWAV-GRA' and 'WAVE-GRI' axis types. If `None`, the
            value given in the initialization is used. The flag only declares what the
            solution's wavelengths mean: no air-vacuum conversion is applied to the
            wavelength values, so overriding it relabels the axis without changing the
            numbers.
        max_residual
            Maximum allowed absolute deviation between the fitted dispersion function and
            the exact solution in pixels. A fit exceeding this limit emits an
            `~astropy.utils.exceptions.AstropyUserWarning`.

        Returns
        -------
        A 1D `~astropy.wcs.WCS` using the Paper III grism dispersion function.

        Raises
        ------
        ValueError
            If the wavelength solution is not set, its unit is not a length unit, or it
            produces non-finite values within the pixel bounds.
        TypeError
            If the wavelength solution does not use a power-series
            `~astropy.modeling.polynomial.Polynomial1D` model.
        RuntimeError
            If the grism dispersion function cannot be evaluated anywhere along the
            spectral axis, meaning the solution is outside the parameter space reachable
            by the dispersion model.

        Notes
        -----
        The reference pixel keywords are exact statements about the wavelength solution
        rather than fit parameters: CRPIX is the solution's reference pixel, and CRVAL and
        CDELT are the value and derivative of the solution polynomial at that pixel. Only
        four grism terms (the grating ruling density PV1_0, the incidence angle PV1_2,
        the refractive index derivative PV1_4, and the camera angle PV1_6; see Greisen
        et al. 2006, Table 6) are fitted, so all approximation error lives in the PV
        terms, and the residual is zero in both value and slope at the reference pixel
        and grows towards the chip edges.

        The grism dispersion function cannot represent an arbitrary polynomial exactly,
        so the returned WCS is a numerical approximation of the wavelength solution and
        the fitted PV values are effective rather than physical. The remaining Table 6
        parameters are fixed at values that make them redundant with the fitted ones: the
        interference order PV1_1 and the grating tilt PV1_5 enter the grism equation only
        through the combination G*m/cos(eps), so they are absorbed by the fitted ruling
        density, and the prism refractive index PV1_3 is fixed at 1 (the paper's plain
        grating case) because it enters only through the product n_r*sin(alpha) and is
        absorbed by the fitted incidence angle; PV1_4 acts as a curvature absorber rather
        than a measure of glass dispersion. No spectrograph physics should be read from
        the fitted PV values. If the fit fails to reach ``max_residual``, the solution is
        far from any grism dispersion curve, and the lossless :attr:`gwcs` property
        should be used instead.

        Vacuum solutions use 'WAVE-GRI' (the grism-in-vacuum form of the algorithm,
        Sect. 5.1.2 of the paper) rather than 'WAVE-GRA' because the 'GRA' form
        (Sect. 5.1.4) is native to air wavelengths: pairing it with a vacuum axis would
        route every evaluation through wcslib's air-refraction model, which is not valid
        below ~200 nm.

        Note also that wcslib normalizes spectral axes to SI units, so evaluating the
        returned WCS yields wavelengths in metres regardless of the CUNIT, and serializing
        it with ``to_header()`` likewise writes CRVAL and CDELT in metres with CUNIT 'm'.
        """
        air = self.wave_air if wave_air is None else wave_air
        if air not in self._wcs_cache:
            self._wcs_cache[air] = self._fit_wcs(air)
        args, max_res = self._wcs_cache[air]

        if max_res > max_residual:
            warnings.warn(
                f"The fitted FITS WCS deviates from the wavelength solution by up to "
                f"{max_res:.2f} pixels, exceeding the given limit of {max_residual} pixels. "
                f"The 'gwcs' property provides a lossless representation.",
                AstropyUserWarning,
            )
        return _make_grism_wcs(*args)

    def _fit_wcs(self, air: bool) -> tuple[tuple, float]:
        """Fit the grism dispersion function to the wavelength solution.

        Caching the arguments of `_make_grism_wcs` rather than the fitted WCS itself keeps
        every call to :meth:`wcs` returning a pristine object: wcslib normalizes a WCS to SI
        units in place the first time it is evaluated, so a shared or copied instance would
        hand out metres where a freshly built one still carries the solution unit. Rebuilding
        costs microseconds against the milliseconds of the fit.

        Parameters
        ----------
        air
            Whether the spectral axis represents air rather than vacuum wavelengths.

        Returns
        -------
        The arguments needed to rebuild the fitted `~astropy.wcs.WCS` and the largest
        deviation from the exact solution within the pixel bounds in pixels.
        """
        if self._p2w is None:
            raise ValueError("Wavelength solution not set.")
        if not isinstance(self._p2w[1], models.Polynomial1D):
            raise TypeError(
                "FITS WCS export requires the wavelength solution to use a power-series "
                f"Polynomial1D model, got {type(self._p2w[1]).__name__}."
            )

        ctype = "AWAV-GRA" if air else "WAVE-GRI"
        cunit = self.unit.to_string("fits")

        try:
            m_to_unit = (1.0 * u.m).to_value(self.unit)
        except u.UnitConversionError:
            raise ValueError(f"FITS WCS export requires a length unit, got '{self.unit}'.")

        crpix = -self.ref_pixel + 1.0
        crval = float(self._p2w[1].c0.value)
        cdelt = float(self._p2w[1].c1.value)
        x_model = np.arange(self.bounds_pix[0], self.bounds_pix[1])
        x_fits = x_model[:, None] + 1.0
        lam = self.p2w(x_model)
        if not np.all(np.isfinite(lam)):
            raise ValueError(
                "The wavelength solution produces non-finite values within the pixel "
                "bounds and cannot be exported as a FITS WCS."
            )

        fit = _fit_grism_dispersion(x_fits, lam, crpix, crval, cdelt, ctype, cunit, m_to_unit)

        with np.errstate(divide="ignore", invalid="ignore"):
            res_pix = np.abs(fit.fun / self.p2w_dldx(x_model))
        max_res = float(np.nanmax(res_pix)) if np.any(np.isfinite(res_pix)) else np.inf
        return (fit.x, crpix, crval, cdelt, ctype, cunit), max_res

    def attach_wcs(
        self, spectrum: Spectrum, wave_air: bool | None = None, max_residual: float = 1.0
    ) -> Spectrum:
        """Attach the fitted FITS WCS to a copy of a pixel-space spectrum.

        Returns a new `~specutils.Spectrum` holding the flux, uncertainty, mask, and metadata
        of the given spectrum with the FITS WCS returned by :meth:`wcs` as its world
        coordinate system. The samples are left where they are and only their spectral
        coordinates change, so the result can be written with the specutils 'wcs1d-fits'
        format, which stores the flux as an image and the wavelength solution as WCS header
        keywords. Use :meth:`resample` instead to rebin the flux onto a wavelength grid.

        The fitted WCS is cached per air/vacuum axis type and reused, so applying one
        solution to many spectra runs the grism fit only once. The cache is cleared whenever
        the pixel-to-wavelength transformation changes.

        Parameters
        ----------
        spectrum
            A pixel-space spectrum sampled on the pixel grid the solution was fitted on.
        wave_air
            Whether the spectral axis represents air rather than vacuum wavelengths. If
            `None`, the value given in the initialization is used.
        max_residual
            Maximum allowed absolute deviation between the fitted dispersion function and
            the exact solution in pixels. A fit exceeding this limit emits an
            `~astropy.utils.exceptions.AstropyUserWarning`.

        Returns
        -------
        A spectrum with the same flux as the input and a FITS WCS spectral axis.

        Raises
        ------
        ValueError
            If the spectrum is not one-dimensional, or if its length does not match the
            pixel range the solution was fitted over.

        Notes
        -----
        The spectrum is assumed to be sampled one flux value per pixel over the solution's
        pixel bounds, which is what the calibration produces: the WCS maps array index i to
        the solution's pixel i, and a length mismatch means the spectrum does not share the
        pixel grid the solution was derived on.

        Because wcslib normalizes spectral axes to SI units, the spectral axis of the
        returned spectrum is in metres regardless of the solution unit.
        """
        if spectrum.flux.ndim != 1:
            raise ValueError(
                f"Applying a wavelength solution requires a one-dimensional spectrum, got "
                f"{spectrum.flux.ndim} dimensions."
            )

        npix = spectrum.flux.shape[0]
        npix_solution = self.bounds_pix[1] - self.bounds_pix[0]
        if npix != npix_solution:
            raise ValueError(
                f"The spectrum has {npix} pixels but the wavelength solution was fitted "
                f"over {npix_solution}."
            )

        return Spectrum(
            flux=spectrum.flux,
            wcs=self.wcs(wave_air=wave_air, max_residual=max_residual),
            uncertainty=spectrum.uncertainty,
            mask=spectrum.mask,
            meta=spectrum.meta,
        )

    def to_asdf(self, path: str | Path, **kwargs) -> None:
        """Serialize the wavelength solution into an ASDF file.

        Stores the pixel-to-wavelength transformation as a GWCS object under the
        'wavelength_solution' key of the ASDF tree. The transformation is written as an
        analytic model rather than as a tabulated array, so the solution read back is
        identical to the one written, and the file can be opened by any ASDF-aware tool
        without specreduce.

        Only the coordinate transformation is stored: the line lists, matched-line tables,
        and fit diagnostics held by the calibration that produced the solution are not
        included.

        Parameters
        ----------
        path
            File to write to, given either as a path or as a writable file object.
        **kwargs
            Additional keyword arguments passed to `asdf.AsdfFile.write_to`.

        Raises
        ------
        ValueError
            If the wavelength solution is not set.

        Notes
        -----
        The pixel bounds travel as the bounding box of the exported GWCS, which is the
        idiomatic way to declare the domain over which a GWCS is valid. The bounding box is
        set on the exported object only: the :attr:`gwcs` property is deliberately left
        unbounded, because a bounded GWCS returns NaN outside its box instead of
        extrapolating.

        The air-wavelength flag is stored as a separate 'wave_air' entry because it states
        what the wavelengths mean rather than how they are computed, and neither GWCS
        spectral frames nor their physical types can express it.
        """
        if self._p2w is None:
            raise ValueError("Wavelength solution not set.")

        w = deepcopy(self.gwcs)
        w.bounding_box = (self.bounds_pix,)
        tree = {"wavelength_solution": {"gwcs": w, "wave_air": bool(self.wave_air)}}
        asdf.AsdfFile(tree).write_to(path, **kwargs)

    @classmethod
    def from_asdf(cls, path: str | Path) -> "WavelengthSolution1D":
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
            If the file holds no wavelength solution written by :meth:`to_asdf`, if the
            stored GWCS carries no bounding box giving the pixel bounds, or if its
            transformation is not a shift followed by a polynomial.
        """
        with asdf.open(path, lazy_load=False) as af:
            try:
                node = af["wavelength_solution"]
                w = node["gwcs"]
                wave_air = node["wave_air"]
            except (KeyError, TypeError):
                raise ValueError(
                    "The ASDF file contains no wavelength solution written by "
                    "WavelengthSolution1D.to_asdf."
                )

            bbox = w.bounding_box
            if bbox is None:
                raise ValueError(
                    "The stored GWCS carries no bounding box giving the pixel bounds of "
                    "the wavelength solution."
                )
            bounds_pix = bbox.bounding_box()
            unit = u.Unit(w.output_frame.unit[0])
            p2w = w.forward_transform.copy()

        # The bounding box lives on the forward transform, so drop it to keep the
        # restored transformation extrapolating exactly like the one that was written.
        del p2w.bounding_box

        if not (isinstance(p2w, CompoundModel) and isinstance(p2w[0], models.Shift)):
            raise ValueError(
                "WavelengthSolution1D requires the stored transformation to be a shift "
                f"followed by a polynomial, got {p2w.__class__.__name__}."
            )
        return cls(p2w, (int(bounds_pix[0]), int(bounds_pix[1])), unit, wave_air=bool(wave_air))

    def resample(
        self,
        spectrum: "Spectrum",
        nbins: int | None = None,
        wlbounds: tuple[float, float] | None = None,
        bin_edges: ArrayLike | None = None,
    ) -> Spectrum:
        """Bin the given pixel-space 1D spectrum to a wavelength space conserving the flux.

        The input flux is taken to be integrated per pixel (e.g. counts per pixel, as
        produced by the extraction methods). Each wavelength bin receives the flux of the
        pixels it overlaps, weighted by the overlapping fraction of each pixel, and the total
        is divided by the bin width, so the output is a flux density per wavelength unit.
        The binning is exact and conserves the integrated flux. The variance is propagated
        with the squared weights, assuming independent pixel noise.

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
            1D spectrum binned to the specified wavelength bins, with the flux in units of
            the input flux unit per wavelength unit and the uncertainty in the same
            uncertainty class as the input.
        """
        if nbins is not None and nbins < 0:
            raise ValueError("Number of bins must be non-zero and positive.")

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

        for i in range(nbins):
            i1, i2 = bin_edge_ix[i : i + 2]
            weights[:] = 0.0
            if i1 != i2:
                weights[i1 + 1 : i2] = 1.0
                weights[i1] = 1 - bin_edge_w[i]
                weights[i2] = bin_edge_w[i + 1]
                sl = slice(i1, i2 + 1)
                w = weights[sl]
                flux_wl[i] = (w * flux[sl]).sum()
                ucty_wl[i] = (w**2 * ucty[sl]).sum()
            else:
                fracw = bin_edges_pix[i + 1] - bin_edges_pix[i]
                flux_wl[i] = fracw * flux[i1]
                ucty_wl[i] = fracw**2 * ucty[i1]

        bin_widths_wav = np.diff(bin_edges_wav)
        flux_wl = flux_wl / bin_widths_wav * spectrum.flux.unit / self.unit
        ucty_wl = VarianceUncertainty(ucty_wl / bin_widths_wav**2).represent_as(ucty_type)
        return Spectrum(flux_wl, bin_centers_wav * self.unit, uncertainty=ucty_wl)
