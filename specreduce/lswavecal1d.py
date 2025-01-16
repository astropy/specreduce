import warnings
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.modeling import models, Model, fitting
from astropy.nddata import VarianceUncertainty
from gwcs import coordinate_frames as cf
from gwcs import wcs
from matplotlib.pyplot import setp, subplots
from scipy import optimize
from scipy.spatial import KDTree

from numpy import ndarray
from specutils import Spectrum1D

from specreduce.calibration_data import load_pypeit_calibration_lines
from specreduce.line_matching import find_arc_lines


def diff_poly1d(m: models.Polynomial1D) -> models.Polynomial1D:
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
    models.Polynomial1D
        A new Polynomial1D model representing the derivative of the input
        Polynomial1D model.
    """
    coeffs = {f'c{i-1}': i*getattr(m, f'c{i}').value for i in range(1, m.degree+1)}
    return models.Polynomial1D(m.degree-1, **coeffs)


class WavelengthSolution1D:
    def __init__(self, *, line_lists,
                 arc_spectra: Spectrum1D | Iterable[Spectrum1D] | None = None,
                 obs_lines: ndarray | Iterable[ndarray] | None = None,
                 wlbounds: tuple[float, float] = (0, np.inf),
                 wave_air: bool = False):

        if arc_spectra is None and obs_lines is None:
            raise ValueError("Either arc_spectra or obs_lines must be provided.")

        if arc_spectra is not None and obs_lines is not None:
            raise ValueError("Only one of arc_spectra or obs_lines can be provided.")

        self.wlbounds: tuple[float, float] = wlbounds
        self.wave_air: bool = wave_air

        self.arc_spectra: Iterable[Spectrum1D] = [arc_spectra] if isinstance(arc_spectra, Spectrum1D) else arc_spectra
        self.lines_pix: Iterable[ndarray] = [obs_lines] if isinstance(obs_lines, ndarray) else obs_lines

        self.lines_wav: Iterable[ndarray] | None = None
        self._trees: Iterable[KDTree] | None = None
        self._read_linelists(line_lists)

        self._fit: optimize.OptimizeResult | None = None
        self._wcs: wcs.WCS | None = None

        self._p2w: Model | None = None           # The fitted pixel -> wavelength model
        self._w2p: Model | None = None           # The fitted wavelength -> pixel model
        self._p2w_dldx: Model | None = None      # delta lambda / delta pixel
        self._w2p_dxdl: Model | None = None      # delta pixel / delta lambda

    def _read_linelists(self, line_lists):
        """Load and filter calibration line lists for specified lamps and wavelength bounds.

        Notes
        -----
        This method uses the `load_pypeit_calibration_lines` function to load the
        line lists for each lamp. It filters the loaded data to only include lines
        within the defined wavelength boundaries. The filtered data is stored in
        `self.linelists`, and the wavelengths are extracted and stored in
        `self.lines_wav`. Additionally, KDTree objects are created for each extracted
        wavelength list and stored in `self._trees` for quick nearest-neighbor
        queries of line wavelengths.
        """
        if not isinstance(line_lists, (tuple, list)):
            line_lists = [line_lists]

        self.lines_wav = []
        for l in line_lists:
            if isinstance(l, ndarray):
                self.lines_wav.append(l)
            else:
                lines = []
                if isinstance(l, str):
                    l = [l]
                for ll in l:
                    lines.append(load_pypeit_calibration_lines(l, wave_air=self.wave_air)['wavelength'].value)
                self.lines_wav.append(np.concatenate(lines))
        for i, l in enumerate(self.lines_wav):
            self.lines_wav[i] = l[(l >= self.wlbounds[0]) & (l <= self.wlbounds[1])]
        self._trees = [KDTree(l[:, None]) for l in self.lines_wav]

    def find_lines(self, fwhm: float):
        """Finds lines in the provided arc spectra.

        This method determines the spectral lines within each spectrum of the arc spectra
        based on the provided initial guess for the line Full Width at Half Maximum (FWHM).

        Parameters
        ----------
        fwhm
            Initial guess for the FWHM for the spectral lines, used as a parameter in
            the ``find_arc_lines`` function to locate and identify spectral arc lines.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lines_obs = [find_arc_lines(sp, fwhm) for sp in self.arc_spectra]
        self.lines_pix = [lo['centroid'].value for lo in lines_obs]

    def fit(self, ref_pixel: float,
            wavelength_bounds: tuple[float, float],
            dispersion_bounds: tuple[float, float],
            popsize: int = 30,
            max_distance: float = 100):
        """Calculate and refine the pixel-to-wavelength and wavelength-to-pixel transformations.

        This method determines the functional relationships between pixel positions and
        wavelengths in a spectrum, including their derivatives, by fitting calibration data
        to given constraints. The transformations include forward (pixel-to-wavelength)
        and backward (wavelength-to-pixel) mappings as well as their respective derivatives
        across the spectral axis.

        Parameters
        ----------
        ref_pixel : float
            The reference pixel position used as the zero point for the transformation.
        wavelength_bounds : tuple of float
            Initial bounds for the wavelength value at the reference pixel.
        dispersion_bounds : tuple of float
            Initial bounds for the dispersion value at the reference pixel.
        popsize : int, optional
            The population size for differential evolution optimization.
        max_distance : float, optional
            The maximum allowable distance when querying the tree in the optimization process.

        Raises
        ------
        ValueError
            Raised if the optimization or fitting process fails due to invalid inputs or constraints.
        """
        model = models.Shift(-ref_pixel) | models.Polynomial1D(3)
        def minfun(x):
            return sum(
                [np.clip(t.query(model.evaluate(l, -ref_pixel, *x)[:, None])[0], 0, max_distance).sum() for t, l in
                 zip(self._trees, self.lines_pix)])

        # Calculate the pixel -> wavelength transform and its derivative along the spectral axis
        self._fit = fit = optimize.differential_evolution(minfun,
                                                          bounds=[wavelength_bounds, dispersion_bounds, [-1e-3, 1e-3], [-1e-4, 1e-4]],
                                                          popsize=popsize)
        self._p2w = models.Shift(-ref_pixel) | models.Polynomial1D(3, **{f'c{i}': fit.x[i] for i in
                                                                                  range(fit.x.size)})
        self._refine_fit()
        p2w = self._p2w
        self._p2w_dldx = models.Shift(p2w.offset_0) | diff_poly1d(p2w[1])

        # Calculate the wavelength -> pixel transform and its derivative along the spectral axis
        vpix = self.arc_spectra[0].spectral_axis.value
        vwav = p2w(vpix)
        w2p = models.Polynomial1D(6, c0=-p2w.offset_0, c1=1 / p2w.c1_1, fixed={'c0': True})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._w2p = w2p = models.Shift(-p2w.c0_1) | fitting.DogBoxLSQFitter()(w2p, vwav - p2w.c0_1.value, vpix)
        self._w2p_dxdl = models.Shift(w2p.offset_0) | diff_poly1d(w2p[1])

    def _refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0):
        """Refine the global fit of the pixel-to-wavelength transformation.

        Refines the fit of a polynomial model to data by performing a fitting operation
        using matched pixel and wavelength data points. The method uses a linear least
        squares fitter to optimize the model parameters based on the match.

        Parameters
        ----------
        degree : int, optional
            The degree of the polynomial model used for fitting. Higher values
            allow for more complex polynomial models. Default is 4.

        match_distance_bound : float, optional
            Maximum allowable distance used to identify matched pixel and wavelength
            data points. Points exceeding the bound will not be considered in the fit.
            Default is 5.0.
        """
        matched_pix, matched_wav = self.match_lines(match_distance_bound)
        model = models.Polynomial1D(degree, **{n: getattr(self._p2w[-1], n).value for n in self._p2w[-1].param_names})
        fitter = fitting.LinearLSQFitter()
        self._p2w = self._p2w[0].copy() | fitter(model, matched_pix + self._p2w.offset_0.value, matched_wav)

    def resample(self, spectrum: Spectrum1D,
                 nbins: int | None = None,
                 wlbounds: tuple[float, float] | None = None,
                 bin_edges: Iterable[float] | None = None) -> Spectrum1D:
        """Resample the given 1D spectrum to a specified wavelength bins conserving the flux.

        This function adjusts the flux data by resampling it over wavelength bins. The wavelength mapping
        is determined by the object's internal methods `_p2w` and `_p2w_dldx` for pixel-to-wavelength
        conversion. The resulting flux values are normalized to ensure consistency with the input flux's
        total spectral intensity.

        Parameters
        ----------
        spectrum : ndarray
            1D array representing the flux data to be resampled over the wavelength space.
        nbins : int or None, optional
            The number of bins for resampling. If not provided, it defaults to the size of the input
            `flux` array.
        wlbounds : tuple of float or None, optional
            A tuple specifying the starting and ending wavelengths for resampling. If not provided, the
            wavelength bounds are inferred from the object's methods and the entire flux array is used.
        bin_edges : Iterable of float or None, optional
            Explicit bin edges in the wavelength space. If provided, `nbins` and `wlbounds` are ignored.

        Returns
        -------
            1D spectrum binned to the specified wavelength bins.
        """
        flux = spectrum.flux.value
        ucty = spectrum.uncertainty.represent_as(VarianceUncertainty).array
        npix = flux.size
        nbins = npix if nbins is None else nbins
        if wlbounds is None:
            l1 = self._p2w(0) - self._p2w_dldx(0)
            l2 = self._p2w(npix) + self._p2w_dldx(npix)
        else:
            l1, l2 = wlbounds

        bin_edges_wav = bin_edges if bin_edges is not None else np.linspace(l1, l2, num=nbins + 1)
        bin_edges_pix = np.clip(self._w2p(bin_edges_wav) + 0.5, 0, npix - 1e-12)
        bin_edge_ix = np.floor(bin_edges_pix).astype(int)
        bin_edge_w = bin_edges_pix - bin_edge_ix
        bin_centers_wav = 0.5 * (bin_edges_wav[:-1] + bin_edges_wav[1:])
        flux_wl = np.zeros(nbins)
        ucty_wl = np.zeros(nbins)
        weights = np.zeros(npix)

        dldx = self._p2w_dldx(np.arange(npix))
        n = flux.sum() / (dldx * flux).sum()
        for i in range(nbins):
            i1, i2 = bin_edge_ix[i:i + 2]
            weights[:] = 0
            if i1 != i2:
                weights[i1 + 1:i2] = 1.0
                weights[i1] = 1 - bin_edge_w[i]
                weights[i2] = bin_edge_w[i + 1]
                flux_wl[i] = (weights[i1:i2 + 1] * flux[i1:i2 + 1] * dldx[i1:i2 + 1]).sum()
                ucty_wl[i] = (weights[i1:i2 + 1] * ucty[i1:i2 + 1] * dldx[i1:i2 + 1]).sum()
            else:
                flux_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * flux[i1] * dldx[i1]
                ucty_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * ucty[i1] * dldx[i1]
        flux_wl = (flux_wl * n) * spectrum.flux.unit
        ucty_wl = VarianceUncertainty(ucty_wl * n).represent_as(type(spectrum.uncertainty))
        return Spectrum1D(flux_wl, bin_centers_wav * u.angstrom, uncertainty=ucty_wl)

    def pix_to_wav(self, pix: ndarray |float) -> ndarray | float:
        return self._p2w(pix)

    def wav_to_pix(self, wav: ndarray | float) -> ndarray | float:
        return self._w2p(wav)

    @property
    def wcs(self):
        pixel_frame = cf.CoordinateFrame(1, "SPECTRAL", [0, ], axes_names=["x", ], unit=[u.pix])
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength", ], unit=[self.linelists[0]['wavelength'].unit])
        pipeline = [(pixel_frame, self.fitted_model), (spectral_frame, None)]
        self._wcs = wcs.WCS(pipeline)
        return self._wcs

    @property
    def fitted_model(self):
        return self._p2w

    def match_lines(self, upper_bound: float = 5):
        matched_lines_wav = []
        matched_lines_pix = []
        for iframe, tree in enumerate(self._trees):
            l, ix = tree.query(self._p2w(self.lines_pix[iframe])[:, None], distance_upper_bound=upper_bound)
            m = np.isfinite(l)
            matched_lines_wav.append(tree.data[ix[m], 0])
            matched_lines_pix.append(self.lines_pix[iframe][m])
        return np.concatenate(matched_lines_pix), np.concatenate(matched_lines_wav)

    def plot_lines(self, axes=None, figsize=None):
        if axes is None:
            fig, axes = subplots(len(self.arc_spectra), 1, figsize=figsize, sharex='all', constrained_layout=True, squeeze=False)
        else:
            fig = axes[0].figure
        for i, sp in enumerate(self.arc_spectra):
            axes[i,0].plot(sp.data / (1.2 * sp.data.max()))
            axes[i,0].vlines(self.lines_pix[i], 0.0, 1, alpha=0.1)
            axes[i,0].vlines(self.lines_pix[i], 0.9, 1)
            axes[i,0].autoscale(enable=True, axis='x', tight=True)
        setp(axes[-1], xlabel='Pixel')
        return fig

    def plot_solution(self, axes=None, figsize=None, model = None):
        if axes is None:
            fig, axes = subplots(len(self.arc_spectra), 1, figsize=figsize, sharex='all', constrained_layout=True, squeeze=False)
        else:
            fig = axes[0].figure

        model = model if model is not None else self.fitted_model

        for i, sp in enumerate(self.arc_spectra):
            axes[i,0].plot(model(sp.spectral_axis.value), sp.data / (1.2 * sp.data.max()))
            axes[i,0].vlines(self.lines_wav[i], 0.0, 1.0, alpha=0.3, ec='darkorange', zorder=0)
            axes[i,0].vlines(model(self.lines_pix[i]), 0.9, 1.0, alpha=1)
            axes[i,0].autoscale(enable=True, axis='x', tight=True)
        setp(axes[-1], xlabel=f'Wavelength [{self.linelists[0]["wavelength"].unit.to_string(format="latex")}]')
        return fig

    def plot_transforms(self, figsize=None):
        fig, axs = subplots(2, 2, figsize=figsize, constrained_layout=True, sharex='col')
        xpix = self.arc_spectra[0].spectral_axis.value
        xwav = np.linspace(*self.pix_to_wav(xpix[[0, -1]]), num=xpix.size)
        axs[0, 0].plot(xpix, self._p2w(xpix), 'k')
        axs[1, 0].plot(xpix[:-1], np.diff(self._p2w(xpix)) / np.diff(xpix), lw=4, c='k')
        axs[1, 0].plot(xpix, self._p2w_dldx(xpix), ls='--', lw=2, c='w')
        axs[0, 1].plot(xwav, self._w2p(xwav), 'k')
        axs[1, 1].plot(xwav[:-1], np.diff(self._w2p(xwav)) / np.diff(xwav), lw=4, c='k')
        axs[1, 1].plot(xwav, self._w2p_dxdl(xwav), ls='--', lw=2, c='w')
        setp(axs[1,0], xlabel='Pixel', ylabel=r'd$\lambda$/dx')
        setp(axs[0,0], ylabel=fr'$\lambda$ [{self.linelists[0]["wavelength"].unit}]')
        setp(axs[1,1], xlabel=fr'$\lambda$ [{self.linelists[0]["wavelength"].unit}]', ylabel=r'dx/d$\lambda$')
        setp(axs[0,1], ylabel='Pixel')
        axs[0,0].set_title('Pixel -> wavelength')
        axs[0,1].set_title('Wavelength -> pixel')
        fig.align_labels()
        return fig