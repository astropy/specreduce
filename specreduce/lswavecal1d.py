import warnings
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.modeling import models, Model, fitting
from gwcs import coordinate_frames as cf
from gwcs import wcs
from matplotlib.pyplot import setp, subplots
from scipy import optimize
from scipy.spatial import KDTree

from numpy import ndarray

from specreduce.calibration_data import load_pypeit_calibration_lines
from specreduce.line_matching import find_arc_lines


def diff_poly1d(m: models.Polynomial1D) -> models.Polynomial1D:
    coeffs = {f'c{i-1}': i*getattr(m, f'c{i}').value for i in range(1, m.degree+1)}
    return models.Polynomial1D(m.degree-1, **coeffs)


class WavelengthSolution1D:
    def __init__(self, arc_spectra, lamps, wlbounds: tuple[float, float], wave_air: bool = False):
        self.arc_spectra = arc_spectra
        self.lamps = lamps
        self.wlbounds = wlbounds
        self.wave_air: bool = wave_air
        self.linelists: list | None = []
        self.lines_pix: ndarray | None = None
        self.lines_wav: ndarray | None = None
        self._fitted_model: Model | None = None
        self._fit: optimize.OptimizeResult | None = None
        self._wcs: wcs.WCS | None = None
        self._trees: Iterable[KDTree] | None = None
        self._p2w: Model | None = None           # The fitted pixel -> wavelength model
        self._w2p: Model | None = None           # The fitted wavelength -> pixel model
        self._p2w_dldx: Model | None = None      # delta lambda / delta pixel
        self._w2p_dxdl: Model | None = None      # delta pixel / delta lambda

        self._read_linelists()

    def _read_linelists(self):
        for l in self.lamps:
            ll = load_pypeit_calibration_lines(l, wave_air=self.wave_air)
            self.linelists.append(ll[(ll['wavelength'].value > self.wlbounds[0]) &
                                     (ll['wavelength'].value < self.wlbounds[1])])
        self.lines_wav = [lo['wavelength'].value for lo in self.linelists]
        self._trees = [KDTree(l[:, None]) for l in self.lines_wav]

    def find_lines(self, fwhm: float):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lines_obs = [find_arc_lines(sp, fwhm) for sp in self.arc_spectra]
        self.lines_pix = [lo['centroid'].value for lo in lines_obs]

    def fit(self, ref_pixel: float, wl0: tuple[float, float], dwl: tuple[float, float],
            popsize: int = 30, max_distance: float = 100):

        model = models.Shift(-ref_pixel) | models.Polynomial1D(3)
        def minfun(x):
            return sum(
                [np.clip(t.query(model.evaluate(l, -ref_pixel, *x)[:, None])[0], 0, max_distance).sum() for t, l in
                 zip(self._trees, self.lines_pix)])

        # Calculate the pixel -> wavelength transform and its derivative along the spectral axis
        self._fit = fit = optimize.differential_evolution(minfun,
                                                          bounds=[wl0, dwl, [-1e-3, 1e-3], [-1e-4, 1e-4]],
                                                          popsize=popsize)
        self._p2w = models.Shift(-ref_pixel) | models.Polynomial1D(3, **{f'c{i}': fit.x[i] for i in
                                                                                  range(fit.x.size)})
        self.refine_fit()
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

    def refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0):
        matched_pix, matched_wav = self.match_lines(match_distance_bound)
        model = models.Polynomial1D(degree, **{n: getattr(self._p2w[-1], n).value for n in self._p2w[-1].param_names})
        fitter = fitting.LinearLSQFitter()
        self._p2w = self._p2w[0].copy() | fitter(model, matched_pix + self._p2w.offset_0.value, matched_wav)

    def resample(self, flux, nbins: int | None = None, wlbounds: tuple[float, float] | None = None,
                 bin_edges: Iterable[float] | None = None):
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
            else:
                flux_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * flux[i1] * dldx[i1]
        flux_wl *= n
        return bin_centers_wav, flux_wl

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