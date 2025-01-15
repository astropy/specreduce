import warnings
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.modeling import models, Model, fitting
from astropy.nddata import StdDevUncertainty, NDData
from gwcs import coordinate_frames as cf
from gwcs import wcs
from matplotlib import cm
from matplotlib.pyplot import setp, subplots
from numpy.random import uniform
from scipy import optimize
from scipy.spatial import KDTree

from numpy import ndarray
from specutils import Spectrum1D

from specreduce.line_matching import find_arc_lines
from specreduce.lswavecal1d import WavelengthSolution1D


def diff_poly2d_x(m):
    coeffs = {}
    for n in m.param_names:
        ix, iy = int(n[1]), int(n[3])
        if ix > 0:
            coeffs[f"c{ix-1}_{iy}"] = ix*getattr(m, n).value
    return models.Polynomial2D(m.degree-1, **coeffs)


class WavelengthSolution2D(WavelengthSolution1D):
    def __init__(self, frames: Iterable[NDData], lamps: Iterable, wlbounds: tuple[float, float], wave_air: bool = False,
                 n_cd_samples: int = 10, cd_samples: Iterable[float] | None = None):
        super().__init__(None, lamps, wlbounds, wave_air)
        self.frames = frames
        self.lamps = lamps

        self.lines_pix_x: Iterable[ndarray] | None = None
        self.lines_pix_y: Iterable[ndarray] | None = None
        self._fitted_model: Model | None = None
        self._wcs: wcs.WCS | None = None

        self.nframes = len(frames)

        self._spectra: Iterable[Spectrum1D] | None = None

        if cd_samples is not None:
            self.cd_samples = np.array(cd_samples)
        else:
            self.cd_samples = np.round(np.arange(1, n_cd_samples + 1) * self.frames[0].shape[0] / (n_cd_samples + 1)).astype(
                int)
        self.ncd = self.cd_samples.size

        self._ref_pixel: tuple[float, float] | None = None
        self._shift = None

    def find_lines(self, fwhm: float):
        self.spectra = []
        lines_pix_x = []
        lines_pix_y = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i, d in enumerate(self.frames):
                self.spectra.append([])
                lines_pix_x.append([])
                lines_pix_y.append([])
                for s in self.cd_samples:
                    spectrum = Spectrum1D((d[s] - np.median(d)) * u.DN,
                                          uncertainty=d[s].uncertainty.represent_as(StdDevUncertainty))
                    lines = find_arc_lines(spectrum, fwhm)
                    lines_pix_x[i].append(lines['centroid'].value)
                    lines_pix_y[i].append(np.full(len(lines), s))
                    self.spectra[i].append(spectrum)
        self.lines_pix_x = [np.concatenate(lpx) for lpx in lines_pix_x]
        self.lines_pix_y = [np.concatenate(lpy) for lpy in lines_pix_y]

    def fit(self, ref_pixel: tuple[float, float],
            wl0: tuple[float, float] = (7000, 7600), dwl: tuple[float, float] = (2.4, 2.8),
            popsize: int = 30, max_distance: float = 100, workers: int = 1):

        self._ref_pixel = ref_pixel
        self._shift = (models.Shift(-ref_pixel[0]) & models.Shift(-ref_pixel[1]))
        model = self._shift | models.Polynomial2D(3)
        trees = [KDTree(l[:, None]) for l in self.lines_wav]

        xx = np.zeros(10)

        def minfun(x):
            xx[:-3] = x
            distance_sum = 0.0
            for j, t in enumerate(trees):
                distance_sum += np.clip(t.query(
                    model.evaluate(self.lines_pix_x[j], self.lines_pix_y[j], -ref_pixel[0], -ref_pixel[1], *xx)[:,
                    None])[0],
                                        0, max_distance).sum()
            return distance_sum

        bounds = np.array([wl0, dwl, [-1e-3, 1e-3], [-1e-5, 1e-5], [-1e-1, 1e-1], [-1e-4, 1e-4], [-1e-5, 1e-5]])
        res = optimize.differential_evolution(minfun, bounds, popsize=popsize, workers=1, updating='deferred')
        self._p2w = (self._shift |
                     models.Polynomial2D(model[-1].degree,
                                         **{model[-1].param_names[i]: res.x[i] for i in range(res.x.size)}))
        self._refine_fit()
        self._calculate_inverse()
        self._calculate_derivaties()

    def _refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0):
        mlines_px, mlines_py, mlines_wav = self._match_lines(match_distance_bound)
        model = self._shift | models.Polynomial2D(degree, **{n: getattr(self._p2w[-1], n).value for n in self._p2w[-1].param_names})
        model.offset_0.fixed = True
        model.offset_1.fixed = True
        fitter = fitting.LMLSQFitter()
        self._p2w = fitter(model, mlines_px, mlines_py, mlines_wav)

    def _match_lines(self, upper_bound: float = 5):
        matched_lines_wav = []
        matched_lines_pix_x = []
        matched_lines_pix_y = []
        for iframe, tree in enumerate(self._trees):
            l, ix = tree.query(self._p2w(self.lines_pix_x[iframe], self.lines_pix_y[iframe])[:, None], distance_upper_bound=upper_bound)
            m = np.isfinite(l)
            matched_lines_wav.append(tree.data[ix[m], 0])
            matched_lines_pix_x.append(self.lines_pix_x[iframe][m])
            matched_lines_pix_y.append(self.lines_pix_y[iframe][m])
        return (np.concatenate(matched_lines_pix_x),
                np.concatenate(matched_lines_pix_y),
                np.concatenate(matched_lines_wav))

    def _calculate_inverse(self, nsamples: int = 1500):
        m = self._p2w
        xx = uniform(0, self.frames[0].shape[1], nsamples)
        yy = uniform(0, self.frames[0].shape[0], nsamples)
        ll = m(xx, yy)
        mm = (models.Shift(-m.c0_0_2) & models.Shift(-self._ref_pixel[1])) | models.Polynomial2D(6, c0_0=-m[0].offset,
                                                                                           c1_0=1 / m[-1].c1_0,
                                                                                           fixed={'c0_0': True})
        mm.offset_0.fixed = True
        mm.offset_1.fixed = True
        self._w2p = fitting.LMLSQFitter()(mm, ll, yy, xx)

    def _calculate_derivaties(self):
        if self._p2w is not None:
            self._p2w_dldx = self._shift | diff_poly2d_x(self._p2w[-1])
        if self._w2p is not None:
            self._w2p_dxdl = (self._w2p[0] & self._w2p[1]) | diff_poly2d_x(self._w2p[-1])

    def resample(self, flux, nbins: int | None = None, wlbounds: tuple[float, float] | None = None,
                 bin_edges: Iterable[float] | None = None):
        ny, nx = flux.shape
        ypix = np.arange(ny)
        nbins = nx if nbins is None else nbins
        if wlbounds is None:
            l1 = self._p2w(0, 0) - self._p2w_dldx(0, 0)
            l2 = self._p2w(nx, 0) + self._p2w_dldx(nx, 0)
        else:
            l1, l2 = wlbounds

        bin_edges_wav = bin_edges if bin_edges is not None else np.linspace(l1, l2, num=nbins + 1)
        bin_edges_pix = np.clip(self._w2p(*np.meshgrid(bin_edges_wav, ypix)) + 0.5, 0, nx - 1e-12)
        bin_edge_ix = np.floor(bin_edges_pix).astype(int)
        bin_edge_w = bin_edges_pix - bin_edge_ix
        bin_centers_wav = 0.5 * (bin_edges_wav[:-1] + bin_edges_wav[1:])

        flux_wl = np.zeros((ny, nbins))
        weights = np.zeros((ny, nx))

        dldx = self._p2w_dldx(*np.meshgrid(np.arange(nx), np.arange(ny)))
        n = flux.sum(1) / (dldx * flux).sum(1)

        ixs = np.tile(np.arange(flux.shape[1]), (flux.shape[0], 1))
        ys = np.arange(flux.shape[0])

        for i in range(nbins):
            i1, i2 = bin_edge_ix[:, i:i + 2].T
            m = i1 == i2
            if m.any():
                flux_wl[:, i] = (bin_edges_pix[:, i + 1] - bin_edges_pix[:, i]) * flux[ys, i1] * dldx[ys, i1]

            if not m.all():
                imin, imax = i1.min(), i2.max() + 1
                ixc = ixs[:, imin: imax]
                w = weights[:, imin:imax]
                w[:] = 0.0
                w[(ixc > i1[:, None]) & (ixc < i2[:, None])] = 1
                w[ys, i1 - imin] = 1.0 - bin_edge_w[:, i]
                w[ys, i2 - imin] = bin_edge_w[:, i + 1]
                flux_wl[~m, i] = (flux[~m, imin:imax] * dldx[~m, imin:imax] * w[~m]).sum(1)
        flux_wl *= n[:, None]
        return bin_centers_wav, flux_wl

    def plot_fit(self, lamp: int, ax=None, figsize=None):
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig = ax.figure

        i = 0
        ncd = len(self.cd_samples)

        t = self._trees[lamp]
        l, ix = t.query(self._p2w(self.lines_pix_x[lamp], self.lines_pix_y[lamp])[:, None], distance_upper_bound=10)
        mask = np.zeros(t.data.size, bool)
        mask[ix[np.isfinite(l)]] = True

        ax.vlines(self.lines_wav[lamp][mask], -1.0, 8.0, alpha=1, ec='darkorange', zorder=-1)
        ax.vlines(self.lines_wav[lamp][~mask], -1.0, 8.0, alpha=0.1, ec='k', zorder=-2)

        for i in range(ncd):
            x = self.spectra[lamp][i].spectral_axis.value
            sl = self._p2w(x, np.full_like(x, self.cd_samples[i]))
            ax.pcolormesh(np.tile(sl, (2, 1)),
                          np.array([np.full_like(sl, i), np.full_like(sl, i + 1)]),
                          self.spectra[lamp][i].data[None, :-1], cmap=cm.Blues,
                          vmax=np.percentile(self.spectra[lamp][i].data, 98))
            ax.axhline(i, c='k', alpha=0.5, ls='-', lw=0.5)

        setp(ax, ylim=(-1.0, ncd + 1), yticks=0.5 + np.arange(ncd), yticklabels=self.cd_samples)
        return fig

    def plot_residuals(self, axes=None, model=None, figsize=None):
        if axes is None:
            fig, axes = subplots(self.nframes, 1, figsize=figsize, constrained_layout=True, sharex='all', sharey='all')
        else:
            fig = axes[0].figure
        model = model if model is not None else self._p2w
        trees = [KDTree(l[:, None]) for l in self.lines_wav]
        for lamp, t in enumerate(trees):
            l, ix = t.query(model(self.lines_pix_x[lamp], self.lines_pix_y[lamp])[:, None], distance_upper_bound=5)
            axes[lamp].plot(self.lines_pix_x[lamp], l, '.')
