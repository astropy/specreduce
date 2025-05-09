import warnings
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.modeling import models, Model, fitting
from astropy.nddata import StdDevUncertainty, NDData
from scipy.optimize import minimize
from scipy.spatial import KDTree

from numpy import ndarray, concatenate, repeat, tile
from specutils import Spectrum1D

from specreduce.line_matching import find_arc_lines
from specreduce.compat import Spectrum


def diff_poly2d_x(m):
    coeffs = {}
    for n in m.param_names:
        ix, iy = int(n[1]), int(n[3])
        if ix > 0:
            coeffs[f"c{ix-1}_{iy}"] = ix * getattr(m, n).value
    return models.Polynomial2D(m.degree - 1, **coeffs)


class TiltCorrection:
    def __init__(
        self,
        ref_pixel: tuple[float, float],
        arc_frames: Sequence[NDData],
        n_cd_samples: int = 10,
        cd_sample_lims: tuple[float, float] | None = None,
        cd_samples: Sequence[float] | None = None,
    ):
        self.ref_pixel = ref_pixel
        self.arc_frames = arc_frames
        self.nframes = len(arc_frames)
        self.n_cd_pix = self.arc_frames[0].data.shape[0]

        self._lines_ref: Sequence[ndarray] | None = None
        self._samples_rec_x: Sequence[ndarray] | None = None
        self._samples_rec_y: Sequence[ndarray] | None = None
        self._samples_det_x: Sequence[ndarray] | None = None
        self._samples_det_y: Sequence[ndarray] | None = None
        self._trees: Sequence[KDTree] | None = None
        self._spectra: Sequence[Spectrum1D] | None = None

        # The rectified space -> detectoir space transform
        self._r2d: Model | None = None

        if cd_samples is not None:
            self.cd_samples = np.array(cd_samples)
        else:
            self.cd_samples = np.round(
                np.arange(1, n_cd_samples + 1) * self.n_cd_pix / (n_cd_samples + 1)
            ).astype(int)
        self.ncd = self.cd_samples.size
        self._shift = None

    def find_arc_lines(self, fwhm: float, noise_factor: float = 5.0):
        self._spectra = []
        self._samples_rec_x = []
        self._lines_ref = []
        samples_x = []
        samples_y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i, d in enumerate(self.arc_frames):
                self._spectra.append([])
                samples_x.append([])
                samples_y.append([])

                # Find the line centroids for the reference row
                spectrum = Spectrum(
                    d.data[self.ref_pixel[1]] * d.unit,
                    uncertainty=d[self.ref_pixel[1]].uncertainty.represent_as(StdDevUncertainty),
                )
                lines = find_arc_lines(spectrum, fwhm, noise_factor=noise_factor)
                self._lines_ref.append(lines["centroid"].value)

                # Find the line centroids for the sample rows
                for s in self.cd_samples:
                    spectrum = Spectrum(
                        d.data[s] * d.unit,
                        uncertainty=d[s].uncertainty.represent_as(StdDevUncertainty),
                    )
                    lines = find_arc_lines(spectrum, fwhm, noise_factor=noise_factor)
                    samples_x[i].append(lines["centroid"].value)
                    samples_y[i].append(np.full(len(lines), s))
                    self._spectra[i].append(spectrum)

        self._samples_det_x = [np.concatenate(lpx) for lpx in samples_x]
        self._samples_det_y = [np.concatenate(lpy) for lpy in samples_y]
        self._samples_rec_y = [repeat(self.cd_samples, lref.size) for lref in self._lines_ref]
        self._samples_rec_x = [tile(lref, self.cd_samples.size) for lref in self._lines_ref]

        self._trees = [
            KDTree(np.vstack([lx, ly]).T) for lx, ly in zip(self._samples_det_x, self._samples_det_y)
        ]

    def fit(self):
        self._shift = models.Shift(-self.ref_pixel[0]) & models.Shift(-self.ref_pixel[1])
        model = self._shift | models.Polynomial2D(3)

        coeffs = np.zeros(10)
        coeffs[0] = self.ref_pixel[0]
        coeffs[1] = 1
        transformed_points = [tile(a, (2, 1)).T.astype("d") for a in self._samples_rec_y]

        def minfun(x):
            coeffs[4:] = x
            distance_sum = 0.0
            for i, t in enumerate(self._trees):
                transformed_points[i][:, 0] = model.evaluate(
                    self._samples_rec_x[i], self._samples_rec_y[i], -self.ref_pixel[0], -self.ref_pixel[1], *coeffs
                )
                distance_sum += t.query(transformed_points[i])[0].sum()
            return distance_sum

        res = minimize(minfun, np.zeros(6), method="powell")
        coeffs[4:] = res.x
        self._r2d = self._shift | models.Polynomial2D(
            model[-1].degree, **{model[-1].param_names[i]: coeffs[i] for i in range(coeffs.size)}
        )

        self._calculate_derivative()
        # self._refine_fit()

    def _refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0):
        rx, ry, ox = self._match_lines(match_distance_bound)
        model = self._shift | models.Polynomial2D(
            degree, **{n: getattr(self._r2d[-1], n).value for n in self._r2d[-1].param_names}
        )
        model.offset_0.fixed = True
        model.offset_1.fixed = True
        for i in range(degree + 1):
            model.fixed[f"c{i}_0_2"] = True

        fitter = fitting.LMLSQFitter()
        self._r2d = fitter(model, rx, ry, ox)
        self._calculate_derivative()

    def _match_lines(self, upper_bound: float = 5):
        matched_lines_obs = []
        matched_lines_ref_x = []
        matched_lines_ref_y = []
        for iframe, tree in enumerate(self._trees):
            x_mapped = self._r2d(self._samples_rec_x[iframe], self._samples_rec_y[iframe])
            l, ix = tree.query(
                np.array([x_mapped, self._samples_rec_y[iframe]]).T, distance_upper_bound=upper_bound
            )
            m = np.isfinite(l)
            matched_lines_obs.append(tree.data[ix[m], 0])
            matched_lines_ref_x.append(self._samples_rec_x[iframe][m])
            matched_lines_ref_y.append(self._samples_rec_y[iframe][m])
        return (
            np.concatenate(matched_lines_ref_x),
            np.concatenate(matched_lines_ref_y),
            np.concatenate(matched_lines_obs),
        )

    def _calculate_derivative(self):
        """Calculate the derivative for the rectified space -> detector space transformation.
        """
        if self._r2d is not None:
            self._r2d_dxdx = self._shift | diff_poly2d_x(self._r2d[-1])

    def rectify(
        self,
        flux: ndarray,
        nbins: int | None = None,
        bounds: tuple[float, float] | None = None,
        bin_edges: Iterable[float] | None = None,
    ):
        """Resample a 2D spectrum from the detector space to a rectified space.

        Resample a 2D spectrum from the detector space to a rectified space where the wavelength
        is constant along the rows. The grid edges are based on the specified number of bins,
        bounds, or bin edges. The resampling is eaxct and conserves flux (as long as the
        rectified space covers the whole detector space.)

        Parameters
        ----------
        flux
            2D array representing the flux values of the distorted input image. The first
            dimension corresponds to rows (typically the y-axis), and the second dimension
            corresponds to columns (typically the x-axis).
        nbins
            Number of bins in the rectified space. If None, the number of bins will be set
            to the number of columns in the `flux` input image.
        bound
            Tuple specifying the start and end coordinates for the rectified space along the
            x-axis. If None, the bounds default to (0, number of columns in `flux`).
        bin_edges
            Explicitly provided edges of the bins in the rectified space. If None, bin
            edges are automatically calculated as a uniform grid based on `nbins` and
            `bounds`.

        Returns
        -------
        rectified_flux : ndarray
            2D array containing the flux values rectified into the uniform grid defined
            by `nbins`, `bounds`, or `bin_edges`. The output has the same number of rows
            as the input `flux`, and its second dimension corresponds to the number of
            rectified bins.
        """
        ny, nx = flux.shape
        ypix = np.arange(ny)
        nbins = nx if nbins is None else nbins
        l1, l2 = bounds if bounds is not None else (0, nx)

        bin_edges_rec = bin_edges if bin_edges is not None else np.linspace(l1, l2, num=nbins + 1)
        bin_edges_det = np.clip(self._r2d(*np.meshgrid(bin_edges_rec, ypix)) , 0, nx - 1e-12)
        bin_edge_ix = np.floor(bin_edges_det).astype(int)
        bin_edge_w = bin_edges_det - bin_edge_ix

        rectified_flux = np.zeros((ny, nbins))
        weights = np.zeros((ny, nx))

        # Calculate the derivative of the rectified space -> detector space transformation  with
        # respect to the detector coordinate (dx_rec / dx_det). This is needed for flux
        # conservation, as it represents how the pixel width changes.
        dtdx = self._r2d_dxdx(*np.meshgrid(np.arange(nx), np.arange(ny)))

        # Calculate a normalization factor 'n' for flux conservation. This factor accounts for the
        # change in pixel size due to the distortion, and ensures that the total flux in each row
        # is conserved after rectification
        n = flux.sum(1) / (dtdx * flux).sum(1)

        ixs = np.tile(np.arange(flux.shape[1]), (flux.shape[0], 1))
        ys = np.arange(flux.shape[0])

        for i in range(nbins):
            # Get the detector pixel indices (left and right edges) for the current rectified bin.
            i1, i2 = bin_edge_ix[:, i : i + 2].T

            # Create a mask 'm' where the left and right detector pixel edges are the same.
            # This means the entire rectified bin falls within a single detector pixel.
            m = i1 == i2

            # For rows where the rectified bin falls within a single detector pixel,
            # the rectified flux is the detector flux in that pixel, scaled by the width of the
            # rectified bin in detector coordinates and the derivative dtdx.
            if m.any():
                rectified_flux[:, i] = (
                    (bin_edges_det[:, i + 1] - bin_edges_det[:, i]) * flux[ys, i1] * dtdx[ys, i1]
                )

            # For rows where the rectified bin spans multiple detector pixels, calculate the
            # rectified flux as a weighted sum of the detector flux, multiplied by dtdx,
            # within the span [imin, imax].
            if not m.all():
                imin, imax = i1.min(), i2.max() + 1
                ixc = ixs[:, imin:imax]
                w = weights[:, imin:imax]
                w[:] = 0.0
                w[(ixc > i1[:, None]) & (ixc < i2[:, None])] = 1
                w[ys, i1 - imin] = 1.0 - bin_edge_w[:, i]
                w[ys, i2 - imin] = bin_edge_w[:, i + 1]
                rectified_flux[~m, i] = (flux[~m, imin:imax] * dtdx[~m, imin:imax] * w[~m]).sum(1)

        # Apply the normalization factor to conserve flux
        rectified_flux *= n[:, None]
        return rectified_flux

    def plot_transform(self, frame: int = 0, nx: int = 50, ny: int = 100, ax=None, figsize=None, plot_lines: bool = False, cmap=None, vmax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        d = self.arc_frames[frame]
        if plot_lines:
            nx = self._lines_ref[frame].size
            xs = tile(self._lines_ref[frame], (ny, 1))
            ys = tile(np.linspace(0, d.shape[0], ny)[:,None], (1, nx))
        else:
            xs = tile(np.linspace(0, d.shape[1], nx), (ny, 1))
            ys = tile(np.linspace(0, d.shape[0], ny)[:,None], (1, nx))
        ax.imshow(d.data, aspect='auto', origin='lower', cmap=cmap, vmax=vmax)
        ax.plot(self._r2d(xs, ys), ys, 'w--', lw=1, alpha=1)
        plt.setp(ax, xlim=(0, d.shape[1]), ylim=(0, d.shape[0]))
        return fig

    def plot_residuals(self, axes=None, model=None, figsize=None):
        if axes is None:
            fig, axes = plt.subplots(
                self.nframes,
                1,
                figsize=figsize,
                constrained_layout=True,
                sharex="all",
                sharey="all",
            )
        else:
            fig = axes[0].figure
        model = model if model is not None else self._p2w
        trees = [KDTree(l[:, None]) for l in self.lines_wav]
        for lamp, t in enumerate(trees):
            l, ix = t.query(
                model(self._samples_det_x[lamp], self._samples_det_y[lamp])[:, None], distance_upper_bound=5
            )
            axes[lamp].plot(self._samples_det_x[lamp], l, ".")