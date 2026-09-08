import warnings
from typing import Sequence, Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from astropy.nddata import StdDevUncertainty, NDData
from numpy import ndarray, repeat, tile
from scipy.optimize import minimize
from scipy.spatial import KDTree
from specutils import Spectrum

from specreduce.core import parse_image
from specreduce.line_matching import find_arc_lines
from specreduce.tilt_solution import TiltSolution
from specreduce.tracing import Trace

__all__ = ["TiltCorrection"]


class TiltCorrection:
    def __init__(
        self,
        arc_frames: NDData | Sequence[NDData],
        trace: Trace | None = None,
        cdisp_ref_pixel: float | None = None,
        disp_ref_pixel: float | None = None,
        n_cdisp_samples: int = 10,
        cdisp_sample_lims: tuple[float, float] | None = None,
        cdisp_samples: Sequence[float] | None = None,
        disp_axis: int = 1,
        mask_treatment: Literal[
            "apply",
            "ignore",
            "propagate",
            "zero_fill",
            "nan_fill",
            "apply_mask_only",
            "apply_nan_only",
        ] = "apply",
    ):
        """A class for 2D spectral tilt correction.

        This class provides tools for correcting spectral tilt (curvature) in 2D
        spectroscopic data using arc lamp frames. It identifies arc lines across the
        cross-dispersion axis and fits a 2D polynomial transformation from a
        tilt-corrected (rectified) coordinate space to detector space. The resulting
        `~specreduce.tilt_solution.TiltSolution` can then be used to resample science
        frames onto a rectified grid.

        Parameters
        ----------
        arc_frames
            A sequence of arc frames as `~astropy.nddata.NDData` instances.
        trace
            A trace object representing the spectrum trace. If provided, it will be used to
            determine the reference positions along the dispersion and cross-dispersion axes.
        cdisp_ref_pixel
            A reference pixel position along the cross-dispersion axis. Should be close to the
            spectrum trace's average cross-dispersion position for the best results.
        disp_ref_pixel
            A reference pixel position along the dispersion axis. Should be close to the
            center of the frame along the dispersion axis for best results.
        n_cdisp_samples
            Number of cross-dispersion (CD) samples to use.
        cdisp_sample_lims
            Tuple specifying the limits for calculating cross-dispersion sampling.
        cdisp_samples
            A list of cross-dispersion locations to use. Overrides ``n_cdisp_samples``
            if provided.
        disp_axis
            The index of the image's dispersion axis.
        mask_treatment
            Specifies how to handle masked or non-finite values in the input image.
            The accepted values are:

            - ``apply``: The image remains unchanged, and any existing mask is combined with a mask
              derived from non-finite values.
            - ``ignore``: The image remains unchanged, and any existing mask is dropped.
            - ``propagate``: The image remains unchanged, and any masked or non-finite pixel
              causes the mask to extend across the entire cross-dispersion axis.
            - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,
              and the mask is dropped.
            - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,
              and the mask is dropped.
            - ``apply_mask_only``: The  image and mask are left unmodified.
            - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped,
              and a new mask is created based on non-finite values.
        """
        self.disp_axis = disp_axis
        self.mask_treatment = mask_treatment

        # IMPLEMENTATION NOTES: the code assumes that the image-parsing routines ensure that the
        # cross-dispersion axis is aligned with the y-axis (1st array dimension) and the
        # dispersion axis with the x-axis (2nd array dimension). However, this should not be
        # visible to the end-user. The rectified spectra are returned with the original axis
        # alignment given by the ``disp_axis`` argument. Also, I've decided to use `x` and `y`
        # naming instead of `col` and `row` because this leads to (slightly) more readable code.
        # The methods that are visible to the end-user use `disp` and `cdisp` naming. -HP

        if not isinstance(arc_frames, Sequence):
            arc_frames = [arc_frames]

        self.arc_frames = []
        for f in arc_frames:
            im = parse_image(f, disp_axis=disp_axis, mask_treatment=mask_treatment)
            self.arc_frames.append(NDData(im.flux, uncertainty=im.uncertainty, mask=im.mask))
        self.nframes = len(arc_frames)
        self._ny, self._nx = self.arc_frames[0].data.shape

        self._lines_ref: Sequence[ndarray] | None = None
        self._samples_rec_x: Sequence[ndarray] | None = None
        self._samples_rec_y: Sequence[ndarray] | None = None
        self._samples_det_x: Sequence[ndarray] | None = None
        self._samples_det_y: Sequence[ndarray] | None = None
        self._arc_spectra: Sequence[Spectrum] | None = None
        self._trees: Sequence[KDTree] | None = None

        if trace is not None:
            self.trace = trace
            disp_ref_pixel = self.trace.trace.size // 2
            cdisp_ref_pixel = int(self.trace.trace[disp_ref_pixel])
        else:
            if cdisp_ref_pixel is None:
                raise ValueError("cdisp_ref_position must be provided if trace is not provided.")
            if disp_ref_pixel is None:
                disp_ref_pixel = self.arc_frames[0].data.shape[disp_axis] // 2

        self.ref_pixel = (cdisp_ref_pixel, disp_ref_pixel)  # Reference pixel (y, x)
        self._shift = models.Shift(-self.ref_pixel[1]) & models.Shift(-self.ref_pixel[0])

        # Calculate the cross-dispersion axis sample positions
        slims = cdisp_sample_lims if cdisp_sample_lims is not None else (0, self._ny)
        if cdisp_samples is not None:
            self.cd_samples = np.array(cdisp_samples)
        else:
            self.cd_samples = slims[0] + np.round(
                np.arange(1, n_cdisp_samples + 1) * (slims[1] - slims[0]) / (n_cdisp_samples + 1)
            ).astype(int)
        self.ncd = self.cd_samples.size

        self.solution: TiltSolution | None = None

    def find_arc_lines(
        self,
        fwhm: float,
        noise_factor: float = 5.0,
        subtract_baseline: bool = True,
        baseline_window: int | None = None,
    ) -> None:
        """Find arc lines from the provided arc frames for all cross-dispersion samples.

        This method locates spectral arc lines from the provided arc frames, calculates
        their centroids, and organizes them into reference lists and sample arrays
        for further analysis.

        The arc frames are used as they are, so by default the baseline (background)
        flux of each row is estimated with a sigma-clipped median and removed before
        the line detection. The detection thresholds the flux against
        ``noise_factor × uncertainty``, so a pedestal above that level would otherwise
        swallow all the lines.

        Parameters
        ----------
        fwhm
            Full width at half maximum of the spectral line to be detected, used
            by the line-finding algorithm.
        noise_factor
            A multiplier for noise thresholding in the line-finding process.
        subtract_baseline
            Estimate and subtract the baseline flux of each row before the line
            detection. Set to ``False`` for arc frames that are already
            background-subtracted.
        baseline_window
            Width in pixels of the chunks used for a running baseline estimate that
            can follow a slowly varying background. If ``None``, a single global
            median is subtracted from each row.
        """
        self._arc_spectra = []
        self._samples_rec_x = []
        self._lines_ref = []
        samples_x = []
        samples_y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i, d in enumerate(self.arc_frames):
                self._arc_spectra.append([])
                samples_x.append([])
                samples_y.append([])

                # Find the line centroids for the reference row
                spectrum = Spectrum(
                    d.data[self.ref_pixel[0]] * d.unit,
                    uncertainty=d.uncertainty[self.ref_pixel[0]].represent_as(StdDevUncertainty),
                )
                lines = find_arc_lines(
                    spectrum,
                    fwhm,
                    noise_factor=noise_factor,
                    subtract_baseline=subtract_baseline,
                    baseline_window=baseline_window,
                )
                self._lines_ref.append(lines["centroid"].value)

                # Find the line centroids for the sample rows
                for s in self.cd_samples:
                    spectrum = Spectrum(
                        d.data[s] * d.unit,
                        uncertainty=d.uncertainty[s].represent_as(StdDevUncertainty),
                    )
                    lines = find_arc_lines(
                        spectrum,
                        fwhm,
                        noise_factor=noise_factor,
                        subtract_baseline=subtract_baseline,
                        baseline_window=baseline_window,
                    )
                    samples_x[i].append(lines["centroid"].value)
                    samples_y[i].append(np.full(len(lines), s))
                    self._arc_spectra[i].append(spectrum)

        self._samples_det_x = [np.concatenate(lpx) for lpx in samples_x]
        self._samples_det_y = [np.concatenate(lpy) for lpy in samples_y]
        self._samples_rec_y = [repeat(self.cd_samples, lref.size) for lref in self._lines_ref]
        self._samples_rec_x = [tile(lref, self.cd_samples.size) for lref in self._lines_ref]

        self._trees = [
            KDTree(np.vstack([lx, ly]).T)
            for lx, ly in zip(self._samples_det_x, self._samples_det_y)
        ]

    def fit(
        self, degree: int = 3, method: str = "Powell", max_distance: float = 10
    ) -> TiltSolution:
        """Fit a 2D polynomial transformation from tilt-corrected space to detector space.

        The transformation is calculated by minimizing the sum of distances between transformed
        samples and their corresponding detector-space targets. The minimization is performed in
        two stages: an initial minimization of a kd-tree based sum of line-line distances using
        `scipy.optimize.minimize` and a refinement using least-squares optimization of matched
        lines.

        Parameters
        ----------
        degree
            The degree of the final 2D polynomial model.
        method
            The optimization method used during the initial fitting stage.
        max_distance
            The maximum allowable distance to constrain the minimization.
        """
        model = self._shift | models.Polynomial2D(3)

        coeffs = np.zeros(10)
        coeffs[0] = self.ref_pixel[1]
        coeffs[1] = 1
        transformed_points = [tile(a, (2, 1)).T.astype("d") for a in self._samples_rec_y]

        def minfun(x):
            coeffs[4:] = x
            total_distance = 0.0
            for i, t in enumerate(self._trees):
                transformed_points[i][:, 0] = model.evaluate(
                    self._samples_rec_x[i],
                    self._samples_rec_y[i],
                    -self.ref_pixel[1],
                    -self.ref_pixel[0],
                    *coeffs,
                )
                total_distance += np.clip(t.query(transformed_points[i])[0], 0, max_distance).sum()
            return total_distance

        res = minimize(minfun, np.zeros(6), method=method)
        coeffs[4:] = res.x

        self.solution = TiltSolution(
            self._shift
            | models.Polynomial2D(
                model[-1].degree,
                **{model[-1].param_names[i]: coeffs[i] for i in range(coeffs.size)},
            ),
            image_shape=(self._ny, self._nx),
        )

        # Calculate the final fit using least-squares optimization between matched lines
        self.refine_fit(degree)
        return self.solution

    def refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0) -> None:
        """Refine the tilt-corrected space -> detector space transformation model parameters.

        Refines the polynomial fit model parameters for matching features with a specified
        degree and match distance bound. The refinement includes matching lines, updating a
        polynomial model, and optimizing the parameters using a least squares fitter
        The derivative is recalculated after the optimization.

        Parameters
        ----------
        degree
            Degree of the polynomial used in the Polynomial2D model.
        match_distance_bound
            Maximum acceptable distance between features to be considered a match.
        """
        if self.solution is None:
            raise ValueError("The solution must be calculated before it can be refined.")

        rx, ry, ox = self.match_lines(match_distance_bound)
        model = self._shift | models.Polynomial2D(
            degree,
            **{
                n: getattr(self.solution.c2d[-1], n).value
                for n in self.solution.c2d[-1].param_names
            },
        )
        model.offset_0.fixed = True
        model.offset_1.fixed = True
        for i in range(degree + 1):
            model.fixed[f"c{i}_0_2"] = True

        fitter = fitting.LMLSQFitter()
        self.solution.c2d = fitter(model, rx, ry, ox)

    def match_lines(
        self, max_distance: float = 5, concatenate: bool = True
    ) -> tuple[ndarray, ndarray, ndarray] | tuple[list[ndarray], list[ndarray], list[ndarray]]:
        """Match the reference arc line locations with the detector-space targets.

        Parameters
        ----------
        max_distance
            Specifies the maximum allowed distance for matching lines. Matches beyond this distance
            will be ignored.
        concatenate
            Specifies whether to concatenate the matched lines.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing three concatenated numpy arrays representing:
            - x-coordinates of matched rectified-space lines.
            - y-coordinates of matched rectified-space lines.
            - x-coordinates of matched detector-space lines.
        """

        if self.solution is None:
            raise ValueError("The solution must be calculated before line matching.")

        matched_det_x = []
        matched_rec_x = []
        matched_rec_y = []
        for iframe, tree in enumerate(self._trees):
            x_mapped = self.solution.c2d(self._samples_rec_x[iframe], self._samples_rec_y[iframe])
            l, ix = tree.query(
                np.array([x_mapped, self._samples_rec_y[iframe]]).T,
                distance_upper_bound=max_distance,
            )
            m = np.isfinite(l)
            matched_det_x.append(tree.data[ix[m], 0])
            matched_rec_x.append(self._samples_rec_x[iframe][m])
            matched_rec_y.append(self._samples_rec_y[iframe][m])

        if concatenate:
            return (
                np.concatenate(matched_rec_x),
                np.concatenate(matched_rec_y),
                np.concatenate(matched_det_x),
            )
        else:
            return matched_rec_x, matched_rec_y, matched_det_x

    def plot_wavelength_contours(
        self,
        n_disp: int = 50,
        n_cdisp: int = 100,
        disp_values: Sequence[float] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        line_args: dict | None = None,
    ):
        """Plot wavelength contour lines in detector space.

        Parameters
        ----------
        n_disp
            The number of dispersion-axis lines.
        n_cdisp
            The number of cross-dispersion axis points for each disp-axis line.
        disp_values
            A sequence specifying the dispersion-axis coordinates explicitly. If not
            provided, it will be automatically calculated based on the arc frame
            dimensions.
        ax
            The Matplotlib Axes on which to plot. If None, a new figure and Axes
            are created.
        figsize
            Tuple specifying the size of the figure to create, applicable only if
            ``ax`` is None.
        line_args
            A dictionary of line properties (e.g., color, linewidth, linestyle).
            These properties modify the default styling provided for grid lines.
            If None, default styles are used. Default is None.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The Matplotlib figure containing the plot. If an Axes instance is passed
            to ``ax``, the associated figure is returned.
        """
        largs = {"c": "k", "lw": 0.5, "alpha": 0.5, "ls": "--"}
        if line_args is not None:
            largs.update(line_args)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if disp_values is None:
            disp_values = tile(
                np.linspace(0, self.arc_frames[0].data.shape[1], n_disp), (n_cdisp, 1)
            )
        else:
            n_disp = len(disp_values)
        rows = tile(np.linspace(0, self.arc_frames[0].data.shape[0], n_cdisp)[:, None], (1, n_disp))

        ax.plot(self.solution.c2d(disp_values, rows), rows, **largs)
        return fig

    def plot_fit_quality(
        self, figsize=None, max_match_distance: float = 5, rlim: tuple[float, float] | None = None
    ):
        """Plot fit quality diagnostics showing residuals of the tilt correction.

        Creates a three-panel figure with a scatter plot of matched line positions
        and marginal residual plots along the dispersion and cross-dispersion axes.

        Parameters
        ----------
        figsize
            Tuple specifying the size of the figure. If None, the default Matplotlib
            figure size is used.
        max_match_distance
            Maximum distance for matching lines, passed to `match_lines`.
        rlim
            Residual axis limits as a tuple (min, max). Applied to both marginal
            residual plots. If None, limits are set automatically.

        Returns
        -------
        matplotlib.figure.Figure
            The Matplotlib figure containing the diagnostic plots.
        """
        fig = plt.Figure(figsize=figsize, layout="constrained")
        gs = plt.GridSpec(2, 2, width_ratios=(4, 1), height_ratios=(1, 3), figure=fig)

        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        rxs, rys, dxs = self.match_lines(max_match_distance, concatenate=False)
        for i, (rx, ry, dx) in enumerate(zip(rxs, rys, dxs)):
            residuals = dx - self.solution.corr_to_det(rx, ry)[0]
            ax1.scatter(rx, ry, s=50 * abs(residuals), label=f"Arc {i+1}")
            ax2.plot(rx, residuals, ".")
            ax3.plot(residuals, ry, ".")
        ax1.legend(loc="upper right")
        ax2.set_xlim(ax1.get_xlim())
        ax3.set_ylim(ax1.get_ylim())
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax1, xlabel="Dispersion axis [pix]", ylabel="Cross-dispersion axis [pix]")
        ax2.set_ylabel("Residuals [pix]")
        ax3.set_xlabel("Residuals [pix]")

        if rlim is not None:
            ax2.set_ylim(rlim)
            ax3.set_xlim(rlim)

        return fig
