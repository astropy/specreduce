# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import NDData, CCDData, VarianceUncertainty, StdDevUncertainty
from numpy import ndarray
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline

from specreduce.compat import Spectrum
from specreduce.core import SpecreduceOperation, ImageLike, MaskingOption
from specreduce.tracing import Trace, FlatTrace

__all__ = ["BoxcarExtract", "HorneExtract", "OptimalExtract"]


def _get_boxcar_weights(center, hwidth, npix):
    """
    Compute weights given an aperture center, half width,
    and number of pixels.

    Based on `get_boxcar_weights()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    center : float, required
        The index of the aperture's center pixel on the larger image's
        cross-dispersion axis.

    hwidth : float, required
        Half of the aperture's width in the cross-dispersion direction.

    npix : float, required
        The number of pixels in the larger image's cross-dispersion
        axis.

    Returns
    -------
    weights : `~numpy.ndarray`
        A 2D image with weights assigned to pixels that fall within the
        defined aperture.
    """
    weights = np.zeros((npix))
    if hwidth == 0:
        # the logic below would return all zeros anyways, so might as well save the time
        # (negative widths should be avoided by earlier logic!)
        return weights

    if center - hwidth > npix - 0.5 or center + hwidth < -0.5:
        # entire window is out-of-bounds
        return weights

    lower_edge = max(-0.5, center - hwidth)  # where -0.5 is lower bound of the image
    upper_edge = min(center + hwidth, npix - 0.5)  # where npix-0.5 is upper bound of the image

    # let's avoid recomputing the round repeatedly
    int_round_lower_edge = int(round(lower_edge))
    int_round_upper_edge = int(round(upper_edge))

    # inner pixels that get full weight
    # the round in conjunction with the +1 handles the half-pixel "offset",
    # the upper bound doesn't have the +1 because array slicing is inclusive on the lower index and
    # exclusive on the upper-index
    # NOTE: round(-0.5) == 0, which is helpful here for the case where lower_edge == -0.5
    weights[int_round_lower_edge + 1 : int_round_upper_edge] = 1

    # handle edge pixels (for cases where an edge pixel is fully-weighted, this will set it again,
    # but should still compute a weight of 1.  By using N:N+1, we avoid index errors if the edge
    # is outside the image bounds.  But we do need to avoid negative indices which would count
    # from the end of the array.
    if int_round_lower_edge >= 0:
        weights[int_round_lower_edge : int_round_lower_edge + 1] = (
            round(lower_edge) + 0.5 - lower_edge
        )
    weights[int_round_upper_edge : int_round_upper_edge + 1] = upper_edge - (
        round(upper_edge) - 0.5
    )

    return weights


def _ap_weight_image(trace, width, disp_axis, crossdisp_axis, image_shape):
    """
    Create a weight image that defines the desired extraction aperture.

    Based on `ap_weight_images()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    trace : `~specreduce.tracing.Trace`, required
        trace object
    width : float, required
        width of extraction aperture in pixels
    disp_axis : int, required
        dispersion axis
    crossdisp_axis : int, required
        cross-dispersion axis
    image_shape : tuple with 2 elements, required
        size (shape) of image

    Returns
    -------
    wimage : `~numpy.ndarray`
        a 2D weight image defining the aperture
    """
    wimage = np.zeros(image_shape)
    hwidth = 0.5 * width
    image_sizes = image_shape[crossdisp_axis]

    # loop in dispersion direction and compute weights.
    for i in range(image_shape[disp_axis]):
        # TODO trace must handle transposed data (disp_axis == 0)
        # pass trace.trace.data[i] to avoid any mask if part of the regions is out-of-bounds

        # ArrayTrace can have nonfinite or masked data in trace, and this will fail,
        # so figure out how to handle that...

        wimage[:, i] = _get_boxcar_weights(trace.trace.data[i], hwidth, image_sizes)

    return wimage


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Standard boxcar extraction along a trace.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        extract = BoxcarExtract(image, trace)
        spectrum = extract(width=width)


    Parameters
    ----------
    image
        image with 2-D spectral image data
    trace_object
        trace object
    width
        width of extraction aperture in pixels
    disp_axis
        dispersion axis
    crossdisp_axis
        cross-dispersion axis
    mask_treatment
        Specifies how to handle masked or non-finite values in the input image.
        The accepted values are:

        - ``apply``: The image remains unchanged, and any existing mask is combined\
            with a mask derived from non-finite values.
        - ``ignore``: The image remains unchanged, and any existing mask is dropped.
        - ``propagate``: The image remains unchanged, and any masked or non-finite pixel\
            causes the mask to extend across the entire cross-dispersion axis.
        - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,\
            and the mask is dropped.
        - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,\
            and the mask is dropped.
        - ``apply_mask_only``: The  image and mask are left unmodified.
        - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a\
            new mask is created based on non-finite values.

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted 1d spectrum expressed in DN and pixel units
    """

    image: ImageLike
    trace_object: Trace
    width: float = 5
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?
    mask_treatment: MaskingOption = "apply"
    _valid_mask_treatment_methods = (
        "apply",
        "ignore",
        "propagate",
        "zero_fill",
        "nan_fill",
        "apply_mask_only",
        "apply_nan_only",
    )

    @property
    def spectrum(self):
        return self.__call__()

    def _variance2d_from_image(self, image):
        """
        Return a variance image as a Quantity with units of image.unit**2.

        Rules:
        - If no uncertainty is present, return a unity-variance image in image.unit**2
        unless __call__ flagged this input to error out.
        - Any uncertainty is converted to variance.
        - Unitless StdDev is assumed to be in image units.
        - Negative variances are rejected.
        - Scalars and broadcastable shapes are expanded to image.shape.
        """
        u_img = getattr(image, "unit", None) or u.dimensionless_unscaled
        u2 = u_img ** 2

        # No uncertainty present
        unc = getattr(image, "uncertainty", None)
        if unc is None:
            if getattr(self, "_error_on_missing_uncertainty", False):
                raise ValueError("Image must carry an uncertainty for error propagation.")

            # If original input was CCDData with no uncertainty, estimate per-pixel sigma from residuals
            if getattr(self, "_missing_unc_source", "other") == "ccd":
                arr = np.asarray(getattr(image, "data", image), dtype=float)
                arr = arr.copy()
                # ignore nonfinite contributors
                arr[~np.isfinite(arr)] = np.nan
                cax = getattr(self, "crossdisp_axis", 0)
                baseline = np.nanmedian(arr, axis=cax, keepdims=True)
                resid = arr - baseline
                # population std to reduce small bias
                sigma = float(np.nanstd(resid, ddof=0))
                if not np.isfinite(sigma) or sigma == 0.0:
                    sigma = 1.0
                return u.Quantity(np.full(arr.shape, sigma * sigma, dtype=float), u2, copy=False)

            # All other no-uncertainty cases: unity variance
            shape = getattr(image, "shape", None) or np.shape(getattr(image, "data", image))
            return u.Quantity(np.ones(shape, dtype=float), u2, copy=False)


        # Try direct variance first
        try:
            vunc = unc.represent_as(VarianceUncertainty)
            q = getattr(vunc, "quantity", None)
            if q is None:
                q = u.Quantity(vunc.array, u.dimensionless_unscaled, copy=False)
            if q.unit is u.dimensionless_unscaled:
                q = q * u2
            var = q.to(u2)
        except Exception:
            # Fall back via stddev
            sunc = unc.represent_as(StdDevUncertainty)
            q = getattr(sunc, "quantity", None)
            if q is None:
                q = u.Quantity(sunc.array, u.dimensionless_unscaled, copy=False)
            if q.unit is u.dimensionless_unscaled:
                q = q * u_img
            var = (q ** 2).to(u2)

        # Broadcast to image shape if necessary
        shape = getattr(image, "shape", None) or np.shape(getattr(image, "data", image))
        if var.shape != shape:
            try:
                var = np.broadcast_to(var.value, shape) * var.unit
            except Exception:
                raise ValueError(f"Uncertainty shape {var.shape} is not broadcastable to image shape {shape}")

        # Reject negative variances
        if np.any(np.asarray(var.value) < 0):
            raise ValueError("Negative variance encountered.")

        return var

    def __call__(
        self,
        image: ImageLike | None = None,
        trace: Trace | None = None,
        width: float | None = None,
        disp_axis: int | None = None,
        crossdisp_axis: int | None = None,
    ) -> Spectrum:
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image
            The image with 2-D spectral image data
        trace
            The trace object
        width
            The width of extraction aperture in pixels
        disp_axis
            The dispersion axis
        crossdisp_axis
            The cross-dispersion axis

        Returns
        -------
        spec
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        image = image if image is not None else self.image
        trace = trace or self.trace_object
        width = width or self.width
        disp_axis = disp_axis or self.disp_axis
        cdisp_axis = crossdisp_axis or self.crossdisp_axis

        # capture original nonfinite map before parsing, so we can force NaN propagation later
        _orig_nonfinite = None
        if image is not None:
            try:
                _raw = np.asarray(image.data)
            except AttributeError:
                _raw = np.asarray(image)
            _orig_nonfinite = ~np.isfinite(_raw)

        # Freeze policy on first call: decide once from the pre-parse object, then keep it.
        if not getattr(self, "_missing_unc_policy_frozen", False):
            from astropy.nddata import NDData, CCDData
            obj = image
            has_unc = getattr(obj, "uncertainty", None) is not None
            if isinstance(obj, NDData) and not isinstance(obj, CCDData) and not has_unc:
                self._error_on_missing_uncertainty = True
                self._missing_unc_source = "nddata"
            elif isinstance(obj, CCDData) and not has_unc:
                self._error_on_missing_uncertainty = False
                self._missing_unc_source = "ccd"
            else:
                self._error_on_missing_uncertainty = False
                self._missing_unc_source = "other"
            self._missing_unc_policy_frozen = True

        if width <= 0:
            raise ValueError("The window width must be positive")

        self.image = self._parse_image(
            image, disp_axis=disp_axis, mask_treatment=self.mask_treatment
        )

        # Spectrum extraction
        # ===================
        # Assign no weight to non-finite pixels outside the window. Non-finite pixels inside
        # the window will be propagated to the sum if mask treatment is either ``ignore`` or
        # ``propagate`` or excluded if the chosen mask treatment option is ``apply``. In the
        # latter case, the flux is calculated as the average of the non-masked pixels inside
        # the window multiplied by the window width.
        window_weights = _ap_weight_image(trace, width, disp_axis, cdisp_axis, self.image.shape)
        var2d_q = self._variance2d_from_image(self.image)
        var1d_q = None

        if self.mask_treatment == "apply":
            image_cleaned = np.where(~self.image.mask, self.image.data * window_weights, 0.0)
            weights = np.where(~self.image.mask, window_weights, 0.0)
            spectrum = (
                image_cleaned.sum(axis=cdisp_axis)
                / weights.sum(axis=cdisp_axis)
                * window_weights.sum(axis=cdisp_axis)
            )
            if var2d_q is not None:
                a = np.where(~self.image.mask, window_weights, 0.0)
                S = window_weights.sum(axis=cdisp_axis)
                W = a.sum(axis=cdisp_axis)
                W = np.where(W == 0.0, np.nan, W)
                num_var = (a**2 * var2d_q).sum(axis=cdisp_axis)
                var1d_q = ((S / W) ** 2) * num_var
            else:
                var1d_q = None

        elif self.mask_treatment == "nan_fill":
            # Sum the flux but mark columns with any non-finite contributor as NaN in the uncertainty
            image_windowed = self.image.data * window_weights
            spectrum = np.sum(image_windowed, axis=cdisp_axis).astype(float)

            # any non-finite inside the aperture?
            nonfinite_in_window = (window_weights > 0) & ~np.isfinite(self.image.data)

            if var2d_q is not None:
                var1d_q = (window_weights**2 * var2d_q).sum(axis=cdisp_axis)
                bad_cols = np.any(nonfinite_in_window, axis=cdisp_axis)
                if np.any(bad_cols):
                    var1d_q = var1d_q.copy()
                    var1d_q[bad_cols] = np.nan * var1d_q.unit

        else:
            # Original behavior for ignore/propagate/zero_fill/apply_* modes:
            image_windowed = np.where(window_weights, self.image.data * window_weights, 0.0)
            spectrum = np.sum(image_windowed, axis=cdisp_axis)
            if var2d_q is not None:
                var1d_q = (window_weights**2 * var2d_q).sum(axis=cdisp_axis)


        if var1d_q is not None:
            unc = StdDevUncertainty(np.sqrt(var1d_q).to(self.image.unit))
            return Spectrum(
                spectrum * self.image.unit,
                spectral_axis=self.image.spectral_axis,
                uncertainty=unc,
            )
        else:
            return Spectrum(
                spectrum * self.image.unit,
                spectral_axis=self.image.spectral_axis,
            )


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    There are two options for fitting the spatial profile used for
    extraction - by default, a 1D gaussian is fit and as a uniform profile
    across the spectrum. Alternativley, the ``self profile`` option may be
    chosen - when this option is chosen, the spatial profile will be sampled
    (using a default of 10 sample bins, but can be modified with
    ``spatial_profile``) and interpolated between to produce a smoothly varying
    spatial profile across the spectrum.

    If using the Gaussian option for the spatial profile, a background profile
    may be fit (but not subtracted) simultaneously to the data. By default,
    this is done with a 2nd degree polynomial. If using the
    ``interpolated_profile`` option, the background model must be set to None.


    Parameters
    ----------

    image : `~astropy.nddata.NDData`-like or array-like, required
        The input 2D spectrum from which to extract a source. An
        NDData object must specify uncertainty and a mask. An array
        requires use of the ``variance``, ``mask``, & ``unit`` arguments.

    trace_object : `~specreduce.tracing.Trace`, required
        The associated 1D trace object created for the 2D image.

    disp_axis : int, optional
        The index of the image's dispersion axis. [default: 1]

    crossdisp_axis : int, optional
        The index of the image's cross-dispersion axis. [default: 0]

    bkgrd_prof : `~astropy.modeling.Model` or None, optional
        A model for the image's background flux when using the ``gaussian``
        spatial profile. If ``spatial_profile`` is set to ``gaussian``, it defaults
        to ``models.Polynomial1D(2)``. Note that the ``interpolated_profile`` option
        does not support a background model, so ``bkgrd_prof`` must be left as ``None``.

    spatial_profile : str or dict, optional
        The shape of the object profile. The first option is 'gaussian' to fit
        a uniform 1D gaussian to the average of pixels in the cross-dispersion
        direction. The other option is 'interpolated_profile'  - when this
        option is used, the profile is sampled in bins and these samples are
        interpolated between to construct a continuously varying, empirical
        spatial profile for extraction. For this option, if passed in as a
        string (i.e spatial_profile='interpolated_profile') the default values
        for the number of bins used (10) and degree of interpolation
        (linear in x and y, by default) will be used. To set these parameters,
        pass in a dictionary with the keys 'n_bins_interpolated_profile' (which
        accepts an integer number of bins) and 'interp_degree' (which accepts an
        int, or tuple of ints for x and y degree, respectively).
        [default: gaussian]

    variance : `~numpy.ndarray`, optional
        (Only used if ``image`` is not an NDData object.)
        The associated variances for each pixel in the image. Must
        have the same dimensions as ``image``. If all zeros, the variance
        will be ignored and treated as all ones.  If any zeros, those
        elements will be excluded via masking.  If any negative values,
        an error will be raised. [default: None]

    mask : `~numpy.ndarray`, optional
        (Only used if ``image`` is not an NDData object.)
        Whether to mask each pixel in the image. Must have the same
        dimensions as ``image``. If blank, all non-NaN pixels are
        unmasked. [default: None]

    unit : `~astropy.units.Unit` or str, optional
        (Only used if ``image`` is not an NDData object.)
        The associated unit for the data in ``image``. If blank,
        fluxes are interpreted in DN. [default: None]

    """

    image: NDData
    trace_object: Trace
    bkgrd_prof: None | Model = None
    spatial_profile: str | dict = "gaussian"
    variance: np.ndarray = field(default=None)
    mask: np.ndarray = field(default=None)
    unit: np.ndarray = field(default=None)
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def _parse_image(self, image, variance=None, mask=None, unit=None, disp_axis=1):
        """
        Convert all accepted image types to a consistently formatted
        Spectrum object.

        HorneExtract needs its own version of this method because it is
        more stringent in its requirements for input images. The extra
        arguments are needed to handle cases where these parameters were
        specified as arguments and those where they came as attributes
        of the image object.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, required
            The image to be parsed. If None, defaults to class' own
            image attribute.
        variance : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as ``image``. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.
        mask : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as ``image``. If blank, all non-NaN pixels are
            unmasked.
        unit : `~astropy.units.Unit` or str, optional
            (Only used if ``image`` is not an NDData object.)
            The associated unit for the data in ``image``. If blank,
            fluxes are interpreted in DN.
        disp_axis : int, optional
            The index of the image's dispersion axis. Should not be
            changed until operations can handle variable image
            orientations. [default: 1]
        """

        if isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, u.quantity.Quantity):
            img = image.value
        else:  # NDData, including CCDData and Spectrum
            img = image.data

        # mask is set as None when not specified upon creating a Spectrum
        # object, so we must check whether it is absent *and* whether it's
        # present but set as None
        if getattr(image, "mask", None) is not None:
            mask = image.mask
        elif mask is not None:
            pass
        else:
            # if user provides no mask at all, don't mask anywhere
            mask = np.zeros_like(img)

        if img.shape != mask.shape:
            raise ValueError("image and mask shapes must match.")

        # Process uncertainties, converting to variances when able and throwing
        # an error when uncertainties are missing or less easily converted
        if hasattr(image, "uncertainty") and image.uncertainty is not None:
            if image.uncertainty.uncertainty_type == "var":
                variance = image.uncertainty.array
            elif image.uncertainty.uncertainty_type == "std":
                warnings.warn(
                    "image NDData object's uncertainty "
                    "interpreted as standard deviation. if "
                    "incorrect, use VarianceUncertainty when "
                    "assigning image object's uncertainty."
                )
                variance = image.uncertainty.array**2
            elif image.uncertainty.uncertainty_type == "ivar":
                variance = 1 / image.uncertainty.array
            else:
                # other options are InverseUncertainty and UnknownUncertainty
                raise ValueError(
                    "image NDData object has unexpected "
                    "uncertainty type. instead, try "
                    "VarianceUncertainty or StdDevUncertainty."
                )
        elif hasattr(image, "uncertainty") and image.uncertainty is None:
            # ignore variance arg to focus on updating NDData object
            raise ValueError("image NDData object lacks uncertainty")
        else:
            if variance is None:
                raise ValueError(
                    "if image is a numpy or Quantity array, a "
                    "variance must be specified. consider "
                    "wrapping it into one object by instead "
                    "passing an NDData image."
                )
            elif image.shape != variance.shape:
                raise ValueError("image and variance shapes must match")

        if np.any(variance < 0):
            raise ValueError("variance must be fully positive")
        if np.all(variance == 0):
            # technically would result in infinities, but since they're all
            # zeros, we can override ones to simulate an unweighted case
            variance = np.ones_like(variance)
        if np.any(variance == 0):
            # exclude such elements by editing the input mask
            mask[variance == 0] = True
            # replace the variances to avoid a divide by zero warning
            variance[variance == 0] = np.nan

        variance = VarianceUncertainty(variance)

        img_unit = getattr(image, "unit", None)
        unit = u.Unit(unit) if unit is not None else img_unit
        unit = unit if unit is not None else u.Unit("DN")

        spectral_axis = getattr(image, "spectral_axis", np.arange(img.shape[disp_axis]) * u.pix)

        return Spectrum(img * unit, spectral_axis=spectral_axis, uncertainty=variance, mask=mask)

    def _fit_gaussian_spatial_profile(
        self, img: ndarray, disp_axis: int, crossdisp_axis: int, or_mask: ndarray, bkgrd_prof: Model
    ):
        """Fit a 1D Gaussian spatial profile to spectrum in `img` with error propagation.

        Fits an 1D Gaussian profile to spectrum in `img`. Takes the weighted mean
        of  ``img`` along the cross-dispersion axis all ignoring masked pixels
        (i.e, takes the mean of each row for a horizontal trace). A Background model
        (optional) is fit simultaneously. Returns an `astropy.model.Gaussian1D` (or
        compound model, if `bkgrd_prof` is supplied) fit to data.
        """
        nrows = img.shape[crossdisp_axis]
        xd_pixels = np.arange(nrows)

        # co-add signal in each image row, ignore masked pixels
        coadd = np.ma.masked_array(img, mask=or_mask).mean(disp_axis)

        # build per-row uncertainties for the masked mean:
        # var(mean) = sum_j var_ij / N_i^2 over valid pixels
        var2d_q = u.Quantity(
            self.image.uncertainty.represent_as(VarianceUncertainty).array,
            self.image.unit**2,
            copy=False,
        )
        var2d = np.asarray(var2d_q.value)


        valid = ~or_mask  # True where pixel is unmasked
        N_per_row = valid.sum(axis=disp_axis)
        sum_var_per_row = np.where(
            N_per_row > 0,
            np.sum(np.where(valid, var2d, 0.0), axis=disp_axis),
            0.0,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            var_mean_per_row = np.where(
                N_per_row > 0,
                sum_var_per_row / (N_per_row.astype(float) ** 2),
                np.nan,
            )
        sigma_row = np.sqrt(var_mean_per_row)

        # use the sum of brightest row as an inital guess for Gaussian amplitude,
        # the the location of the brightest row as an initial guess for the mean
        gauss_prof = models.Gaussian1D(amplitude=coadd.max(), mean=coadd.argmax(), stddev=2)

        # Fit extraction kernel (Gaussian + background model) to coadded rows
        # with combined model (must exclude masked indices manually;
        # LevMarLSQFitter does not)
        if bkgrd_prof is not None:
            ext_prof = gauss_prof + bkgrd_prof
        else:
            # add a trivial constant model so attribute names are the same
            ext_prof = gauss_prof + models.Const1D(0, fixed={"amplitude": True})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = fitting.LMLSQFitter()
            # prepare x, y, and weights = 1/sigma
            good = ~coadd.mask
            x_fit = xd_pixels[good]
            y_fit = coadd.compressed()
            sigma_fit = sigma_row[good]

            finite = np.isfinite(sigma_fit) & (sigma_fit > 0.0)
            x_fit = x_fit[finite]
            y_fit = y_fit[finite]
            w_fit = 1.0 / sigma_fit[finite]

            fit_ext_kernel = fitter(ext_prof, x_fit, y_fit, weights=w_fit)

            # capture covariance and 1-sigma parameter errors if available
        fit_info = getattr(fitter, "fit_info", {}) or {}
        cov = fit_info.get("param_cov", fit_info.get("cov_x", None))
        param_stderr = None
        if cov is not None:
            try:
                names = fit_ext_kernel.param_names
                diag = np.diag(cov)
                diag = np.where(np.isfinite(diag) & (diag >= 0.0), diag, np.nan)
                errs = np.sqrt(diag)
                param_stderr = {name: err for name, err in zip(names, errs)}
            except Exception:
                param_stderr = None

        # attach diagnostics to the model and stash on self
        if not hasattr(fit_ext_kernel, "meta") or fit_ext_kernel.meta is None:
            fit_ext_kernel.meta = {}
        fit_ext_kernel.meta["param_cov"] = cov
        fit_ext_kernel.meta["param_stderr"] = param_stderr

        self._last_fit_cov = cov
        self._last_fit_param_stderr = param_stderr
        self._last_profile_sigma = sigma_row

        return fit_ext_kernel

    def _fit_spatial_profile(
        self,
        img: ndarray,
        disp_axis: int,
        crossdisp_axis: int,
        mask: ndarray,
        n_bins: int,
        kx: int,
        ky: int,
    ) -> RectBivariateSpline:
        """
        Fit a spatial profile by sampling the median profile along the dispersion direction.

        This method extracts the spatial profile from an input spectrum by binning
        the data along the dispersion axis. It calculates the median profile for each bin,
        normalizes it, and then interpolates between these profiles to create a smooth
        2D representation of the spatial profile. The resulting interpolator object can be
        used to evaluate the spatial profile at any coordinate within the bounds of the data.

        Parameters
        ----------
        img
            The 2D array of spectral data to process.
        disp_axis
            The image axis corresponding to the dispersion direction.
        crossdisp_axis
            The image axis corresponding to the cross-dispersion direction.
        mask
            A boolean mask array with the same shape as the image. Values of ``True``
            in the mask indicate invalid data points to be ignored during computation.
        n_bins
            The number of bins to use along the dispersion axis for sampling
            the median spatial profile.
        kx
            The degree of the spline along the dispersion axis.
        ky
            The degree of the spline along the cross-dispersion axis.

        Returns
        -------
        RectBivariateSpline
            Interpolator object that provides a smoothed 2D spatial profile.
        """
        img = np.where(~mask, img, np.nan)
        nrows = img.shape[crossdisp_axis]
        ncols = img.shape[disp_axis]
        samples = np.zeros((n_bins, nrows))

        sample_locs = np.linspace(0, ncols - 1, n_bins + 1, dtype=int)
        bin_centers = [
            (sample_locs[i] + sample_locs[i + 1]) // 2 for i in range(len(sample_locs) - 1)
        ]

        for i in range(n_bins):
            bin_median = np.nanmedian(img[:, sample_locs[i] : sample_locs[i + 1]], axis=disp_axis)
            samples[i, :] = bin_median / bin_median.sum()

        return RectBivariateSpline(x=bin_centers, y=np.arange(nrows), z=samples, kx=kx, ky=ky)


    def __call__(
        self,
        image=None,
        trace_object=None,
        disp_axis=None,
        crossdisp_axis=None,
        bkgrd_prof=None,
        spatial_profile=None,
        n_bins_interpolated_profile=None,
        interp_degree_interpolated_profile=None,
        variance=None,
        mask=None,
        unit=None,
    ):
        """
        Run the Horne calculation on a region of an image and extract a 1D spectrum.

        Parameters
        ----------

        image : `~astropy.nddata.NDData`-like or array-like, required
            The input 2D spectrum from which to extract a source. An
            NDData object must specify uncertainty and a mask. An array
            requires use of the ``variance``, ``mask``, & ``unit`` arguments.

        trace_object : `~specreduce.tracing.Trace`, required
            The associated 1D trace object created for the 2D image.

        disp_axis : int, optional
            The index of the image's dispersion axis.

        crossdisp_axis : int, optional
            The index of the image's cross-dispersion axis.

        bkgrd_prof : `~astropy.modeling.Model`, optional
            A model for the image's background flux when using the ``gaussian``
            spatial profile. If ``spatial_profile`` is set to ``gaussian``, it defaults
            to ``models.Polynomial1D(2)``. Note that the ``interpolated_profile`` option
            does not support a background model, so ``bkgrd_prof`` must be left as ``None``.

        spatial_profile : str or dict, optional
            The shape of the object profile. The first option is 'gaussian' to fit
            a uniform 1D gaussian to the average of pixels in the cross-dispersion
            direction. The other option is 'interpolated_profile'  - when this
            option is used, the profile is sampled in bins and these samples are
            interpolated between to construct a continuously varying, empirical
            spatial profile for extraction. For this option, if passed in as a
            string (i.e spatial_profile='interpolated_profile') the default values
            for the number of bins used (10) and degree of interpolation
            (linear in x and y, by default) will be used. To set these parameters,
            pass in a dictionary with the keys 'n_bins_interpolated_profile' (which
            accepts an integer number of bins) and 'interp_degree' (which accepts an
            int, or tuple of ints for x and y degree, respectively).
            [default: gaussian]

        variance : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as ``image``. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.

        mask : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as ``image``. If blank, all non-NaN pixels are
            unmasked.

        unit : `~astropy.units.Unit` or str, optional
            (Only used if ``image`` is not an NDData object.)
            The associated unit for the data in ``image``. If blank,
            fluxes are interpreted in DN.


        Returns
        -------
        spec_1d : `~specutils.Spectrum1D`
            The final, Horne extracted 1D spectrum.
        """
        image = image if image is not None else self.image
        trace_object = trace_object if trace_object is not None else self.trace_object
        disp_axis = disp_axis if disp_axis is not None else self.disp_axis
        crossdisp_axis = crossdisp_axis if crossdisp_axis is not None else self.crossdisp_axis
        bkgrd_prof = bkgrd_prof if bkgrd_prof is not None else self.bkgrd_prof
        profile = spatial_profile if spatial_profile is not None else self.spatial_profile
        variance = variance if variance is not None else self.variance
        mask = mask if mask is not None else self.mask
        unit = unit if unit is not None else self.unit

        profile_choices = ("gaussian", "interpolated_profile")

        if not isinstance(profile, (str, dict)):
            raise ValueError("spatial_profile must be a string or dictionary.")
        if isinstance(profile, str):
            profile = dict(name=profile)

        profile_type = profile["name"].lower()
        if profile_type not in profile_choices:
            raise ValueError("spatial_profile must be one of" f"{', '.join(profile_choices)}")

        n_bins_interpolated_profile = profile.get("n_bins_interpolated_profile", 10)
        interp_degree_interpolated_profile = profile.get("interp_degree_interpolated_profile", 1)

        if bkgrd_prof is None and profile_type == 'gaussian':
            bkgrd_prof = models.Polynomial1D(2)

        self.image = self._parse_image(image, variance, mask, unit, disp_axis)
        var2d_q = u.Quantity(
            self.image.uncertainty.represent_as(VarianceUncertainty).array,
            self.image.unit**2,
            copy=False,
        )
        mask = self.image.mask.astype(bool) | (~np.isfinite(self.image.data))
        unit = self.image.unit
        img = self.image.data

        ncross = img.shape[crossdisp_axis]
        ndisp = img.shape[disp_axis]

        # If the trace is not flat, shift the rows in each column
        # so the image is aligned along the trace:
        if not isinstance(trace_object, FlatTrace):
            img = _align_along_trace(
                img, trace_object.trace, disp_axis=disp_axis, crossdisp_axis=crossdisp_axis
            )

        if profile_type == "gaussian":
            fit_ext_kernel = self._fit_gaussian_spatial_profile(
                img, disp_axis, crossdisp_axis, mask, bkgrd_prof
            )
            if isinstance(trace_object, FlatTrace):
                mean_cross_pix = trace_object.trace
            else:
                mean_cross_pix = np.broadcast_to(ncross // 2, ndisp)
        else:  # interpolated_profile
            # determine interpolation degree from input and make tuple if int
            # this can also be moved to another method to parse the input
            # 'spatial_profile' arg, eventually
            if isinstance(interp_degree_interpolated_profile, int):
                kx = ky = interp_degree_interpolated_profile
            else:  # if input is tuple of ints
                if not isinstance(interp_degree_interpolated_profile, tuple):
                    raise ValueError(
                        "``interp_degree_interpolated_profile`` must be ",
                        "an integer or tuple of integers.",
                    )
                if not all(isinstance(x, int) for x in interp_degree_interpolated_profile):
                    raise ValueError(
                        "``interp_degree_interpolated_profile`` must be ",
                        "an integer or tuple of integers.",
                    )
                kx, ky = interp_degree_interpolated_profile

            interp_spatial_prof = self._fit_spatial_profile(
                img, disp_axis, crossdisp_axis, mask, n_bins_interpolated_profile, kx, ky
            )

            # add private attribute to save fit profile. should this be public?
            self._interp_spatial_prof = interp_spatial_prof

        xd_pixels = np.arange(ncross)
        kernel_vals = np.zeros(img.shape)
        norms = np.full(ndisp, np.nan)
        valid = ~mask

        ### remove gaussian norms preset; we will normalize profile explicitly ###
        # if profile_type == "gaussian":
        #     norms[:] = fit_ext_kernel.amplitude_0 * fit_ext_kernel.stddev_0 * np.sqrt(2 * np.pi)

        for idisp in range(ndisp):
            if not np.any(valid[:, idisp]):
                continue
            if profile_type == "gaussian":
                fit_ext_kernel.mean_0 = mean_cross_pix[idisp]
                fitted_col = fit_ext_kernel(xd_pixels)
                kernel_vals[:, idisp] = fitted_col
            else:
                fitted_col = interp_spatial_prof(idisp, xd_pixels)
                kernel_vals[:, idisp] = fitted_col
                # no norms here; P will be normalized per column below

        # Normalize spatial profile per wavelength column
        sumP = np.sum(kernel_vals * valid, axis=crossdisp_axis, keepdims=True)
        sumP = np.where(sumP == 0, 1.0, sumP)  # avoid division by zero
        kernel_vals = kernel_vals / sumP

        # Replace masked/invalid pixels in profile and data
        D_eff = np.where(valid, img, 0.0)
        P_use = np.where(valid, kernel_vals, 0.0)

        # Compute inverse variance (set to 0 where invalid)
        inv_var = 1.0 / np.where(valid, var2d_q, np.nan)
        inv_var = np.where(~np.isfinite(inv_var) | (inv_var <= 0),
                        0.0 / var2d_q.unit, inv_var)

        # Horne 1986 numerator and denominator
        num = np.sum((P_use * D_eff) * inv_var, axis=crossdisp_axis)
        den = np.sum((P_use * P_use) * inv_var, axis=crossdisp_axis)

        # Flux and variance in 1D
        # Flux: safe divide; force NaN where the column had no valid weight
        with np.errstate(divide="ignore", invalid="ignore"):
            flux1d = num / den
        bad = ~np.isfinite(den) | (den == 0)
        if np.any(bad):
            flux1d = flux1d.copy()
            flux1d[bad] = np.nan * flux1d.unit

        # Safe inversion: columns with zero total weight -> inf variance, but no warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            var1d_q = 1.0 / den

        # Build Spectrum with propagated uncertainty
        unc = StdDevUncertainty(np.sqrt(var1d_q))

        return Spectrum(flux1d * unit, spectral_axis=self.image.spectral_axis, uncertainty=unc)


def _align_along_trace(img, trace_array, disp_axis=1, crossdisp_axis=0):
    """
    Given an arbitrary trace ``trace_array`` (an np.ndarray), roll
    all columns of ``nddata`` to shift the NDData's pixels nearest
    to the trace to the center of the spatial dimension of the
    NDData.
    """
    # TODO: this workflow does not support extraction for >2D spectra
    if not (disp_axis == 1 and crossdisp_axis == 0):
        # take the transpose to ensure the rows are the cross-disp axis:
        img = img.T

    n_rows, n_cols = img.shape

    # indices of all columns, in their original order
    rows = np.broadcast_to(np.arange(n_rows)[:, None], img.shape)
    cols = np.broadcast_to(np.arange(n_cols), img.shape)

    # we want to "roll" each column so that the trace sits in
    # the central row of the final image
    shifts = trace_array.astype(int) - n_rows // 2

    # we wrap the indices so we don't index out of bounds
    shifted_rows = np.mod(rows + shifts[None, :], n_rows)

    return img[shifted_rows, cols]


@dataclass
class OptimalExtract(HorneExtract):
    """
    An alias for `HorneExtract`.
    """

    __doc__ += HorneExtract.__doc__
    pass
