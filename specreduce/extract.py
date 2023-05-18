# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np

from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import NDData, VarianceUncertainty
from scipy.interpolate import RectBivariateSpline

from specreduce.core import SpecreduceOperation
from specreduce.tracing import Trace, FlatTrace
from specutils import Spectrum1D

__all__ = ['BoxcarExtract', 'HorneExtract', 'OptimalExtract']


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

    if center-hwidth > npix-0.5 or center+hwidth < -0.5:
        # entire window is out-of-bounds
        return weights

    lower_edge = max(-0.5, center-hwidth)  # where -0.5 is lower bound of the image
    upper_edge = min(center+hwidth, npix-0.5)  # where npix-0.5 is upper bound of the image

    # let's avoid recomputing the round repeatedly
    int_round_lower_edge = int(round(lower_edge))
    int_round_upper_edge = int(round(upper_edge))

    # inner pixels that get full weight
    # the round in conjunction with the +1 handles the half-pixel "offset",
    # the upper bound doesn't have the +1 because array slicing is inclusive on the lower index and
    # exclusive on the upper-index
    # NOTE: round(-0.5) == 0, which is helpful here for the case where lower_edge == -0.5
    weights[int_round_lower_edge+1:int_round_upper_edge] = 1

    # handle edge pixels (for cases where an edge pixel is fully-weighted, this will set it again,
    # but should still compute a weight of 1.  By using N:N+1, we avoid index errors if the edge
    # is outside the image bounds.  But we do need to avoid negative indices which would count
    # from the end of the array.
    if int_round_lower_edge >= 0:
        weights[int_round_lower_edge:int_round_lower_edge+1] = round(lower_edge) + 0.5 - lower_edge
    weights[int_round_upper_edge:int_round_upper_edge+1] = upper_edge - (round(upper_edge) - 0.5)

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
        wimage[:, i] = _get_boxcar_weights(trace.trace.data[i], hwidth, image_sizes)

    return wimage


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
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        extract = BoxcarExtract(image, trace)
        spectrum = extract(width=width)


    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like, required
        image with 2-D spectral image data
    trace_object : Trace, required
        trace object
    width : float, optional
        width of extraction aperture in pixels
    disp_axis : int, optional
        dispersion axis
    crossdisp_axis : int, optional
        cross-dispersion axis

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted 1d spectrum expressed in DN and pixel units
    """
    image: NDData
    trace_object: Trace
    width: float = 5
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def __call__(self, image=None, trace_object=None, width=None,
                 disp_axis=None, crossdisp_axis=None):
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, required
            image with 2-D spectral image data
        trace_object : Trace, required
            trace object
        width : float, optional
            width of extraction aperture in pixels [default: 5]
        disp_axis : int, optional
            dispersion axis [default: 1]
        crossdisp_axis : int, optional
            cross-dispersion axis [default: 0]


        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        image = image if image is not None else self.image
        trace_object = trace_object if trace_object is not None else self.trace_object
        width = width if width is not None else self.width
        disp_axis = disp_axis if disp_axis is not None else self.disp_axis
        crossdisp_axis = crossdisp_axis if crossdisp_axis is not None else self.crossdisp_axis

        # handle image processing based on its type
        self.image = self._parse_image(image)

        # TODO: this check can be removed if/when implemented as a check in FlatTrace
        if isinstance(trace_object, FlatTrace):
            if trace_object.trace_pos < 1:
                raise ValueError('trace_object.trace_pos must be >= 1')

        if width < 0:
            raise ValueError("width must be positive")

        # weight image to use for extraction
        wimg = _ap_weight_image(
            trace_object,
            width,
            disp_axis,
            crossdisp_axis,
            self.image.shape)

        # extract, assigning no weight to non-finite pixels outside the window
        # (non-finite pixels inside the window will still make it into the sum)
        image_windowed = np.where(wimg, self.image.data*wimg, 0)
        ext1d = np.sum(image_windowed, axis=crossdisp_axis)
        return Spectrum1D(ext1d * self.image.unit,
                          spectral_axis=self.image.spectral_axis)


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    There are two options for fitting the spatial profile used for
    extraction - by default, a 1D gaussian is fit and as a uniform profile
    across the spectrum. Alternativley, the ``self profile`` option may be
    chosen - when this option is chosen, the spatial profile will be sampled
    at various locations (set by <>) and interpolated between to produce a
    smoothly varying spatial profile across the spectrum.

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
        A model for the image's background flux. If ``spatial_profile`` is set
        to ``interpolated_profile``, then ``bkgrd_prof`` must be set to None.
        [default: models.Polynomial1D(2)].

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
    bkgrd_prof: Model = field(default=models.Polynomial1D(2))
    spatial_profile: str = 'gaussian'  # can actually be str, dict
    variance: np.ndarray = field(default=None)
    mask: np.ndarray = field(default=None)
    unit: np.ndarray = field(default=None)
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def _parse_image(self, image,
                     variance=None, mask=None, unit=None, disp_axis=1):
        """
        Convert all accepted image types to a consistently formatted
        Spectrum1D object.

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
        else:  # NDData, including CCDData and Spectrum1D
            img = image.data

        # mask is set as None when not specified upon creating a Spectrum1D
        # object, so we must check whether it is absent *and* whether it's
        # present but set as None
        if getattr(image, 'mask', None) is not None:
            mask = image.mask
        elif mask is not None:
            pass
        else:
            # if user provides no mask at all, don't mask anywhere
            mask = np.zeros_like(img)

        if img.shape != mask.shape:
            raise ValueError('image and mask shapes must match.')

        # Process uncertainties, converting to variances when able and throwing
        # an error when uncertainties are missing or less easily converted
        if (hasattr(image, 'uncertainty')
                and image.uncertainty is not None):
            if image.uncertainty.uncertainty_type == 'var':
                variance = image.uncertainty.array
            elif image.uncertainty.uncertainty_type == 'std':
                warnings.warn("image NDData object's uncertainty "
                              "interpreted as standard deviation. if "
                              "incorrect, use VarianceUncertainty when "
                              "assigning image object's uncertainty.")
                variance = image.uncertainty.array**2
            elif image.uncertainty.uncertainty_type == 'ivar':
                variance = 1 / image.uncertainty.array
            else:
                # other options are InverseUncertainty and UnknownUncertainty
                raise ValueError("image NDData object has unexpected "
                                 "uncertainty type. instead, try "
                                 "VarianceUncertainty or StdDevUncertainty.")
        elif (hasattr(image, 'uncertainty')
              and image.uncertainty is None):
            # ignore variance arg to focus on updating NDData object
            raise ValueError('image NDData object lacks uncertainty')
        else:
            if variance is None:
                raise ValueError("if image is a numpy or Quantity array, a "
                                 "variance must be specified. consider "
                                 "wrapping it into one object by instead "
                                 "passing an NDData image.")
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

        unit = getattr(image, 'unit',
                       u.Unit(unit) if unit is not None else u.Unit('DN'))

        spectral_axis = getattr(image, 'spectral_axis',
                                np.arange(img.shape[disp_axis]) * u.pix)

        return Spectrum1D(img * unit, spectral_axis=spectral_axis,
                          uncertainty=variance, mask=mask)

    def _fit_gaussian_spatial_profile(self, img, disp_axis, crossdisp_axis,
                                      or_mask, bkgrd_prof):

        """
            Fits an 1D Gaussian profile to spectrum in `img`. Takes the mean
            of  ``img`` along the cross-dispersion axis (i.e, takes the mean of
            each row for a horizontal trace). Any columns with non-finite values
            are omitted from the fit. Background model (optional) is fit
            simultaneously. Returns an `astropy.model.Gaussian1D` (or compound
            model, if `bkgrd_prof` is supplied) fit to data.
        """

        # co-add signal in each image row
        nrows = img.shape[crossdisp_axis]
        ncols = img.shape[disp_axis]
        xd_pixels = np.arange(nrows)

        # for now, mask row with any non-finite value.
        row_mask = np.logical_or.reduce(or_mask, axis=disp_axis)
        coadd = np.ma.masked_array(np.sum(img, axis=disp_axis) / ncols,
                                   mask=row_mask)

        # use the sum of brightest row as an inital guess for Gaussian amplitude,
        # the the location of the brightest row as an initial guess for the mean
        gauss_prof = models.Gaussian1D(amplitude=coadd.max(),
                                       mean=coadd.argmax(), stddev=2)

        # Fit extraction kernel (Gaussian + background model) to coadded rows
        # with combined model (must exclude masked indices manually;
        # LevMarLSQFitter does not)
        if bkgrd_prof is not None:
            ext_prof = gauss_prof + bkgrd_prof
        else:
            # add a trivial constant model so attribute names are the same
            ext_prof = gauss_prof + models.Const1D(0, fixed={'amplitude': True})

        fitter = fitting.LevMarLSQFitter()
        fit_ext_kernel = fitter(ext_prof, xd_pixels[~row_mask], coadd[~row_mask])

        return fit_ext_kernel

    def _fit_self_spatial_profile(self, img, disp_axis, crossdisp_axis, or_mask,
                                  n_bins_interpolated_profile, kx, ky):

        """
            Fit a spatial profile to spectrum by sampling the median profile in
            bins (number of which set be `n_bins_interpolated_profile` along the
            dispersion direction, and interpolating between
            samples. Columns (assuming horizontal trace) with any non-finite
            values will be omitted from the fit. Returns an interpolator object
            (RectBivariateSpline) that can be evaluated at any x,y.
        """

        # boundaries of bins for sampling profile.
        sample_locs = np.linspace(0, img.shape[disp_axis]-1,
                                  n_bins_interpolated_profile+1, dtype=int)
        # centers of these bins, roughly
        bin_centers = [(sample_locs[i]+sample_locs[i+1]) // 2 for i in
                       range(len(sample_locs) - 1)]

        # for now, since fitting isn't enabled for arrays with any nans
        # mask out the columns with any non-finite values to make sure
        # these don't contribute to the fit profile
        col_mask = np.logical_or.reduce(or_mask, axis=crossdisp_axis)

        # make a full mask for the image based on which cols have nans
        img_col_mask = np.tile(col_mask, (img.shape[0], 1))

        # need to make a new masked array since this mask is different
        # omit all columns with nans from contributing to the fit
        new_masked_arr = np.ma.array(img.data.copy(), mask=img_col_mask)

        # sample at these locations, normalize to area so this just reflects the
        # shape of the spectrum any shifts in center location should be corrected
        # by _align_along_trace (this should be addressed later with a better way
        # to flatten the trace, because trace can be wiggly even if 'flat'...)
        samples = []
        for i in range(n_bins_interpolated_profile):
            slicee = new_masked_arr[:, sample_locs[i]:sample_locs[i+1]]
            bin_median = np.ma.median(slicee, axis=disp_axis)
            bin_median_sum = np.ma.sum(bin_median)
            samples.append(bin_median / bin_median_sum)

        interp_2d = RectBivariateSpline(x=bin_centers,
                                        y=np.arange(img.shape[crossdisp_axis]),
                                        z=samples, kx=kx, ky=ky)

        return interp_2d

    def __call__(self, image=None, trace_object=None,
                 disp_axis=None, crossdisp_axis=None,
                 bkgrd_prof=None, spatial_profile=None,
                 n_bins_interpolated_profile=None,
                 interp_degree_interpolated_profile=None,
                 variance=None, mask=None, unit=None):
        """
        Run the Horne calculation on a region of an image and extract a
        1D spectrum.

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
            A model for the image's background flux.

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
        spatial_profile = (spatial_profile if spatial_profile is not None else
                           self.spatial_profile)
        variance = variance if variance is not None else self.variance
        mask = mask if mask is not None else self.mask
        unit = unit if unit is not None else self.unit

        # figure out what 'spatial_profile' was provided
        # put this parsing into another method at some point, its a lot..
        interp_degree_interpolated_profile = None
        n_bins_interpolated_profile = None
        spatial_profile_choices = ('gaussian', 'interpolated_profile')

        if isinstance(spatial_profile, str):
            spatial_profile = spatial_profile.lower()
            if spatial_profile not in spatial_profile_choices:
                raise ValueError("spatial_profile must be one of"
                                 f"{', '.join(spatial_profile_choices)}")
            if spatial_profile == 'interpolated_profile':  # use defaults
                bkgrd_prof = None
                n_bins_interpolated_profile = 10
                interp_degree_interpolated_profile = 1
        elif isinstance(spatial_profile, dict):
            # first, figure out what type of profile is indicated
            # right now, the only type that should use a dictionary is 'interpolated_profile'
            # but this may be extended in the future hence the 'name' key. also,
            # the gaussian option could be supplied as a single-key dict

            # will raise key error if not present, and also fail on .lower if not
            # a string - think this is informative enough
            spatial_profile_type = spatial_profile['name'].lower()

            if spatial_profile_type not in spatial_profile_choices:
                raise ValueError("spatial_profile must be one of"
                                 f"{', '.join(spatial_profile_choices)}")
            if spatial_profile_type == 'gaussian':
                spatial_profile = 'gaussian'
            else:
                if 'n_bins_interpolated_profile' in spatial_profile.keys():
                    n_bins_interpolated_profile = \
                        spatial_profile['n_bins_interpolated_profile']
                else:  # use default
                    n_bins_interpolated_profile = 10
                if 'interp_degree_interpolated_profile' in spatial_profile.keys():
                    interp_degree_interpolated_profile = \
                        spatial_profile['interp_degree_interpolated_profile']
                else:  # use default
                    interp_degree_interpolated_profile = 1
            spatial_profile = spatial_profile_type
            bkgrd_prof = None
        else:
            raise ValueError('``spatial_profile`` must either be string or dictionary.')

        # parse image and replace optional arguments with updated values
        self.image = self._parse_image(image, variance, mask, unit, disp_axis)
        variance = self.image.uncertainty.array
        mask = self.image.mask
        unit = self.image.unit

        img = np.ma.masked_array(self.image.data, mask=mask)

        # create separate mask including any previously uncaught non-finite
        # values for purposes of calculating fit
        or_mask = np.logical_or(img.mask,
                                ~np.isfinite(self.image.data))

        # If the trace is not flat, shift the rows in each column
        # so the image is aligned along the trace:
        if not isinstance(trace_object, FlatTrace):
            img = _align_along_trace(
                img,
                trace_object.trace,
                disp_axis=disp_axis,
                crossdisp_axis=crossdisp_axis)

        if self.spatial_profile == 'gaussian':

            # fit profile to average (mean) profile along crossdisp axis
            fit_ext_kernel = self._fit_gaussian_spatial_profile(img,
                                                                disp_axis,
                                                                crossdisp_axis,
                                                                or_mask,
                                                                bkgrd_prof)

            # this is just creating an array of the trace to shift the mean
            # when iterating over each wavelength. this needs to be fixed in the
            # future to actually account for the trace shape in a non-flat trace
            # (or possibly omitted all togehter as it might be redundant if
            # _align_along_trace is correcting this already)
            if isinstance(trace_object, FlatTrace):
                mean_init_guess = trace_object.trace
            else:
                mean_init_guess = np.broadcast_to(
                    img.shape[crossdisp_axis] // 2, img.shape[disp_axis]
                )

        else:  # interpolated_profile
            # for now, bkgrd_prof must be None because a compound model can't
            # be created with a interpolator + model. i think there is a way
            # around this, but will follow up later
            if bkgrd_prof is not None:
                raise ValueError('When `spatial_profile`is `interpolated_profile`,'
                                 '`bkgrd_prof` must be None. Background should'
                                 ' be fit and subtracted from `img` beforehand.')
            # make sure n_bins doesnt exceed the number of (for now) finite
            # columns. update this when masking is fixed.
            n_finite_cols = np.logical_or.reduce(or_mask, axis=crossdisp_axis)
            n_finite_cols = np.count_nonzero(n_finite_cols.astype(int) == 0)

            # determine interpolation degree from input and make tuple if int
            # this can also be moved to another method to parse the input
            # 'spatial_profile' arg, eventually
            if isinstance(interp_degree_interpolated_profile, int):
                kx = ky = interp_degree_interpolated_profile
            else:  # if input is tuple of ints

                if not isinstance(interp_degree_interpolated_profile, tuple):
                    raise ValueError("``interp_degree_interpolated_profile`` must be ",
                                     "an integer or tuple of integers.")
                if not all(isinstance(x, int) for x in interp_degree_interpolated_profile):
                    raise ValueError("``interp_degree_interpolated_profile`` must be ",
                                     "an integer or tuple of integers.")

                kx, ky = interp_degree_interpolated_profile

            if n_bins_interpolated_profile >= n_finite_cols:

                raise ValueError(f'`n_bins_interpolated_profile` ({n_bins_interpolated_profile}) '
                                 'must be less than the number of fully-finite '
                                 f'wavelength columns ({n_finite_cols}).')

            interp_spatial_prof = self._fit_self_spatial_profile(img, disp_axis,
                                                                 crossdisp_axis,
                                                                 or_mask,
                                                                 n_bins_interpolated_profile,
                                                                 kx, ky)

            # add private attribute to save fit profile. should this be public?
            self._interp_spatial_prof = interp_spatial_prof

        col_mask = np.logical_or.reduce(or_mask, axis=crossdisp_axis)
        nonf_col = [np.nan] * img.shape[crossdisp_axis]

        # array of 'x' values for each wavelength for extraction
        nrows = img.shape[crossdisp_axis]
        xd_pixels = np.arange(nrows)

        kernel_vals = []
        norms = []
        for col_pix in range(img.shape[disp_axis]):

            # for now, skip columns with any non-finite values
            # NOTE: fit and other kernel operations should support masking again
            # once a fix is in for renormalizing columns with non-finite values
            if col_mask[col_pix]:
                kernel_vals.append(nonf_col)
                norms.append(np.nan)
                continue

            if self.spatial_profile == 'gaussian':

                # set compound model's mean to column's matching trace value
                # again, this is probably not necessary
                if bkgrd_prof is not None:  # attr names will be diff. if not compound
                    fit_ext_kernel.mean_0 = mean_init_guess[col_pix]
                else:
                    fit_ext_kernel.mean = mean_init_guess[col_pix]

                # evaluate fit model (with shifted mean, based on trace)
                fitted_col = fit_ext_kernel(xd_pixels)

                # save result and normalization
                # this doesn't need to be in this loop, address later
                kernel_vals.append(fitted_col)

                norms.append(fit_ext_kernel.amplitude_0
                             * fit_ext_kernel.stddev_0 * np.sqrt(2*np.pi))

            else:  # interpolated_profile
                fitted_col = interp_spatial_prof(col_pix, xd_pixels)
                kernel_vals.append(fitted_col)
                norms.append(np.trapz(fitted_col, dx=1)[0])

        # transform fit-specific information
        kernel_vals = np.vstack(kernel_vals).T
        norms = np.array(norms)

        # calculate kernel normalization
        g_x = np.sum(kernel_vals**2 / variance, axis=crossdisp_axis)

        # sum by column weights
        weighted_img = np.divide(img * kernel_vals, variance)
        result = np.sum(weighted_img, axis=crossdisp_axis) / g_x

        # multiply kernel normalization into the extracted signal
        extraction = result * norms

        # convert the extraction to a Spectrum1D object
        return Spectrum1D(extraction * unit,
                          spectral_axis=self.image.spectral_axis)


@dataclass
class OptimalExtract(HorneExtract):
    """
    An alias for `HorneExtract`.
    """
    __doc__ += HorneExtract.__doc__
    pass
