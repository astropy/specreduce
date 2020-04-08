# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from astropy.table import Table


__all__ = ['trace', 'extract']


def _gaus(x, a, b, x0, sigma):
    """
    Define a simple Gaussian curve

    Could maybe be swapped out for astropy.modeling.models.Gaussian1D

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def trace(img, ilum=(1,), nbins=20, guess=-1, window=0, display=False):
    """
    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample trace along every X pixel.

    Parameters
    ----------
    img : 2d numpy array, or CCDData object
        This is the image to run trace over
    ilum : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    nbins : int, optional
        number of bins in wavelength (X) direction to chop image into. Use
        fewer bins if trace is having difficulty, such as with faint
        targets (default is 20, minimum is 4)
    guess : int, optional
        A guess at where the desired trace is in the spatial direction (Y). If set,
        overrides the normal max peak finder. Good for tracing a fainter source if
        multiple traces are present.
    window : int, optional
        If set, only fit the trace within a given region around the guess position.
        Useful for tracing faint sources if multiple traces are present, but
        potentially bad if the trace is substantially bent or warped.
    display : bool, optional
        If set to true display the trace over-plotted on the image

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis

    Improvements Needed
    -------------------
    1) switch to astropy models for Gaussian (?)
    2) return info about trace width (?)
    3) add re-fit trace functionality (or break off into another method)
    4) add other interpolation modes besides spline (?)
    """

    # define the wavelength & spatial axis, if we want to enable swapping programatically later
    # defined to agree with e.g.: img.shape => (1024, 2048) = (spatial, wavelength)
    Waxis = 1 # wavelength axis
    Saxis = 0 # spatial axis

    # Require at least 4 big bins along the trace to define shape. Sometimes can get away with very few
    if (nbins < 4):
        raise ValueError('nbins must be > 4')

    # if illuminated portion not defined, just use all of spatial axis extent
    if (len(ilum)<2):
        ilum = np.arange(img.shape[Saxis])

    # Pick the highest peak, bad if mult. obj. on slit...
    ztot = np.nansum(img, axis=Waxis)[ilum] / img.shape[Waxis] # average data across all wavelengths
    peak_y = ilum[np.nanargmax(ztot)]
    # if the user set a guess for where the peak was, adopt that
    if guess > 0:
        peak_y = guess

    # guess the peak width as the FWHM, roughly converted to gaussian sigma
    width_guess = np.size(ilum[ztot > (np.nanmax(ztot)/2.)]) / 2.355
    # enforce some (maybe sensible?) rules about trace peak width
    if width_guess < 2.:
        width_guess = 2.
    if width_guess > 25:
        width_guess = 25

    # [avg peak height, baseline, Y location of peak, width guess]
    peak_guess = [np.nanmax(ztot), np.nanmedian(ztot), peak_y, width_guess]

    # fit a Gaussian to peak
    popt_tot, pcov = curve_fit(_gaus, ilum, ztot, p0=peak_guess)

    if (window > 0):
        if (guess > 0):
            ilum2 = ilum[np.arange(guess-window, guess+window, dtype=np.int)]
        else:
            ilum2 = ilum[np.arange(popt_tot[2] - window, popt_tot[2] + window, dtype=np.int)]
    else:
        ilum2 = ilum

    xbins = np.linspace(0, img.shape[Waxis], nbins+1, dtype='int')
    ybins = np.zeros(len(xbins)-1, dtype='float') * np.NaN

    for i in range(0,len(xbins)-1):
        #-- fit gaussian w/i each window
        zi = np.nansum(img[ilum2, xbins[i]:xbins[i+1]], axis=Waxis)
        peak_y = ilum2[np.nanargmax(zi)]
        width_guess = np.size(ilum2[zi > (np.nanmax(zi) / 2.)]) / 2.355
        if width_guess < 2.:
            width_guess = 2.
        if width_guess > 25:
            width_guess = 25
        pguess = [np.nanmax(zi), np.nanmedian(zi), peak_y, width_guess]
        try:
            popt, _ = curve_fit(_gaus, ilum2, zi, p0=pguess)

            # if gaussian fits off chip, then fall back to previous answer
            if (popt[2] <= min(ilum2)) or (popt[2] >= max(ilum2)):
                ybins[i] = popt_tot[2]
            else:
                ybins[i] = popt[2]
                popt_tot = popt  # if a good measurment was made, switch to these parameters for next fall-back

        except RuntimeError:
            popt = pguess

    # recenter the bin positions
    xbins = (xbins[:-1] + xbins[1:]) / 2.

    yok = np.where(np.isfinite(ybins))[0]
    if len(yok) > 0:
        xbins = xbins[yok]
        ybins = ybins[yok]

        # run a cubic spline thru the bins
        ap_spl = UnivariateSpline(xbins, ybins, k=3, s=0)

        # interpolate the spline to 1 position per column
        mx = np.arange(0, img.shape[Waxis])
        my = ap_spl(mx)
    else:
        mx = np.arange(0, img.shape[Waxis])
        my = np.zeros_like(mx) * np.NaN
        import warnings
        warnings.warn("TRACE ERROR: No Valid points found in trace")

    if display is True:
        plt.figure()
        plt.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(img, (5, 98)))
        plt.scatter(xbins, ybins, alpha=0.5)
        plt.plot(mx, my)
        plt.show()

    return my


def extract(img, trace_line, apwidth=8, skysep=3, skywidth=7, skydeg=0,
            display=False):
    """
    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An major simplification at present. To be changed!

    2. Fits a polynomial to the sky at each column

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.

    3. Computes the uncertainty in each pixel

    Parameters
    ----------
    img : CCDData object
        This is the image to run extract over
    trace_line : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from `trace`
    apwidth : int, optional
        The width along the Y axis on either side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels, must be at least 1 pixel)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 3, must be at least 1 pixel)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 7, must be at least 1 pixel)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0)

    Returns
    -------
    extract table : astropy table, with columns:
        flux
            The summed flux at each column about the trace. Note: is not
            sky subtracted!
        fluxerr
            the uncertainties of the flux values
        skyflux
            The integrated sky values along each column, suitable for
            subtracting from the `flux` above


    Improvements Needed
    -------------------
    1. optionally take a wavelength solution, and interpolate sky region onto the
        same wavelengths as the trace (BIG IMPROVEMENT)
    2. optionally allow mode to be either simple aperture (current) or the
        "optimal" (variance weighted) extraction algorithm
    """

    if apwidth < 1:
        raise ValueError('apwidth must be >= 1')
    if skysep < 1:
        raise ValueError('skysep must be >= 1')
    if skywidth < 1:
        raise ValueError('skywidth must be >= 1')

    onedspec = np.zeros_like(trace_line)
    skysubflux = np.zeros_like(trace_line)
    fluxerr = np.zeros_like(trace_line)

    for i in range(0,len(trace_line)):
        #-- first do the aperture flux
        # juuuust in case the trace gets too close to an edge
        widthup = apwidth
        widthdn = apwidth
        if (trace_line[i]+widthup > img.shape[0]):
            widthup = img.shape[0]-trace_line[i] - 1
        if (trace_line[i]-widthdn < 0):
            widthdn = trace_line[i] - 1

        # simply add up the total flux around the trace_line +/- width
        onedspec[i] = np.nansum(img[int(trace_line[i]-widthdn):int(trace_line[i]+widthup+1), i])

        #-- now do the sky fit
        itrace_line = int(trace_line[i])
        y = np.append(np.arange(itrace_line-apwidth-skysep-skywidth, itrace_line-apwidth-skysep),
                      np.arange(itrace_line+apwidth+skysep+1, itrace_line+apwidth+skysep+skywidth+1))

        z = img[y,i]
        if (skydeg>0):
            # fit a polynomial to the sky in this column
            pfit = np.polyfit(y,z,skydeg)
            # define the aperture in this column
            ap = np.arange(trace_line[i]-apwidth, trace_line[i]+apwidth+1)
            # evaluate the polynomial across the aperture, and sum
            skysubflux[i] = np.nansum(np.polyval(pfit, ap))
        elif (skydeg==0):
            skysubflux[i] = np.nanmean(z)*(apwidth*2.0 + 1)

        #-- finally, compute the error in this pixel
        sigB = np.nanstd(z) # stddev in the background data
        N_B = np.float(len(y)) # number of bkgd pixels
        N_A = apwidth * 2. + 1 # number of aperture pixels

        # based on aperture phot err description by F. Masci, Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        fluxerr[i] = np.sqrt(np.nansum((onedspec[i]-skysubflux[i])) +
                             (N_A + N_A**2. / N_B) * (sigB**2.))

    if display:
        plt.figure()
        plt.imshow(img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(img, (5, 98)))

        plt.plot(np.arange(len(trace_line)), trace_line, c='C0')
        plt.fill_between(np.arange(len(trace_line)), trace_line + apwidth, trace_line-apwidth, color='C0', alpha=0.5)
        plt.fill_between(np.arange(len(trace_line)), trace_line + apwidth + skysep, trace_line + apwidth + skysep + skywidth, color='C1', alpha=0.5)
        plt.fill_between(np.arange(len(trace_line)), trace_line - apwidth - skysep, trace_line - apwidth - skysep - skywidth, color='C1', alpha=0.5)
        plt.ylim(np.min(trace_line - (apwidth + skysep + skywidth)*2), np.max(trace_line + (apwidth + skysep + skywidth)*2))
        # plt.show()

    tbl_out = Table()
    tbl_out.add_columns([onedspec * img.unit, fluxerr * img.unit, skysubflux * img.unit],
                        names=['flux', 'fluxerr', 'skyflux'])
    # return onedspec, skysubflux, fluxerr
    return tbl_out
