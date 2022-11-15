import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, NDData, VarianceUncertainty
from astropy.utils.data import download_file

from specreduce.extract import HorneExtract
from specreduce.tracing import FlatTrace
from specutils import Spectrum1D, SpectralAxis

# fetch test image
fn = download_file('https://stsci.box.com/shared/static/exnkul627fcuhy5akf2gswytud5tazmw.fits',
                   cache=True)

# duplicate image in all accepted formats
# (one Spectrum1D variant has a physical spectral axis; the other is in pixels)
img = fits.getdata(fn).T
flux = img * u.MJy / u.sr
sax = SpectralAxis(np.linspace(14.377, 3.677, flux.shape[-1]) * u.um)
unc = VarianceUncertainty(np.random.rand(*flux.shape))

all_images = {}
all_images['arr'] = img
all_images['s1d'] = Spectrum1D(flux, spectral_axis=sax, uncertainty=unc)
all_images['s1d_pix'] = Spectrum1D(flux, uncertainty=unc)
all_images['ccd'] = CCDData(img, uncertainty=unc, unit=flux.unit)
all_images['ndd'] = NDData(img, uncertainty=unc, unit=flux.unit)
all_images['qnt'] = img * flux.unit

# save default values used for spectral axis and uncertainty when they are not
# available from the image object or provided by the user
sax_def = np.arange(img.shape[1]) * u.pix
unc_def = np.ones_like(img)


# (for use inside tests)
def compare_images(key, collection, compare='s1d'):
    # was input converted to Spectrum1D?
    assert isinstance(collection[key], Spectrum1D), (f"image '{key}' not "
                                                     "of type Spectrum1D")

    # do key's fluxes match its comparison's fluxes?
    assert np.allclose(collection[key].data,
                       collection[compare].data), (f"images '{key}' and "
                                                   f"'{compare}' have unequal "
                                                   "flux values")

    # if the image came with a spectral axis, was it kept? if not, was the
    # default spectral axis in pixels applied?
    sax_provided = hasattr(all_images[key], 'spectral_axis')
    assert np.allclose(collection[key].spectral_axis,
                       (all_images[key].spectral_axis if sax_provided
                        else sax_def)), (f"spectral axis of image '{key}' does "
                                         f"not match {'input' if sax_provided else 'default'}")

    # if the image came with an uncertainty, was it kept? if not, was the
    # default uncertainty created?
    unc_provided = hasattr(all_images[key], 'uncertainty')
    assert np.allclose(collection[key].uncertainty.array,
                       (all_images[key].uncertainty.array if unc_provided
                        else unc_def)), (f"uncertainty of image '{key}' does "
                                         f"not match {'input' if unc_provided else 'default'}")

    # were masks created despite none being given? (all indices should be False)
    assert (getattr(collection[key], 'mask', None)
            is not None), f"no mask was created for image '{key}'"
    assert np.all(collection[key].mask == 0), ("mask not all False "
                                               f"for image '{key}'")


# test consistency of general image parser results
def test_parse_general():
    all_images_parsed = {k: FlatTrace._parse_image(object, im)
                         for k, im in all_images.items()}

    for key in all_images_parsed.keys():
        compare_images(key, all_images_parsed)


# use verified general image parser results to check HorneExtract's image parser
def test_parse_horne():
    # HorneExtract's parser is more stringent than the general one, hence the
    # separate test. Given proper inputs, both should produce the same results.
    images_collection = {k: {} for k in all_images.keys()}

    for key, col in images_collection.items():
        img = all_images[key]
        col['general'] = FlatTrace._parse_image(object, img)

        if hasattr(all_images[key], 'uncertainty'):
            defaults = {}
        else:
            # save default values of attributes used in general parser when
            # they are not available from the image object. HorneExtract always
            # requires a variance, so it's chosen here to be on equal footing
            # with the general case
            defaults = {'variance': unc_def,
                        'mask': np.ma.masked_invalid(img).mask,
                        'unit': getattr(img, 'unit', u.DN)}

        col[key] = HorneExtract._parse_image(object, img, **defaults)

        compare_images(key, col, compare='general')
