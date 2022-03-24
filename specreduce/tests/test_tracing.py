import numpy as np

from astropy.modeling import models
from specreduce.utils.synth_data import make_2dspec_image
from specreduce.tracing import Trace, FlatTrace, ArrayTrace, KosmosTrace

IM = make_2dspec_image()


# test basic trace class
def test_basic_trace():
    t_pos = IM.shape[0] / 2
    t = Trace(IM)

    assert(t[0] == t_pos)
    assert(t[0] == t[-1])
    assert(t.shape[0] == IM.shape[1])

    t.shift(100)
    assert(t[0] == 600.)

    t.shift(-1000)
    assert(np.ma.is_masked(t[0]))


# test flat traces
def test_flat_trace():
    t = FlatTrace(IM, 550.)

    assert(t.trace_pos == 550)
    assert(t[0] == 550.)
    assert(t[0] == t[-1])

    t.set_position(400.)
    assert(t[0] == 400.)

    t.set_position(-100)
    assert(np.ma.is_masked(t[0]))


# test array traces
def test_array_trace():
    arr = np.ones_like(IM[0]) * 550.
    t = ArrayTrace(IM, arr)

    assert(t[0] == 550.)
    assert(t[0] == t[-1])

    t.shift(100)
    assert(t[0] == 650.)

    t.shift(-1000)
    assert(np.ma.is_masked(t[0]))

    arr_long = np.ones(5000) * 550.
    t_long = ArrayTrace(IM, arr_long)

    assert(t_long.shape[0] == IM.shape[1])

    arr_short = np.ones(50) * 550.
    t_short = ArrayTrace(IM, arr_short)

    assert(t_short[0] == 550.)
    assert(np.ma.is_masked(t_short[-1]))
    assert(t_short.shape[0] == IM.shape[1])


# test KOSMOS trace algorithm
def test_kosmos_trace():
    # create image (process adapted from compare_extractions.ipynb)
    np.random.seed(7)
    nrows = 200
    ncols = 160
    sigma_pix = 4
    sigma_noise = 1

    col_model = models.Gaussian1D(amplitude=1, mean=nrows/2, stddev=sigma_pix)
    noise = np.random.normal(scale=sigma_noise, size=(nrows, ncols))

    index_arr = np.tile(np.arange(nrows), (ncols, 1))
    img = col_model(index_arr.T) + noise

    # calculate trace on normal image
    t = KosmosTrace(img)

    # test shifting
    shift_up = int(-img.shape[0]/4)
    t_shift_up = t.trace + shift_up

    shift_out = img.shape[0]

    t.shift(shift_up)
    assert np.sum(t.trace == t_shift_up) == t.trace.size, 'valid shift failed'

    t.shift(shift_out)
    assert t.trace.mask.all(), 'invalid values not masked'

    # create same-shaped variations of image with invalid values
    img_all_nans = np.tile(np.nan, (nrows, ncols))

    window = 10
    guess = int(nrows/2)
    img_win_nans = img.copy()
    img_win_nans[guess - window:guess + window] = np.nan

    # error on trace of otherwise valid image with all-nan window around guess
    try:
        KosmosTrace(img_win_nans, guess=guess, window=window)
    except ValueError as e:
        print(f"All-NaN window error message: {e}")
    else:
        raise RuntimeError('Trace was erroneously calculated on all-NaN window')

    # error on trace of all-nan image
    try:
        KosmosTrace(img_all_nans)
    except ValueError as e:
        print(f"All-NaN image error message: {e}")
    else:
        raise RuntimeError('Trace was erroneously calculated on all-NaN image')

    # could try to catch warning thrown for all-nan bins
