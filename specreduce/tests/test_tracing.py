import numpy as np

from specreduce.utils.synth_data import make_2dspec_image
from specreduce.tracing import Trace, FlatTrace, ArrayTrace

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
