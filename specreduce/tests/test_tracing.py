from specreduce.utils.synth_data import make_2dspec_image
from specreduce.tracing import BasicTrace


# test basic tracing of row-parallel traces
def test_basictrace():
    im = make_2dspec_image()
    t_pos = im.shape[0] / 2
    t = BasicTrace(im, t_pos)

    assert(t[0] == t_pos)
    assert(t[0] == t[-1])
    assert(t.shape[0] == im.shape[1])

    t2 = BasicTrace(im)
    assert(t2.trace_pos == t_pos)

    t2(600)
    assert(t2.trace_pos == 600.)
