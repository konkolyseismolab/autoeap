import numpy as np
import autoeap
from numpy.testing import assert_array_almost_equal

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

def test_raw_lightcurve():
    time,flux,flux_err = autoeap.createlightcurve('EPIC220198696',campaign=8)

    lc = np.genfromtxt(os.path.join(PACKAGEDIR,"EPIC220198696_c8_autoEAP.lc"),skip_header=1).T

    assert_array_almost_equal(time,lc[0])
    assert_array_almost_equal(flux,lc[1].astype(np.float32))
    assert_array_almost_equal(flux_err,lc[2].astype(np.float32))
