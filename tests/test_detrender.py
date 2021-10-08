import numpy as np
from autoeap.detrender import detrend_wrt_PDM
from numpy.testing import assert_array_almost_equal

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

def test_detrender():
    time,flux,flux_err = np.genfromtxt(os.path.join(PACKAGEDIR,"EPIC220198696_c8_autoEAP.lc"),skip_header=1,unpack=True)

    corr_flux = detrend_wrt_PDM(time,flux,flux_err)

    corr_flux_orig = np.genfromtxt(os.path.join(PACKAGEDIR,"EPIC220198696_c8_autoEAP.lc_corr"))

    assert_array_almost_equal(corr_flux_orig,corr_flux)
