# Purpose: Cython wrapper for the PDM module
# Author: Attila Bódi
# Version: 0.1  2021APR01

import cython
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void pdm_binned_step(
        double* t,
        double* y,
        double* w,
        double* freqs,
        double* power,
        const int ndata,
        const int nfreqs,
        const float dphi,
        const int nbins)

cdef extern void pdm_binned_linterp(
        double* t,
        double* y,
        double* w,
        double* freqs,
        double* power,
        const int ndata,
        const int nfreqs,
        const float dphi,
        const int nbins)

cdef extern void pdm_binless_tophat(
        double* t,
        double* y,
        double* w,
        double* freqs,
        double* power,
        const int ndata,
        const int nfreqs,
        const float dphi,
        const int nbins)

cdef extern void pdm_binless_gauss(
        double* t,
        double* y,
        double* w,
        double* freqs,
        double* power,
        const int ndata,
        const int nfreqs,
        const float dphi,
        const int nbins)

@cython.boundscheck(False)
@cython.wraparound(False)
def PDM(    np.ndarray[np.double_t, ndim=1, mode="c"] t not None,
            np.ndarray[np.double_t, ndim=1, mode="c"] y not None,
            np.ndarray[np.double_t, ndim=1, mode="c"] w not None,
            np.ndarray[np.double_t, ndim=1, mode="c"] freqs not None,
            float dphi=0.05,
            int nbins=10,
            str kind = 'binned_linterp',

    ):
    """
    ``Polyfit`` fits polynomial chain to phase curve

    Parameters
    ----------
    t: array
        time values5
    y: array
        Flux/mag values
    w: array
        weights
    freqs: array
        freqs
    """

    cdef int ndata  = t.size
    cdef int nfreqs = freqs.size
    cdef np.ndarray[np.float_t, ndim=1, mode="c"] power   = np.zeros((freqs.size),dtype=np.floating)

    # --- Call C funtion ---
    if kind == 'binned_step':
        pdm_binned_step(
                &t[0],
                &y[0],
                &w[0],
                &freqs[0],
                &power[0],
                ndata,
                nfreqs,
                dphi,
                nbins)
    elif kind == 'binned_linterp':
        pdm_binned_linterp(
                &t[0],
                &y[0],
                &w[0],
                &freqs[0],
                &power[0],
                ndata,
                nfreqs,
                dphi,
                nbins)
    elif kind == 'binless_tophat':
        pdm_binless_tophat(
                &t[0],
                &y[0],
                &w[0],
                &freqs[0],
                &power[0],
                ndata,
                nfreqs,
                dphi,
                nbins)
    elif kind == 'binless_gauss':
        pdm_binless_gauss(
                &t[0],
                &y[0],
                &w[0],
                &freqs[0],
                &power[0],
                ndata,
                nfreqs,
                dphi,
                nbins)
    else:
        raise KeyError('Function not available. Please use one of the followings: ' + \
                            'binless_tophat, binless_gauss, binned_linterp, binned_step')

    return power
