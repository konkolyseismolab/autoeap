import PDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import LombScargle
from scipy.optimize import minimize
from wotan import flatten
from astropy.stats import sigma_clip
import sys, os
import warnings

__all__ = ['detrend_wrt_PDM']

def weights(err):
    """ generate observation weights from uncertainties """
    w = np.power(err, -2)
    return w/sum(w)

def theta_by_coeffs(coeffs,*params):
    x,y,yerr,testf,const = params

    ydetrended = y.copy()
    for order,coeff in enumerate(coeffs[::-1]):
        ydetrended -= coeff*x**(order+1)
    ydetrended -= const

    w = weights( yerr )

    kind='binned_linterp'

    with warnings.catch_warnings(record=True) as warnmessage:
        testf = np.atleast_1d(testf)
        x =          np.asarray(x,dtype=np.floating)
        ydetrended = np.asarray(ydetrended,dtype=np.floating)
        testf =      np.asarray(testf,dtype=np.floating)
        w =          np.asarray(w,dtype=np.floating)
        pows = PDM.PDM(x, ydetrended, w, testf, kind=kind, nbins=50, dphi=0.05)

    return 1 - pows


def get_trend(x,y,err,freq,polyorder=5):
    p0 = np.polyfit(x,y,polyorder)
    const = p0[-1]

    # Set bounds to first and second orders to avoid divergence
    bounds = tuple([(-np.inf,np.inf) for _ in range(len(p0)-3)])
    try: bounds += ((p0[-3] - np.abs(p0[-3])*2., p0[-3] + np.abs(p0[-3])*2. ),)
    except IndexError: pass
    try: bounds += ((p0[-2] - np.abs(p0[-2])*1., p0[-2] + np.abs(p0[-2])*1. ),)
    except IndexError: pass

    if len(bounds) == 0: bounds=None

    res = minimize(theta_by_coeffs, p0[:-1], args=(x,y,err,freq,const), method='Powell', bounds=bounds)

    return res,const

def clean_lightcurve(x,y,sigma=4,plotting=False):
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    goodpts = np.isfinite(y)
    freq,power = LombScargle(x[goodpts], y[goodpts]).autopower( normalization='psd',
                                                                nyquist_factor=0.8,
                                                                minimum_frequency=1/x.ptp(),
                                                                samples_per_peak=50)
    bestper = 1/freq[np.argmax(power)]

    with HiddenPrints():
        # Fit detrended data with cosine to remove outliers
        flatten_lc, cos_trend = flatten(x, y,
                                    window_length=bestper*0.5,
                                    return_trend=True,
                                    method='median')

    # Handle if trend is all nan
    if np.all(np.isnan(cos_trend)):
        flatten_lc = y
        cos_trend  = np.ones_like(cos_trend) * np.mean(y)

    # Find outliers after removing cosine fit
    goodpts = sigma_clip(flatten_lc, sigma_lower=sigma, sigma_upper=sigma, maxiters=5, cenfunc='mean', stdfunc='std')
    goodpts = ~goodpts.mask

    if plotting:
        fig = plt.figure(figsize=(14,5))
        plt.title('Original & w/out outliers')
        plt.plot(x[goodpts], y[goodpts],'k.', ms=2)
        plt.plot(x[~goodpts], y[~goodpts],'rx', ms=6)
        plt.plot(x,cos_trend,c='C1')
        plt.show()
        plt.close()

    return  goodpts


def detrend_wrt_PDM(time,flux,fluxerr,polyorder='auto',sigma=10,show_plots=False,save_plots=False,filename=None,debug=False):
    """
    Deterending with a polynomial that is fitted w.r.t. phase dispersion minimization.

    Parameters
    ----------
    time : array-like
        Time values.
    flux : array-like
        Corresponding flux values.
    fluxerr : array-like
        Corresponding flux errors.
    polyorder : int , default : 9
        The order of the detrending polynomial.
    show_plots: bool, default: False
        If `True` the plot will be displayed.
    save_plots: bool, default: False
        If `True` the plot will be saved to a subdirectory.
    filename : string
        If `save_plots` is `True`, the name of the output PNG file,
        without extension.
    debug : bool, default: False
        If `True` debug plots/prints will be displayed.

    Returns
    -------
    corrflux : array-like
        Detrended flux values.
    """

    with warnings.catch_warnings(record=True) as w:
        x = np.ascontiguousarray(time, dtype=np.float64)
        yorig = np.ascontiguousarray(flux, dtype=np.float64)
        err = np.ascontiguousarray(fluxerr, dtype=np.float64)
    y = yorig.copy()
    xorig = x.copy()
    errorig = err.copy()

    goodpts = np.isfinite(y)
    x = x[goodpts]
    y = y[goodpts]
    err = err[goodpts]

    # flatten cannot handle negative values
    if y.mean() <=0:
        y -= y.min()
        yorig -= yorig.min()

    ## Clean light curve roughly by rolling median filtering in time and sigma clipping

    goodpts = clean_lightcurve(x,y,sigma=sigma,plotting=debug)

    # Keep only good points
    x = x[goodpts]
    y = y[goodpts]
    err = err[goodpts]

    # Get period
    lsfreqs,lspow = LombScargle(x,y).autopower(nyquist_factor=1,samples_per_peak=50,minimum_frequency=0.05)
    testf = lsfreqs[np.argmax(lspow)]

    # If test period is larger than data length, remove linear
    if testf <= 1/np.ptp(x):
        y -= np.poly1d(np.polyfit(x,y,1))(x)

        lsfreqs,lspow = LombScargle(x,y).autopower(normalization='psd',
                                                nyquist_factor=1,
                                                minimum_frequency=0.05,
                                                samples_per_peak=50)

        testf = lsfreqs[np.argmax(lspow)]

        # If period is still larger than data length
        if testf <= 1/np.ptp(x):
            testf = 1/np.ptp(x)

    del lsfreqs,lspow

    if debug:
        print('Best period is', round(1/testf,8),'days' )

        fig,ax = plt.subplots(1,2,figsize=(14,5))
        ax[1].plot( (x*testf)%1, y, '.', ms=2)
        ax[0].plot( x, y, '.', ms=2)
        plt.show()
        plt.close()

    #### Normalize light curve
    meanx = np.nanmean(x)
    meany = np.nanmean(y)
    x -= meanx
    y -= meany

    # Estimate polyorder based on covered cycles
    if polyorder == 'auto':
        if x.ptp()/(1/testf) < 5:
            polyorder = 0
        elif x.ptp()/(1/testf) < 200:
            polyorder = 5
        elif x.ptp()/(1/testf) < 500:
            polyorder = 7
        else:
            polyorder = 9

    # Run PDM-based polynomial fitting
    res,const = get_trend(x,y,err,freq=testf,polyorder=polyorder)

    # If PDM-based trend fit fails use simple polyfit instead
    if np.std( y-np.poly1d(np.append(res.x,const))(x) )/np.std(y) > 1.0e3:
        class result:
            def __init__(self,x):
                self.x = x

        _pol = np.polyfit(x,y,polyorder)
        res = result(_pol[:-1])
        const = _pol[-1]

        del _pol
        del result

    ### Plot detrending

    gs = GridSpec(3,2)

    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,:])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[2,1])

    ax1.set_title('Black: inital polinomial -- Red: final polynomial')
    ax1.scatter(x+meanx-2454833,y+meany,s=4,c=x,cmap='copper')
    ax1.plot(x+meanx-2454833,  np.poly1d(np.polyfit(x,y,polyorder))(x) +meany,'k')

    ax1.plot(x+meanx-2454833, np.poly1d(np.append(res.x,const))(x) +meany,'r' )
    ax1.grid()
    ax1.set_ylabel('Flux')

    ax2.set_title('Corrected light curve')
    ax2.scatter(x+meanx-2454833, y-np.poly1d(np.append(res.x,const))(x) +meany,s=4,c=x,cmap='copper'  )
    ax2.grid()
    ax2.set_ylabel('Flux')
    ax2.set_xlabel('Time - 2454833 [BKJD days]')

    ax3.set_title('Original')
    ax3.scatter( (x*testf)%1, y+meany, c=x, s=4,cmap='copper')
    ax3.grid()
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Flux')
    ax4.set_title('Corrected')
    ax4.scatter( (x*testf)%1, y-np.poly1d(np.append(res.x,const))(x) +meany, c=x, s=4,cmap='copper')
    ax4.grid()
    ax4.set_xlabel('Phase')

    plt.tight_layout()
    if save_plots and filename is not None: plt.savefig(filename+'.png',format='png',dpi=80)
    if show_plots: plt.show()
    plt.close()

    ### Update variables

    ycorr = yorig - np.poly1d(np.append(res.x,const))(xorig-meanx)

    ### Return detrended light curve

    return ycorr
