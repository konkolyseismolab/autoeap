"""
AutoEAP is the automated version of Extended Aperture Photometry
developed for K2 RR Lyrae stars.
"""

import numpy as np
import matplotlib.pyplot as plt
import lightkurve
import warnings

class autoeapFutureWarning(Warning):
    """Class for knowing that LightKurve 2.x fucked up my life."""
    pass

def apdrawer(limitc):
    intgrid=limitc*1
    down=[];up=[];left=[];right=[]
    for i, eachline in enumerate(intgrid):
        for j, each in enumerate(eachline):
            if each==1:
                down.append([[j,j+1],[i,i]])
                up.append([[j,j+1],[i+1,i+1]])
                left.append([[j,j],[i,i+1]])
                right.append([[j+1,j+1],[i,i+1]])

    together=[]
    for each in down: together.append(each)
    for each in up: together.append(each)
    for each in left: together.append(each)
    for each in right: together.append(each)

    filtered=[]
    for each in together:
        c=0
        for EACH in together:
            if each==EACH:
                c+=1
        if c==1:
            filtered.append(each)

    return filtered


def draw_a_single_aperture(tpf,cadence,segm,eachfile,show_plots=False,save_plots=False):

    colnums = np.arange( tpf.column, tpf.column+tpf.shape[2])
    rownums = np.arange( tpf.row,    tpf.row   +tpf.shape[1])

    fig = plt.figure()
    # Switch off warnings for nan,inf values
    with warnings.catch_warnings(record=True) as w:
        try: plt.imshow(np.log(20+tpf.flux[cadence].value),cmap='viridis',origin='lower')
        except AttributeError: plt.imshow(np.log(20+tpf.flux[cadence]),cmap='viridis',origin='lower')

    ax=plt.gca()

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
        rotation_mode="anchor")

    ax.set_xticks(np.arange(len(colnums)))
    ax.set_yticks(np.arange(len(rownums)))
    ax.set_xticklabels(colnums)
    ax.set_yticklabels(rownums)

    plt.title('C'+str(tpf.campaign)+' '+str(tpf.targetid)+'\nCadence no: '+str(cadence),fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label('log(Calibrated Flux)', rotation=270, fontsize=20)
    filtered=apdrawer((segm.data>0)*1)
    for x in range(len(filtered)):
        plt.plot(np.asarray(filtered[x][0])-0.5,np.asarray(filtered[x][1])-0.5,c='r',linewidth=12)
        plt.plot(np.asarray(filtered[x][0])-0.5,np.asarray(filtered[x][1])-0.5,c='w',linewidth=6)
    if save_plots: plt.savefig(eachfile+'_plots/'+eachfile+'_single_tpf_cadencenum_'+str(cadence)+'.png')
    if show_plots: plt.show()
    plt.close(fig)


def aperture_prep(inputfile,campaign=None,show_plots=False,save_plots=False):

    import os

    def isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    from sklearn.cluster import DBSCAN
    import matplotlib.gridspec as gridspec
    from lightkurve.utils import LightkurveWarning
    import os
    if isnotebook(): from tqdm.notebook import tqdm
    else: from tqdm import tqdm

    import autoeap.photutils_stable as photutils

    try:
        tpf = lightkurve.targetpixelfile.KeplerTargetPixelFile(inputfile)
        print('Using file '+inputfile)
    except OSError:
        print('Local TPF not found, trying to download TPF instead')
        result = lightkurve.search_targetpixelfile(inputfile,campaign=campaign)
        if len(result)>1:
            warnings.warn('The target has been observed in the following campaigns: '+\
            str(result.table['observation'].tolist())+\
            '. Only the first file has been downloaded. Please specify campaign (campaign=<number>) to limit your search.',
            LightkurveWarning)

            tpf = result[0].download()
        else:
            if len(result) == 0:
                raise FileNotFoundError('Empty search result. No target has been found in the given campaign!')
            tpf = result.download()
            print('TPF found on MAST: '+result.table['observation'].tolist()[0] )

    # Add underscores to output filenames
    inputfile = inputfile.replace(' ','_')
    inputfile = os.path.abspath(inputfile).split('/')[-1]

    # create folder to store plots
    if save_plots:
        try:
            os.mkdir(  os.path.abspath(inputfile+'_plots')  )
        except FileExistsError: pass

    campaignnum=tpf.campaign

    print('Finding PSF centroids and removing outliers')
    try:
        psfc1 = tpf.estimate_centroids()[0].value
        psfc2 = tpf.estimate_centroids()[1].value
    except AttributeError:
        warnings.warn('LightKurve version 2.x will force all numeric columns as Quantity',
                    autoeapFutureWarning)
        psfc1 = tpf.estimate_centroids()[0]
        psfc2 = tpf.estimate_centroids()[1]

    # Remove wrong cadences via detecting outlier photocenters
    if len(np.where(np.abs(psfc1)>1024)[0])==0 and len(np.where(np.abs(psfc2)>1024)[0])==0:

        newpsfc = np.c_[psfc1,psfc2]

        db = DBSCAN(eps=0.3, min_samples=10).fit(newpsfc)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

    else:
        core_samples_mask=np.full((1, len(tpf.flux)), True)[0]


    if save_plots or show_plots:
        fig = plt.figure(figsize=(4.5,5))

        gs = gridspec.GridSpec(5, 5)
        ax0 = plt.subplot(gs[1:, 1:])
        ax1 = plt.subplot(gs[0, 1:])
        ax2 = plt.subplot(gs[1:, 0])

        ax0.scatter(psfc1,psfc2,s=5)
        ax0.scatter(psfc1[np.where(core_samples_mask == False)],psfc2[np.where(core_samples_mask == False)],s=5,c='r')
        ax0.yaxis.tick_right()
        ax0.yaxis.set_label_position("right")
        ax0.set_xlabel('PSF CENTR1',fontsize=14)
        ax0.set_ylabel('PSF CENTR2',fontsize=14)

        try:
            ax1.scatter(tpf.time.value,psfc1,s=5)
            ax1.scatter(tpf.time[np.where(core_samples_mask == False)].value,psfc1[np.where(core_samples_mask == False)],s=5,c='r')
        except AttributeError:
            ax1.scatter(tpf.time,psfc1,s=5)
            ax1.scatter(tpf.time[np.where(core_samples_mask == False)],psfc1[np.where(core_samples_mask == False)],s=5,c='r')
        ax1.set_xlabel('BJD',fontsize=14)
        ax1.set_ylabel('PSF\nCENTR1',fontsize=14)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')

        try:
            ax2.scatter(psfc2, tpf.time.value,s=5)
            ax2.scatter(psfc2[np.where(core_samples_mask == False)],tpf.time[np.where(core_samples_mask == False)].value,s=5,c='r')
        except AttributeError:
            ax2.scatter(psfc2, tpf.time,s=5)
            ax2.scatter(psfc2[np.where(core_samples_mask == False)],tpf.time[np.where(core_samples_mask == False)],s=5,c='r')
        ax2.set_xlabel('PSF CENTR2',fontsize=14)
        ax2.set_ylabel('BJD',fontsize=14)
        if save_plots: plt.savefig(inputfile+'_plots/'+inputfile+'_PSF_centroid.png')
        if show_plots: plt.show()
        plt.close(fig)

    print('Optimizing apertures for each cadence')
    # Segment targets for each cadence
    try: countergrid_all = np.zeros_like(tpf.flux[0].value,dtype=np.int)
    except AttributeError: countergrid_all = np.zeros_like(tpf.flux[0],dtype=np.int)
    for i,tpfdata in tqdm(enumerate(tpf.flux[core_samples_mask]),total=len(tpf.flux[core_samples_mask])):
        # Switch off warnings fo detected nan,inf values
        with warnings.catch_warnings(record=True) as w:
            if i==0:
                threshold = photutils.detect_threshold(tpfdata, nsigma=1.8)
                segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None)
                use_meanstd_threshold = False
                if segm.nlabels==1:
                    # if there is only one target, check if it is a merger of two
                    threshold = np.mean(tpfdata)+.5*np.std(tpfdata)
                    segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None, connectivity=4)
                    if segm is None:
                        # if nothing found, go back to previuos state
                        threshold = photutils.detect_threshold(tpfdata, nsigma=1.8)
                        segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None)
                        use_meanstd_threshold = False
                    if segm.nlabels>1:
                        # managed to detect two targets
                        use_meanstd_threshold = True
                    else:
                        # still one target, go back to previuos state
                        threshold = photutils.detect_threshold(tpfdata, nsigma=1.8)
                        segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None)
                        use_meanstd_threshold = False
            elif use_meanstd_threshold:
                threshold = np.mean(tpfdata)+.5*np.std(tpfdata)
                segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None, connectivity=4)
            else:
                threshold = photutils.detect_threshold(tpfdata, nsigma=1.8)
                segm = photutils.detect_sources(tpfdata, threshold, npixels=1, filter_kernel=None)

        if i%1000==0:
            if save_plots or show_plots:
                draw_a_single_aperture(tpf,i,segm,inputfile,show_plots=show_plots,save_plots=save_plots)

        countergrid_all += (segm.data>0)*1 # IMPORTANT BIT, NOT TO LET segm.data TO CONTAIN OTHER VALUES THAN 0 & 1

    return countergrid_all, tpf, len(core_samples_mask), campaignnum



def plot_numofstars_vs_threshold(numfeatureslist,iterationnum,ROI,apindex,eachfile,show_plots=False,save_plots=False):

    #plt.figure(figsize=(10,3))
    fig = plt.figure()
    plt.title('Range Of Interest',fontsize=20)
    plt.plot(numfeatureslist)
    plt.axvline(x=apindex,c='r',alpha=0.5)

    plt.axvspan(ROI[0], ROI[1], alpha=0.2, color='blue')

    plt.xlabel('Threshold',fontsize=20)
    plt.ylabel('# stars found',fontsize=20)

    plt.tight_layout()
    if save_plots: plt.savefig(eachfile+'_plots/'+eachfile+'_plot_numofstars_vs_threshold'+str(iterationnum)+'.png')
    if show_plots: plt.show()
    plt.close(fig)


def pixelremoval(gapfilledaperturelist,variableindex):

    subgapfilledaperturelist = np.delete(gapfilledaperturelist,variableindex,axis=0)
    removethesepixels = subgapfilledaperturelist.sum(axis=0) > 0

    return removethesepixels


def defineaperture(numfeatureslist,countergrid_all,ROI,filterpassingpicsnum,TH):
    for apindex, nfeature in enumerate(numfeatureslist):
        if apindex>ROI[0] and apindex<ROI[1]:
            if nfeature>numfeatureslist[apindex-1]:
                # apindex used as threshold
                apertures=countergrid_all>apindex
                extensionprospects=True
                break
    else:
        apindex=int(filterpassingpicsnum/TH)
        apertures=(countergrid_all>apindex)
        extensionprospects=False

    return apertures, extensionprospects, apindex


def tpfplot(tpf,apindex,apertures,aps):

    fig = plt.figure()
    plt.title('Frame: '+str(apindex),fontsize=24)
    # Switch off warnings for nan,inf values
    with warnings.catch_warnings(record=True) as w:
        try: plt.pcolormesh(np.log(20+tpf.flux[apindex].value), cmap='viridis')
        except AttributeError: plt.pcolormesh(np.log(20+tpf.flux[apindex]), cmap='viridis')

    filtered=apdrawer(apertures)
    for x in range(len(filtered)):
        plt.plot(filtered[x][0],filtered[x][1],c='red', linewidth=8)

    for x in range(1,np.max(aps)+1):
        cords=np.where(x==aps)
        plt.text(cords[1][0],cords[0][0],str(x),fontsize=30)

    #if show_plots: plt.show()
    return fig


def apgapfilling(aperture):
    from scipy.ndimage import median_filter

    aper = aperture.copy()

    medfilt =  median_filter(aper,size=(3,3),mode='mirror')
    aper[medfilt] = True

    return aper*1

def which_one_is_a_variable(lclist,iterationnum,eachfile,show_plots=False,save_plots=False):

    from astropy.timeseries import LombScargle

    print('Iteration:', iterationnum)

    #iterationnum is a one based index, the number of the lc is zero based indexed
    chopped_power_max_over_mean=[]

    maxpowerlist=[]
    minpowerlist=[]

    nrows = len(lclist)
    fig,axs = plt.subplots(nrows,1,figsize=(12,2*nrows),squeeze=False)
    for ii,lc in enumerate(lclist):

        ls =  LombScargle(lc.time, lc.flux)
        frequency, power = ls.autopower(normalization='psd',
                                        minimum_frequency=2/lc.time.ptp(),
                                        nyquist_factor=1)
        axs[ii,0].plot(frequency, power,label='Target %d' % (ii+1))
        axs[ii,0].set_xlabel('Frequency',fontsize=20)
        axs[ii,0].set_ylabel('Power',fontsize=20)
        #plt.xlim([0, nyquist/2])
        #plt.xlim([2/lclist[q].time.ptp(), nyquist/2])
        axs[ii,0].set_ylim(bottom=0)
        axs[ii,0].legend()

        maxpowerlist.append(np.max(power[10:]))
        minpowerlist.append(np.min(power[10:]))

        chopped_freq=[]
        chopped_power=[]
        try:
            frequency = frequency.value
            power = power.value
        except AttributeError:
            pass
        for index, each in enumerate(frequency):
            #if each<(0.5*nyquist) and 0.1<each:
            if 0.1<each:
                chopped_freq.append(each)
                chopped_power.append(power[index])

        chopped_power_max_over_mean.append(np.max(chopped_power)/np.mean(chopped_power))
    #plt.ylim([-0.1,0.8])
    plt.tight_layout()
    if save_plots: plt.savefig(eachfile+'_plots/'+eachfile+'_Frequencyspace_iterationnum_'+str(iterationnum)+'.png')
    if show_plots: plt.show()
    plt.close(fig)

    return np.where(chopped_power_max_over_mean==np.max(chopped_power_max_over_mean))[0][0]


def afgdrawer(afg,filename, tpf,show_plots=False,save_plots=False):

    colnums = np.arange( tpf.column, tpf.column+tpf.shape[2])
    rownums = np.arange( tpf.row,    tpf.row   +tpf.shape[1])

    fig, ax = plt.subplots()
    im = ax.imshow(afg,cmap='viridis',origin='lower')

    ax.set_xticks(np.arange(len(colnums)))
    ax.set_yticks(np.arange(len(rownums)))
    ax.set_xticklabels(colnums)
    ax.set_yticklabels(rownums)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(colnums)):
        for j in range(len(rownums)):
            text = ax.text(i, j, afg[j, i],
                           ha="center", va="center", color="w")

    ax.set_title("Aperture frequency grid",fontsize=20)
    fig.tight_layout()

    if save_plots: plt.savefig(filename+'.png')
    if show_plots: plt.show()
    plt.close(fig)

def splinecalc(time,flux,window_length=20):
    from wotan import flatten
    from numpy import nanmean

    splinedLC, trendLC = flatten(time, flux,
                            method='rspline',
                            window_length=window_length,
                            return_trend=True,
                            break_tolerance=0.5,
                            edge_cutoff=False)

    splinedLC *= nanmean(flux)

    return splinedLC, trendLC


def createlightcurve(targettpf, apply_K2SC=False, remove_spline=False, save_lc=False, campaign=None, TH=8,
                        show_plots=False, save_plots=False, window_length=20):
    """
    ``createlightcurve`` performs photomerty on K2 variable stars

    Parameters
    ----------
    targettpf : string
        The location of the local TPF file  on which the photometry will
        be performed. If not found, TPF will be downladed from MAST, but
        ``campaign`` must be defined.
    apply_K2SC : bool, default: False
        If `True`, after the raw photomery, K2SC will be applied to remove
        systematics from the extracted light curve.
    remove_spline : bool, default: False
        If `True`, after the raw photomery, a low-order spline will be fitted
        and removed from the extracted light curve. If ``apply_K2SC`` is
        also `True`, then this step will be done after the K2SC.
    save_lc: bool, default: False
        If `True`, the final light curve will be save as a file.
    campaign : int, default: None
        If local TPF file is not found, it will be downloaded from MAST, but
        ``campaign`` number should be defined as well, if the target has been
        observed in more than one campaign.
    TH : int of float, default: 8
        Threshold to segment each target in each TPF candence. Only used if
        targets cannot be separated normally.
    show_plots: bool, default: False
        If `True`, all the plots will be displayed.
    save_plots: bool, default: False
        If `True`, all the plots will be saved to a subdirectory.
    window_length: int of float, default: 20
        The length of filter window for spline correction given in days. Applies only
        if ``remove_spline`` is `True`.
    Returns
    -------
    time : array-like
        Time values
    flux : array-like
        Raw flux values or K2SC corrected flux values, if ``apply_K2SC``
        is `True`.
    flux_err : array-like
        Flux error values
    """

    import scipy.ndimage.measurements as snm
    import os
    from astropy.io import ascii

    countergrid_all, tpf, filterpassingpicsnum, campaignnum = aperture_prep(targettpf,campaign=campaign,show_plots=show_plots,save_plots=save_plots)

    # Add underscores to output filenames
    targettpf = targettpf.replace(' ','_')
    targettpf = os.path.abspath(targettpf).split('/')[-1]

    #draw AFG before stacking TPFs
    if save_plots or show_plots:
        afgdrawer(countergrid_all,targettpf+'_plots/'+targettpf+'_AFG_before_stacking',tpf,show_plots=show_plots,save_plots=save_plots)

    print('Starting iteration')
    iterationnum=0
    while True:

        iterationnum+=1

        numfeatureslist=[]
        for th in range(np.max(countergrid_all)):
            prelimap=countergrid_all > th
            prelimap_int=prelimap*1
            # labelled_array, num_features=snm.label(prelimap_int)
            _, num_features=snm.label(prelimap_int)
            numfeatureslist.append(num_features)

        ROI=[100, len(tpf.flux)*0.85]
        apertures, extensionprospects, apindex = defineaperture(numfeatureslist,countergrid_all, ROI, filterpassingpicsnum, TH)

        if save_plots or show_plots:
            plot_numofstars_vs_threshold(numfeatureslist,iterationnum,ROI,apindex,targettpf,show_plots=show_plots)

        aps, numpeaks = snm.label(apertures)

        # Split each target aperture
        aperturelist = []
        for x in range(1, np.max(aps)+1):
            aperturelist.append(aps==x)

        # Filling gaps in apertures
        gapfilledaperturelist = []
        for each in aperturelist:
            gapfilledaperturelist.append(apgapfilling(each))
        gapfilledaperturelist = np.asarray(gapfilledaperturelist)>0

        if save_plots or show_plots:
            fig = tpfplot(tpf,apindex,apertures,aps)

            colorlist=['black','yellow','green','blue','cyan','magenta','white','black','yellow','green',
                       'blue','cyan','magenta','white','black','yellow','green','blue','cyan','magenta',
                       'white','black','yellow','green','blue','cyan','magenta','white']
            for i, ithap in enumerate(gapfilledaperturelist):
                filtered=apdrawer(ithap)
                for x in range(len(filtered)):
                    plt.plot(filtered[x][0],filtered[x][1],c=colorlist[i],linewidth=4)

            if save_plots: plt.savefig(targettpf+'_plots/'+targettpf+'_tpfplot_iternum_'+str(iterationnum)+'.png')
            if show_plots: plt.show()
            plt.close(fig)

        # Do photometry on each target
        fig,axs = plt.subplots(np.max(aps),1,figsize=(12,np.max(aps)*2),squeeze=False)
        lclist=[]
        for x in range(np.max(aps)):
            lc=tpf.to_lightcurve(aperture_mask=gapfilledaperturelist[x]).remove_nans().remove_outliers()
            lclist.append(lc)

            try: axs[x,0].plot(lc.time.value,lc.flux.value)
            except AttributeError: axs[x,0].plot(lc.time,lc.flux)
            axs[x,0].title.set_text('Target '+str(x+1))
            axs[x,0].set_xlabel('Time',fontsize=20)
            axs[x,0].set_ylabel('Flux',fontsize=20)
        plt.tight_layout()
        if save_plots: plt.savefig(targettpf+'_plots/'+targettpf+'_lc_iternum_'+str(iterationnum)+'.png')
        if show_plots: plt.show()
        plt.close(fig)

        variableindex = which_one_is_a_variable(lclist,iterationnum,targettpf,show_plots=show_plots)
        if save_plots or show_plots:
            fig = plt.figure(figsize=(20,4))
            try: plt.plot(lclist[variableindex].time.value,lclist[variableindex].flux.value,c='k')
            except AttributeError: plt.plot(lclist[variableindex].time,lclist[variableindex].flux,c='k')
            plt.xlabel('Time')
            plt.ylabel('Flux')
            plt.title('The lc which is identified as a variable')
            plt.tight_layout()
            if save_plots: plt.savefig(targettpf+'_plots/'+targettpf+'_lc_which_is_variable_iternum_'+str(iterationnum)+'.png')
            if show_plots: plt.show()
            plt.close(fig)

        if extensionprospects:
            '''If we have more than one target,
               we remove the non-variables, then detect stars again.'''

            removethesepixels = pixelremoval(gapfilledaperturelist,variableindex)

            countergrid_all = countergrid_all*np.invert(removethesepixels)*1

            #draw AFG after we removed non-variable targets
            if save_plots or show_plots:
                afgdrawer(countergrid_all,targettpf+'_plots/'+targettpf+'_AFG_after_'+str(iterationnum)+'_iterations',tpf,
                            show_plots=show_plots,save_plots=save_plots)

        else:
            print('Iteration finished')

            if apply_K2SC:
                print('Applying K2SC')

                from autoeap.k2sc_stable import psearch,k2sc_lc

                lclist[variableindex].primary_header = tpf.hdu[0].header
                lclist[variableindex].data_header = tpf.hdu[1].header
                lclist[variableindex].__class__ = k2sc_lc
                try:
                    period, fap = psearch(lclist[variableindex].time.value,lclist[variableindex].flux.value,min_p=0,max_p=4)
                except AttributeError:
                    period, fap = psearch(lclist[variableindex].time,lclist[variableindex].flux,min_p=0,max_p=4)

                lclist[variableindex].k2sc(campaign=campaignnum, kernel='quasiperiodic',kernel_period=period)

                if save_plots or show_plots:
                    fig = plt.figure(figsize=(20,4))
                    try: plt.plot(lclist[variableindex].time.value,lclist[variableindex].corr_flux)
                    except AttributeError: plt.plot(lclist[variableindex].time,lclist[variableindex].corr_flux)
                    plt.title('K2SC corrected lc for '+targettpf)
                    plt.xlabel('Time')
                    plt.ylabel('Flux')
                    if save_plots: plt.savefig(targettpf+'_plots/'+targettpf+'_k2sc_lc_plot.png')
                    if show_plots: plt.show()
                    plt.close(fig)

                if remove_spline:
                    print('Removing spline')
                    try:
                        splinedLC, trendLC = splinecalc(lclist[variableindex].time.value, lclist[variableindex].corr_flux,window_length=window_length)
                    except AttributeError:
                        splinedLC, trendLC = splinecalc(lclist[variableindex].time, lclist[variableindex].corr_flux,window_length=window_length)
                    if save_lc:
                        print('Saving lc as '+targettpf+'_autoEAP_lc_TH'+str(TH)+'_k2sc_spline.lc')
                        table = lclist[variableindex].to_table()['time','flux','flux_err']
                        table['corr_flux'] = lclist[variableindex].corr_flux
                        table['splined_flux'] = splinedLC
                        ascii.write(table,targettpf+'_autoEAP_lc_TH'+str(TH)+'_k2sc_spline.lc',overwrite=True)

                    print('Done')
                    try:
                        return lclist[variableindex].time.value, splinedLC, lclist[variableindex].flux_err.value
                    except AttributeError:
                        return lclist[variableindex].time, splinedLC, lclist[variableindex].flux_err

                if save_lc:
                    print('Saving lc as '+targettpf+'_autoEAP_lc_TH'+str(TH)+'_k2sc.lc')
                    table = lclist[variableindex].to_table()['time','flux','flux_err']
                    table['corr_flux'] = lclist[variableindex].corr_flux
                    ascii.write(table,targettpf+'_autoEAP_lc_TH'+str(TH)+'_k2sc.lc',overwrite=True)

                print('Done')
                try:
                    return lclist[variableindex].time.value, lclist[variableindex].corr_flux, lclist[variableindex].flux_err.value
                except AttributeError:
                    return lclist[variableindex].time, lclist[variableindex].corr_flux, lclist[variableindex].flux_err

            break

    if remove_spline:
        print('Removing spline')
        try:
            splinedLC, trendLC = splinecalc(lclist[variableindex].time.value, lclist[variableindex].flux.value,window_length=window_length)
        except AttributeError:
            splinedLC, trendLC = splinecalc(lclist[variableindex].time, lclist[variableindex].flux,window_length=window_length)
        if save_lc:
            print('Saving lc as '+targettpf+'_autoEAP_lc_TH'+str(TH)+'_spline.lc')
            table = lclist[variableindex].to_table()['time','flux','flux_err']
            table['splined_flux'] = splinedLC
            ascii.write(table,targettpf+'_autoEAP_lc_TH'+str(TH)+'_spline.lc',overwrite=True)

        print('Done')
        try:
            return lclist[variableindex].time.value, splinedLC, lclist[variableindex].flux_err.value
        except AttributeError:
            return lclist[variableindex].time, splinedLC, lclist[variableindex].flux_err

    if save_lc:
        print('Saving lc as '+targettpf+'_autoEAP_lc_TH'+str(TH)+'.lc')
        table = lclist[variableindex].to_table()['time','flux','flux_err']
        ascii.write(table,targettpf+'_autoEAP_lc_TH'+str(TH)+'.lc',overwrite=True)

    print('Done')
    try:
        return lclist[variableindex].time.value, lclist[variableindex].flux.value, lclist[variableindex].flux_err.value
    except AttributeError:
        return lclist[variableindex].time, lclist[variableindex].flux, lclist[variableindex].flux_err
