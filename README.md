<div align="center">
<img src="docs/autoeap_logo.png" width="50%">

## **Automated Extended Aperture Photometry**

[![pip](https://img.shields.io/badge/pip-install%20autoeap-blue.svg)](https://pypi.org/project/autoeap/)
[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/zabop/autoeap/tree/master/examples)
[![Image](https://img.shields.io/badge/arXiv-1909.00446-blue.svg)](https://arxiv.org/abs/1909.00446)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

</div>

--------------------------------------------------------------------------------

The Automated Extended Aperture Photometry (autoEAP) is developed for high amplitude K2 variable stars. The details of EAP are published in [Plachy et al.,2019,ApJS,244,32](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...32P/abstract).
A short summary of automatization is published [here](https://ui.adsabs.harvard.edu/abs/2020arXiv200908786P/abstract).

# Installation and dependencies

To install the package, use:

```bash
pip install numpy cython autoeap
```

## Example usage

To create your own photomery, you'll need a Target Pixel File, such as [this one.](https://github.com/zabop/autoeap/blob/master/examples/ktwo212466080-c17_lpd-targ.fits)
Then, after starting Python, you can do:

```python
yourtpf = '/path/to/your/tpf/ktwo212466080-c17_lpd-targ.fits'

import autoeap

time, flux, flux_err = autoeap.createlightcurve(yourtpf)
```

Or if you want to let autoEAP download the TPF from MAST database, you can just provide a target name and _optionally_ a campaign number:

```python
import autoeap

targetID = 'EPIC 212466080'
campaign = 17
time, flux, flux_err = autoeap.createlightcurve(targetID,campaign=campaign)
```

**With this last line, you can create autoEAP photometry for any K2 variable star.**

Plotting our results gives:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(time,flux,'.')
plt.xlabel('KBJD')
plt.ylabel('Flux')
plt.show()
```
![example scatter plot2](https://raw.githubusercontent.com/zabop/autoeap/master/docs/ktwo212466080-c17_raw.jpg)

The details of the workflow is described in [docs](https://github.com/zabop/autoeap/tree/master/docs).

You can find Google Colab friendly tutorials [in the examples](https://github.com/zabop/autoeap/tree/master/examples).

### Apply K2 Systematics Correction (K2SC)
If you want to apply K2SC correction for your freshly made raw-photometry, first you should install [K2SC](https://github.com/OxES/k2sc). AutoEAP is based on that package, so if you find K2SC useful, please cite [Aigrain et al.,2016,MNRAS,459,2408](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.2408A/abstract).

Installation with ``george``:
```bash
pip install george k2sc
```

And then without much hassle, you can apply the correction in python:
```python
import autoeap

targetID = 'EPIC 212466080'
campaign = 17
time, flux, flux_err = autoeap.createlightcurve(targetID,campaign=campaign,apply_K2SC=True)
```

The result is quite delightful:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(time,flux,'.')
plt.xlabel('KBJD')
plt.ylabel('Flux')
plt.show()
```
![k2sc result](https://raw.githubusercontent.com/zabop/autoeap/master/docs/ktwo212466080-c17_k2sc.jpg)

### Apply spline correction
We have also built-in a method to remove trends using low-order splines. To correct the raw light curve, do:
```python
import autoeap

time, flux, flux_err = autoeap.createlightcurve(yourtpf,remove_spline=True)
```

Or do this to remove a spline from the K2SC light curve:
```python
import autoeap

time, flux, flux_err = autoeap.createlightcurve(yourtpf,apply_K2SC=True,remove_spline=True)
```

## Available options
 - `apply_K2SC` If `True`, after the raw photomery, K2SC will be applied to remove systematics from the extracted light curve.
 - `remove_spline` If `True`, after the raw photomery, a low-order spline will be fitted and removed from the extracted light curve. If ``apply_K2SC`` is also `True`, then this step will be done after the K2SC.
 - `save_lc` If `True`, the intermediate and the final light curves will be save as a file.
 - `campaign` If local TPF file is not found, it will be downloaded from MAST, but ``campaign`` number should be defined as well, if the target has been observed in more than one campaign. Otherwise that campaign will be used in which the target were first observed.
 - `show_plots` If `True`, all the plots that visualize each step of photometry will be displayed.
 - `save_plots` If `True`, all the plots that visualize each step of photometry will be saved to a subdirectory.
 - `polyorder` The order of the detrending polynomial. Applies only if ``remove_spline`` is `True`. Default is `'auto'`.
 - `sigma_detrend` The number of standard deviations to use for sigma clipping limit before spline correction. Applies only if ``remove_spline`` is `True`. Default is `10`.
 - `max_missing_pos_corr` Maximum number of missing position correction (POS_CORR) values.
     If too many POS_CORR is missing, then less reliable photometrically
     estimated centroids will be used for K2SC. Missing POS_CORR values
     reduce the number of light curve points! Default is `10`.
 - `TH` Threshold to segment each target in each TPF candence. Only used if targets cannot be separated normally. Default is `8`. Do not change this value unless you are aware of what you are doing!
 - `ROI_lower` The aperture frequency grid range of interest threshold given in absolute number of selections above which pixels are considered to define the apertures.  Do not change this value unless you are aware of what you are doing! Default is `100`.
 - `ROI_upper` The aperture frequency grid range of interest threshold given in relative number of selections w.r.t. the number of all cadences below which pixels are considered to define the apertures. Do not change this value unless you are aware of what you are doing! Default is `0.85`.
 - `**kwargs` Dictionary of arguments to be passed to ``k2sc.detrend``. [See options here](https://github.com/OxES/k2sc/blob/master/src/standalone.py#L41).

## Command-line tools
After installation, ``autoEAP`` will be available from the command line:

 - ``autoEAP <EPIC number or TPF path> [options]``

 Listed below are the usage instructions:

```bash
$ autoeap --help

usage: autoeap [-h] [--campaign <campaign-number>] [--applyK2SC]
               [--removespline] [--polyorder <detrending-polynomial-order>]
               [--sigmadetrend <detrending-sigma-limit>] [--saveplots]
               [--maxmissingposcorr <max-missing-pos-corr>]
               [--TH <threshold-value>] [--ROIlower <lower-ROI-value>]
               [--ROIupper <upper-ROI-value>]
               <path-to-targettpf-or-EPIC-number>

Perform autoEAP photometry on K2 variable stars.

positional arguments:
  <path-to-targettpf-or-EPIC-number>
                        The location of the local TPF file or an EPIC number to be downloaded from MAST. Valid inputs include: The name of the object as a string, e.g. 'ktwo251812081'. The EPIC identifier as an integer, e.g. 251812081.

optional arguments:
  -h, --help            show this help message and exit
  --campaign <campaign-number>
                        If the target has been observed in more than one campaign, download this light curve. If not given, the first campaign will be downloaded.
  --applyK2SC           After the raw photomery, apply K2SC to remove systematics from the extracted light curve.
  --removespline        After the raw or K2SC photomery, remove a low-order spline from the extracted light curve.
  --polyorder <detrending-polynomial-order>
                        The order of the detrending polynomial. Default is auto.
  --sigmadetrend <detrending-sigma-limit>
                        The number of standard deviations to use for sigma clipping limit before spline correction. Default is 10.0
  --saveplots           Save all the plots that show each step into a subdirectory.
  --maxmissingposcorr <max-missing-pos-corr>
                        Maximum number of missing position correction (POS_CORR) values. If too many POS_CORR is missing, then less reliable photometrically estimated centroids will be used for K2SC. Missing POS_CORR values reduce the number of light curve points!
  --TH <threshold-value>
                        Threshold to segment each target in each TPF candence. Only used if targets cannot be separated normally. Default is 8.
  --ROIlower <lower-ROI-value>
                        The aperture frequency grid range of interest threshold given in absolute number of selections above which pixels are considered to define the apertures. Default is 100.
  --ROIupper <upper-ROI-value>
                        The aperture frequency grid range of interest threshold given in relative number of selections w.r.t. the number of all cadences below which pixels are considered to define the apertures. Default is 0.85.
```

## Data Access

We provide photometry for targets for the following Guest Observation Programs:
GO0055,
GO1018,
GO2040,
GO3040,
GO4069,
GO5069,
GO6082,
GO7082,
GO8037,
GO10037,
GO12111,
GO13111,
GO14058,
GO15058,
GO16058,
GO17033,
GO18033,
GO19033.

Slightly less than 2000 RRLs. See: [K2 approved targets & programs.](https://keplergo.github.io/KeplerScienceWebsite/k2-approved-programs.html)

The data we have already created have been uploaded to our [webpage](https://konkoly.hu/KIK/data_en.html).

## Standalone PDM-based detrender

The PDM-based polynomial fitting and detrending can be used as a standalone module to be applied to e.g. raw `autoEAP` light curves or any other data sets given a time series with times, fluxes and flux errors. The `detrend_wrt_PDM` method returns the corrected flux values.

```python
from autoeap.detrender import detrend_wrt_PDM

corrflux = detrend_wrt_PDM(time,               # Time values
                           flux,               # Flux values
                           flux_err,           # Flux errors
                           polyorder='auto',   # Polynomial order or 'auto'
                           sigma=10,           # Sigma value for sigma clipping before PDM calculation
                           show_plots=True,    # Show the detrending process
                           save_plots=False,   # Save the detrending process plot
                           filename=None)      # to this file as PNG.
```

## Contributing
Feel free to open PR / Issue, or contact us [here](bodi.attila@csfk.org) or [here](ps738@cam.ac.uk).

## Citing
If you use data provided by this pipeline please cite [Plachy et al.,2019,ApJS,244,32](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...32P/abstract), until the new paper is not ready. Here is the BibTeX source:
```
@ARTICLE{2019ApJS..244...32P,
       author = {{Plachy}, Emese and {Moln{\'a}r}, L{\'a}szl{\'o} and {B{\'o}di}, Attila and {Skarka}, Marek and {Szab{\'o}}, P{\'a}l and {Szab{\'o}}, R{\'o}bert and {Klagyivik}, P{\'e}ter and {S{\'o}dor}, {\'A}d{\'a}m and {Pope}, Benjamin J.~S.},
        title = "{Extended Aperture Photometry of K2 RR Lyrae stars}",
      journal = {\apjs},
     keywords = {RR Lyrae variable stars: 1410, Light curves (918, Space telescopes (1547, 1410, 918, 1547, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2019,
        month = oct,
       volume = {244},
       number = {2},
          eid = {32},
        pages = {32},
          doi = {10.3847/1538-4365/ab4132},
archivePrefix = {arXiv},
       eprint = {1909.00446},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJS..244...32P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Acknowledgements
This project was made possible by the funding provided by the National Research, Development and Innovation Office of Hungary, funding granted under project 2018-2.1.7-UK_GYAK-2019-00009 and by the Lendület Program of the Hungarian Academy of Sciences, project No LP2018-7.
