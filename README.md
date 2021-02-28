[![pip](https://img.shields.io/badge/pip-install%20autoeap-blue.svg)](https://pypi.org/project/autoeap/)
[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/zabop/autoeap/tree/master/examples)
[![Image](https://img.shields.io/badge/arXiv-1909.00446-blue.svg)](https://arxiv.org/abs/1909.00446)

# autoEAP - Automated Extended Aperture Photometry, developed for high amplitude K2 variable stars

The details of Extended Aperture Photometry are published in [Plachy et al.,2019,ApJS,244,32](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...32P/abstract).
A short summary of automatization is published [here](https://ui.adsabs.harvard.edu/abs/2020arXiv200908786P/abstract).

## Installation

To install the package, use:

```bash
pip install autoeap
```

## Example usage

To create your own photomery, you'll need a Target Pixel File, such as [this one.](https://github.com/zabop/autoeap/blob/master/examples/ktwo212466080-c17_lpd-targ.fits)
Then, after starting Python, you can do:

```python
yourtpf = '/path/to/your/tpf/ktwo212466080-c17_lpd-targ.fits'

import autoeap

time, flux, flux_err = autoeap.createlightcurve(yourtpf)
```

Or if you want to let autoEAP download the TPF from MAST database, you can just provide a target name and a campaign number:

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

Installation:
```
git clone https://github.com/OxES/k2sc.git
cd k2sc
python setup.py install --user
```
And then without much hassle, you can use in python:
```python
import autoeap

time, flux, flux_err = autoeap.createlightcurve(yourtpf,apply_K2SC=True)
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
We have also built-in a method to remove trends using low-order splines. Just do to correct the raw light curve:
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
 - `show_plots` If `True`, all the plots will be displayed.
 - `save_plots` If `True`, all the plots will be saved to a subdirectory.
 - `window_length` The length of filter window for spline correction given in days. Applies only if ``remove_spline`` is `True`. Default is `20` days.
 - `sigma_lower` The number of standard deviations to use as the lower bound for sigma clipping limit before spline correction. Applies only if ``remove_spline`` is `True`. Default is `3`.
 - `sigma_upper` The number of standard deviations to use as the upper bound for sigma clipping limit before spline correction. Applies only if ``remove_spline`` is `True`. Default is `3`.
  - `TH` Threshold to segment each target in each TPF candence. Only used if targets cannot be separated normally. Default is `8`. Do not change this value unless you are aware of what you are doing!
  - `ROI_lower` The aperture frequency grid range of interest threshold given in absolute number of selections above which pixels are considered to define the apertures.  Do not change this value unless you are aware of what you are doing! Default is `100`.
  - `ROI_upper` The aperture frequency grid range of interest threshold given in relative number of selections w.r.t. the number of all cadences below which pixels are considered to define the apertures. Do not change this value unless you are aware of what you are doing! Default is `0.85`.
- `**kwargs` Dictionary of arguments to be passed to ``k2sc.detrend``.

## Command-line tools
After installation, ``autoEAP`` will be available from the command line:

 - ``autoEAP <EPIC number or TPF path> [options]``

 Listed below are the usage instructions:

```bash
$ autoeap --help

usage: autoeap [-h] [--campaign <campaign-number>] [--applyK2SC] [--removespline] [--windowlength <window-length-in-days>] [--sigmalower <sigma-lower>] [--sigmaupper <sigma-upper>] [--saveplots]
               [--TH <threshold-value>] [--ROIlower <lower-ROI-value>] [--ROIupper <upper-ROI-value>]
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
  --windowlength <window-length-in-days>
                        The length of filter window for spline correction given in days. Default is 20 days.
  --sigmalower <sigma-lower>
                        The number of standard deviations to use as the lower bound for sigma clipping limit before spline correction. Default is 3.
  --sigmaupper <sigma-upper>
                        The number of standard deviations to use as the upper bound for sigma clipping limit before spline correction. Default is 3.
  --saveplots           Save all the plots that show each step into a subdirectory.
  --TH <threshold-value>
                        Threshold to segment each target in each TPF candence. Only used if targets cannot be separated normally. Default is 8.
  --ROIlower <lower-ROI-value>
                        The aperture frequency grid range of interest threshold given in absolute number of selections above which pixels are considered to define the apertures. Default is 100.
  --ROIupper <upper-ROI-value>
                        The aperture frequency grid range of interest threshold given in relative number of selections w.r.t. the number of all cadences below which pixels are considered to define the apertures. Default is 0.85.
```

## Data Access

We provide photometry for targets for the following Guest Observation Programs:
```GO12111,GO8037,GO13111,GO14058,GO6082,GO16058,GO18033,GO10037,GO15058,GO17033.```

Slightly less than 2000 RRLs. See: [K2 approved targets & programs.](https://keplerscience.arc.nasa.gov/k2-approved-programs.html)

The data we have already created have been uploaded to our [webpage](https://konkoly.hu/KIK/data_en.html).

## Contributing
Feel free to open PR / Issue, or contact us [here](bodi.attila@csfk.org) or [here](ps738@cam.ac.uk).

## Citing
If you find this code useful, please cite [Plachy et al.,2019,ApJS,244,32](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...32P/abstract), until the new paper is not ready. Here is the BibTeX source:
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
