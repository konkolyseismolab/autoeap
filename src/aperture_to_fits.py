# These functions are from lightkurve 1.x
# Source: https://github.com/lightkurve/lightkurve/blob/v1.11.x/lightkurve/targetpixelfile.py

from astropy.io import fits
import lightkurve
import os
import numpy as np

__all__ = ['save_aperture_fits']

def _boolean_mask_to_bitmask(aperture_mask):
    """Takes in an aperture_mask and returns a Kepler-style bitmask
    Parameters
    ----------
    aperture_mask : array-like
        2D aperture mask. The mask can be either a boolean mask or an integer
        mask mimicking the Kepler/TESS convention; boolean or boolean-like masks
        are converted to the Kepler/TESS conventions.  Kepler bitmasks are
        returned unchanged except for possible datatype conversion.
    Returns
    -------
    bitmask : numpy uint8 array
        A bitmask incompletely mimicking the Kepler/TESS convention: Bit 2,
        value = 3, means "pixel was part of the custom aperture".  The other
        bits have no meaning and are currently assigned a value of 1.
    """
    # Masks can either be boolean input or Kepler pipeline style
    clean_mask = np.nan_to_num(aperture_mask)

    contains_bit2 = (clean_mask.astype(np.int) & 2).any()
    all_zeros_or_ones = ( (clean_mask.dtype in ['float', 'int']) &
                            ((set(np.unique(clean_mask)) - {0,1}) == set()) )
    is_bool_mask = ( (aperture_mask.dtype == 'bool') | all_zeros_or_ones )

    if is_bool_mask:
        out_mask = np.ones(aperture_mask.shape, dtype=np.uint8)
        out_mask[aperture_mask == 1] = 3
        out_mask = out_mask.astype(np.uint8)
    elif contains_bit2:
        out_mask = aperture_mask.astype(np.uint8)
    else:
        out_mask = None
    return out_mask

def _header_template(extension):
    """Returns a template `fits.Header` object for a given extension."""
    template_fn = os.path.join(lightkurve.__path__[0], "data",
                               "tpf-ext{}-header.txt".format(extension))
    return fits.Header.fromtextfile(template_fn)


def save_aperture_fits(aperture_mask,filename,overwrite=False):
    """Saves an `ImageHDU` object containing the 'APERTURE' extension."""
    bitmask = _boolean_mask_to_bitmask(aperture_mask)
    hdu = fits.ImageHDU(bitmask)

    # Set the header from the template TPF again
    template = _header_template(2)
    for kw in template:
        if kw not in ['XTENSION', 'NAXIS1', 'NAXIS2', 'CHECKSUM', 'BITPIX']:
            hdu.header[kw] = (template[kw],
                              template.comments[kw])

    # Override the defaults where necessary
    for keyword in ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CUNIT1',
                    'CUNIT2', 'CDELT1', 'CDELT2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
            hdu.header[keyword] = ""  # override wcs keywords
    hdu.header['EXTNAME'] = 'APERTURE'

    hdu.writeto(filename,overwrite=overwrite)

    pass
