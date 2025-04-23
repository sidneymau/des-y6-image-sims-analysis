import logging
import os
import functools
import operator
from pathlib import Path

import numpy as np

import galsim
import h5py
import hpgeom as hpg
import healsparse
import yaml
from scipy import interpolate, signal

from esutil.pbar import PBar
from ngmix.medsreaders import NGMixMEDS
from pizza_cutter_metadetect.masks import get_slice_bounds

from . import const


logger = logging.getLogger(__name__)


def flux_to_mag(flux):
    return const.ZEROPOINT - 2.5 * np.log10(flux)


def flux_to_mag_with_error(flux, flux_err):
    _mag = flux_to_mag(flux)
    # _mag_err = 2.5 / np.log(10) * _mag * flux_err
    _mag_err = -2.5 * np.log10(1 + flux_err / flux)
    return _mag, _mag_err


def mag_to_flux(mag):
    return np.power(
        10,
        -(mag - const.ZEROPOINT) / 2.5,
    )


def mag_to_flux_with_error(mag, mag_err):
    _flux = mag_to_flux(mag)
    _flux_err = np.log(10) / 2.5 * _flux * mag_err
    return _flux, _flux_err


def parse_shear_arguments(shear_string):
    return dict(
        map(
            lambda x: (x[0], eval(x[1])),
            map(
                lambda x: x.split("="),
                shear_string.split("__")
            )
        )
    )

# thanks @eli
def extractor(m, poly):
    pixels = poly.get_pixels(nside=m.nside_sparse)

    extracted = healsparse.HealSparseMap.make_empty(
        nside_coverage=m.nside_coverage,
        nside_sparse=m.nside_sparse,
        dtype=np.bool_,
        sentinel=False,
        bit_packed=True,
    )
    extracted[pixels] = m[pixels]

    return extracted


def load_wcs(tilename, band="r"):
    if isinstance(tilename, bytes):
        tilename = tilename.astype(str)

    image_paths = list(
        Path(os.environ["IMSIM_DATA"]).glob(
            f"des-pizza-slices-y6/{tilename}/sources-{band}/OPS_Taiga/multiepoch/*/*/{tilename}/*/coadd/{tilename}_*_{band}.fits.fz"
        )
    )
    assert len(image_paths) > 0, f"images for {tilename} not found"
    image_path = image_paths.pop().as_posix()
    if len(image_paths) > 0:
        logger.warning(f"Warning: found multiple images for {tilename}: {image_paths}")
    logger.info(f"Found following image for {tilename}: {image_path}")

    coadd_header = galsim.fits.FitsHeader(image_path)
    coadd_wcs, origin = galsim.wcs.readFromFitsHeader(coadd_header)

    return coadd_wcs


def get_tile_mask(tile, band, shear=None, mdet_mask=None, border=True):
    logger.info(f"computing effective tile mask for {tile}/{band}")
    wcs = load_wcs(
        tile,
        band=band,
    )
    match shear:
        case "plus":
            applied_shear = galsim.Shear(g1=0.02, g2=0.00)
        case "minus":
            applied_shear = galsim.Shear(g1=-0.02, g2=0.00)
        case _:
            applied_shear = galsim.Shear(g1=0.00, g2=0.00)

    if border:
        ra_vertices = [
            wcs.toWorld(galsim.PositionD(250, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(9750, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(9750, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(250, 0).shear(applied_shear)).ra.deg,
        ]
        dec_vertices = [
            wcs.toWorld(galsim.PositionD(0, 250).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 250).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 9750).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 9750).shear(applied_shear)).dec.deg,
        ]
    else:
        ra_vertices = [
            wcs.toWorld(galsim.PositionD(0, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(10000, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(10000, 0).shear(applied_shear)).ra.deg,
            wcs.toWorld(galsim.PositionD(0, 0).shear(applied_shear)).ra.deg,
        ]
        dec_vertices = [
            wcs.toWorld(galsim.PositionD(0, 0).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 0).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 10000).shear(applied_shear)).dec.deg,
            wcs.toWorld(galsim.PositionD(0, 10000).shear(applied_shear)).dec.deg,
        ]

    polygon = healsparse.Polygon(
        ra=ra_vertices,
        dec=dec_vertices,
        value=True,
    )

    if mdet_mask is not None:
        valid_map = extractor(mdet_mask, polygon)
    else:
        valid_map = polygon.get_map(
            nside_coverage=32,
            nside_sparse=131072,
            dtype=np.bool_,
        )

    return valid_map


def get_tile_area(tile, band, shear=None, mdet_mask=None, border=True):
    logger.info(f"computing effective tile area for {tile}/{band}")
    valid_map = get_tile_mask(tile, band, shear=shear, mdet_mask=mdet_mask, border=border)
    valid_area = valid_map.get_valid_area(degrees=True)
    logger.info(f"effective tile area for {tile}/{band}: {valid_area:.3f} deg^2")

    return valid_area


def load_mdet_mask(fname="/dvs_ro/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-v2.hsp"):
    logger.info(f"loading mdet mask from {fname}")
    hmap = healsparse.HealSparseMap.read(
        fname,
    )
    return hmap


def read_meds(fname):
    coadd_dims = (10_000, 10_000)

    m = NGMixMEDS(fname)

    obj_data = m.get_cat()
    meta = m.get_meta()
    pz_config = yaml.safe_load(meta['config'][0])
    buffer_size = int(pz_config['coadd']['buffer_size'])
    central_size = int(pz_config['coadd']['central_size'])

    full_image = np.zeros(coadd_dims, dtype=np.float32)

    for slice_ind in PBar(range(m.size), desc="reading slices"):
        obslist = m.get_obslist(slice_ind)
        scol = obj_data["orig_start_col"][slice_ind, 0]
        srow = obj_data["orig_start_row"][slice_ind, 0]
        slice_bnds = get_slice_bounds(
            orig_start_col=scol,
            orig_start_row=srow,
            central_size=central_size,
            buffer_size=buffer_size,
            coadd_dims=coadd_dims,
        )
        if len(obslist) > 0:
            img = obslist[0].image
            full_image[
                slice_bnds["min_row"]+srow:slice_bnds["max_row"]+srow,
                slice_bnds["min_col"]+scol:slice_bnds["max_col"]+scol,
            ] = img[
                slice_bnds["min_row"]:slice_bnds["max_row"],
                slice_bnds["min_col"]:slice_bnds["max_col"],
            ]
        else:
            full_image[
                slice_bnds["min_row"]+srow:slice_bnds["max_row"]+srow,
                slice_bnds["min_col"]+scol:slice_bnds["max_col"]+scol,
            ] = np.nan

    return full_image


# from https://github.com/beckermr/des-y6-analysis/blob/main/2024_10_21_fgmodels/des_y6_nz_modeling.py
def sompz_integral(y, x, low, high):
    low = np.minimum(x[-1], np.maximum(low, x[0]))
    high = np.minimum(x[-1], np.maximum(high, x[0]))
    low_ind = np.digitize(low, x)
    # for the lower index we do not use right=True, but
    # we still want to clip to a valid index of x
    low_ind = np.minimum(low_ind, x.shape[0] - 1)
    high_ind = np.digitize(high, x, right=True)
    dx = x[1:] - x[:-1]

    # high point not in same bin as low point
    not_in_single_bin = high_ind > low_ind

    # fractional bit on the left
    ileft = np.where(
        not_in_single_bin,
        (y[low_ind - 1] + y[low_ind])
        / 2.0
        * (1.0 - (low - x[low_ind - 1]) / dx[low_ind - 1])
        * dx[low_ind - 1],
        (y[low_ind - 1] + y[low_ind]) / 2.0 * (high - low),
    )

    # fractional bit on the right
    iright = np.where(
        not_in_single_bin,
        (y[high_ind - 1] + y[high_ind]) / 2.0 * (high - x[high_ind - 1]),
        0.0,
    )

    # central bits
    yint = (y[1:] + y[:-1]) / 2.0 * dx
    yind = np.arange(yint.shape[0])
    msk = (yind >= low_ind) & (yind < high_ind - 1)
    icen = np.where(
        np.any(msk),
        np.sum(
            np.where(
                msk,
                yint,
                np.zeros_like(yint),
            )
        ),
        0.0,
    )

    return ileft + icen + iright


# from Boyan
class Tz:
    def __init__(self, dz, nz, z0=None):
        '''Class representing sawtooth n(z) kernels (bins) in z.
        First kernel is centered at z0, which defaults to dz if not
        given.  If z0<dz, then its triangle is adjusted to go to zero at
        0, then peak at z0, down to zero at z0+dz/2.
        Arguments:
        `dz`: the step between kernel centers
        `nz`: the number of kernels.
        `z0`: peak of first bin'''
        self.dz = dz
        self.nz = nz
        if z0 is None:
            self.z0 = dz
        else:
            self.z0 = z0
        # Set a flag if we need to treat kernel 0 differently
        self.cut0 = self.z0<dz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.
        # SM: updated to use normal numpy instead of JAX

        if self.cut0 and k==0:
            # Lowest bin is unusual:
            out = np.where(z>self.z0, 1-(z-self.z0)/self.dz, z/self.z0)
            out = np.maximum(0., out) / ((self.z0+self.dz)/2.)
        else:
            out = np.maximum(0., 1 - np.abs((z-self.z0)/self.dz-k)) / self.dz
        return out

    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmax = np.arange(1,1+self.nz)*self.dz + self.z1
        zmin = zmax = 2*self.dz
        if self.cut0:
            zmin[0] = 0.
        return np.stack( (zmin, zmax), axis=1)

    def dndz(self,coeffs, z):
        '''Calculate dn/dz at an array of z values given set(s) of
        coefficients for the kernels/bins.  The coefficients will
        be normalized to sum to unity, i.e. they will represent the
        fractions within each kernel.
        Arguments:
        `coeffs`:  Array of kernel fractions of shape [...,nz]
        `z`:       Array of redshifts of arbitrary length
        Returns:
        Array of shape [...,len(z)] giving dn/dz at each z for
        each set of coefficients.'''

        # Make the kernel coefficients at the z's
        kk = np.array([self(k,z) for k in range(self.nz)])
        return np.einsum('...i,ij->...j',coeffs,kk) / np.sum(coeffs, axis=-1)

def rebin(nz):
    redshift_original = np.append(nz, np.array([0, 0]))
    zbinsc_laigle = np.arange(0,3.02,0.01)
    zbinsc_integrate = np.arange(0.015 ,3.015,0.01)

    interp_func = interpolate.interp1d(zbinsc_laigle, redshift_original, kind='linear', axis=0, bounds_error=False, fill_value=0)
    values = interp_func(zbinsc_integrate)


    values = values.reshape((60, 5))
    redshift_integrated = np.sum(values, axis=1)

    return redshift_integrated
