import concurrent.futures
import functools
import logging
import glob
import os

import coord
import fitsio
import galsim
import numpy as np
from sklearn.neighbors import BallTree

from . import const, util


logger = logging.getLogger(__name__)


def match(
    tilename,
    hf_imsim,
    mdet_step="noshear",
    g1_slice=0.00,
    g2_slice=0.00,
    g1_other=0.00,
    g2_other=0.00,
    zlow=0.0,
    zhigh=6.0,
    query_radius_arcsec=1,
):
    if mdet_step != "noshear":
        raise ValueError(f"only noshear supported")

    wcs = util.load_wcs(tilename)

    logger.info(f"Loading {tilename} at {wcs.center.ra.deg, wcs.center.dec.deg}")

    in_tile = hf_imsim["mdet"][mdet_step]["tilename"][:] == str.encode(tilename)

    catalog_indices = np.indices(
        in_tile.shape,
    ).ravel()

    truth_table = fitsio.FITS(const.TRUTH_CATALOGS[tilename])[1]

    observed_table = {
        key: hf_imsim["mdet"][mdet_step][key][in_tile]
        for key in [
            "ra", "dec",
            "pgauss_band_flux_g", "pgauss_band_flux_err_g",
            "pgauss_band_flux_r", "pgauss_band_flux_err_r",
            "pgauss_band_flux_i", "pgauss_band_flux_err_i",
            "pgauss_band_flux_z", "pgauss_band_flux_err_z",
        ]
    }

    # https://github.com/des-science/montara/blob/main/montara/coadd_mixed_scene.py#L76-L94
    # https://github.com/des-science/montara/blob/main/montara/z_slice_shear.py#L5-L25

    shear_slice = galsim.Shear(g1=g1_slice, g2=g2_slice)
    shear_other = galsim.Shear(g1=g1_other, g2=g2_other)

    sheared_radec = []
    for i in range(truth_table.get_nrows()):
        ra = truth_table["ra_sim"][i]
        dec = truth_table["dec_sim"][i]
        redshift = truth_table["photoz"][i]

        world_pos = coord.CelestialCoord(ra=ra * coord.degrees, dec=dec * coord.degrees)
        u, v = wcs.center.project(world_pos, projection="gnomonic")
        pos = galsim.PositionD(u.rad, v.rad)

        if (redshift > zlow) & (redshift <= zhigh):
            sheared_pos = pos.shear(shear_slice)
        else:
            sheared_pos = pos.shear(shear_other)

        u2 = sheared_pos.x * coord.radians
        v2 = sheared_pos.y * coord.radians
        sheared_world_pos = wcs.center.deproject(u2, v2, projection="gnomonic")

        sheared_ra = sheared_world_pos.ra.deg
        sheared_dec = sheared_world_pos.dec.deg

        sheared_radec.append((sheared_ra, sheared_dec))

    truth_points = np.deg2rad(np.array(sheared_radec))

    observed_points = np.deg2rad(
        np.array([
            observed_table["ra"],
            observed_table["dec"],
        ]).T
    )

    logger.info("Constructing tree from truth points")
    bt = BallTree(
        truth_points,
        metric="haversine",
    )

    query_radius = np.deg2rad(query_radius_arcsec / 60 / 60)

    logger.info("Querying tree at observed points with radius {query_radius} rad")
    indices = bt.query_radius(
        observed_points,
        r=query_radius,
    )

    n_match = sum(map(len, indices))

    if n_match > 0:

        logger.info(f"Sorting matches by Chi^2")
        _min_chi2_indices = np.array([
            np.argsort(
                np.sum(
                    [
                        np.square(
                            np.divide(
                                np.subtract(
                                    observed_table[f"pgauss_band_flux_{band}"][_i],
                                    truth_table[f"flux_{band}"][indices[_i]]
                                ),
                                observed_table[f"pgauss_band_flux_err_{band}"][_i]
                            )
                        )
                        for band in const.BANDS
                    ],
                    axis=0
                )
            )[0] if len(indices[_i]) > 0 else 0
            for _i in range(len(observed_points))
        ])

        _observed_matched_indices = np.array([
            i
            for (i, _i) in enumerate(indices) if len(_i) > 0
        ])
        _truth_matched_indices = np.array([
            _i[_ii]
            for (_i, _ii) in zip(indices, _min_chi2_indices) if len(_i) > 0
        ])

        # _observed_matched_filter = np.isin(
        #     np.arange(len(indices)),
        #     _observed_matched_indices,
        # )

        # _truth_matched_filter = np.isin(
        #     np.arange(len(indices)),
        #     _truth_matched_indices,
        # )

        # assert np.unique(_observed_matched_indices, return_counts=True)[1].max() <= 1
        # assert len(_truth_matched_indices) == len(_observed_matched_indices)

        # observed_indices = catalog_indices[in_tile]
        observed_matched_indices = catalog_indices[in_tile][_observed_matched_indices]
        truth_matched = truth_table[_truth_matched_indices]
        # observed_matched_indices = catalog_indices[in_tile][_observed_matched_filter]
        # truth_matched = truth_table[_truth_matched_filter]

        # n_unmatched = len(
        #     np.setdiff1d(
        #         observed_indices,
        #         observed_matched_indices,
        #     )
        # )

        return observed_matched_indices, truth_matched

    else:
        return None, None

