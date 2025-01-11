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

from . import util


logger = logging.getLogger(__name__)


SHEAR_STEPS = [
    'g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0',
]

BANDS = ["g", "r", "i", "z"]


TRUTH_BASE = "/global/cfs/cdirs/desbalro/cosmos_simcat/"
TRUTH_CATALOGS = {}
for _truth_file in glob.glob(f"{TRUTH_BASE}/*.fits"):
    _tilename = _truth_file.split("/")[-1].split("_")[3]
    TRUTH_CATALOGS[_tilename] = _truth_file


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

    truth_table = fitsio.FITS(TRUTH_CATALOGS[tilename])[1]

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

        if (redshift > zlow) & (redshift <= zhigh):
            _shear = shear_slice
        else:
            _shear = shear_other
        logger.debug(f"Applying shear {_shear} to object with redshift {redshift}")

        world_pos = coord.CelestialCoord(ra=ra * coord.degrees, dec=dec * coord.degrees)
        u, v = wcs.center.project(world_pos, projection="gnomonic")
        pos = galsim.PositionD(u.rad, v.rad)

        sheared_pos = pos.shear(_shear)
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
                    for band in BANDS
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

    assert np.unique(_observed_matched_indices, return_counts=True)[1].max() == 1
    assert len(_truth_matched_indices) == len(_observed_matched_indices)

    observed_indices = catalog_indices[in_tile]
    observed_matched_indices = catalog_indices[in_tile][_observed_matched_indices]
    truth_matched = truth_table[_truth_matched_indices]

    n_unmatched = len(
        np.setdiff1d(
            observed_indices,
            observed_matched_indices,
        )
    )
    logger.info(f"{n_unmatched} of {len(observed_indices)} objects without match")

    return observed_indices, observed_matched_indices, truth_matched

    # observed_unmatched_indices = np.setdiff1d(
    #     np.indices((len(observed_points), )).ravel(),
    #     observed_matched_indices,
    #     assume_unique=True
    # )
    # truth_unmatched_indices = np.setdiff1d(
    #     np.indices((len(truth_points), )).ravel(),
    #     truth_matched_indices,
    #     assume_unique=True
    # )

    # assert np.unique(observed_unmatched_indices, return_counts=True)[1].max() == 1

    # return (
    #     observed_matched_indices, observed_unmatched_indices,
    #     truth_matched_indices, truth_unmatched_indices,
    # )

    # match_indices = {
    #     "observed": observed_matched_indices,
    #     "truth": truth_matched_indices,
    # }

    # return match_indices


def _main(catalog, shear_step):
    import h5py

    hf_imsim = h5py.File(catalog, mode="r")

    tilenames = np.unique(hf_imsim["mdet"]["noshear"]["tilename"][:].astype(str))

    print(f"matching {shear_step}")
    shear_args = dict(
        map(
            lambda x: (x[0], eval(x[1])),
            map(
                lambda x: x.split("="),
                shear_step.split("__")
            )
        )
    )

    match_filename = f"match_{shear_step}.hdf5"
    with h5py.File(match_filename, "w") as hf:
        for i, tilename in enumerate(tilenames):
            print(f"{tilename} ({i + 1} / {len(tilenames)})", end="\r", flush=True)

            _match_indices, _truth = match(
                tilename,
                hf_imsim,
                mdet_step="noshear",
                **shear_args,
            )

            tile_group = hf.create_group(tilename)
            mdet_group = tile_group.create_group("noshear")
            mdet_group.create_dataset("indices", data=_match_indices)
            mdet_group.create_dataset("redshift", data=_truth["photoz"])

        print(f"{match_filename} written")

    return 0

if __name__ == "__main__":
    imsim_base = "/global/cfs/cdirs/des/y6-image-sims/fiducial/"
    imsim_catalogs = {
        shear_step: os.path.join(
            imsim_base,
            shear_step,
            "metadetect_cutsv6_all.h5",
        )
        for shear_step in SHEAR_STEPS
    }

    results = {}
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        for shear_step, catalog in imsim_catalogs.items():
            future = executor.submit(_main, catalog, shear_step)
            futures[shear_step] = future

        # for shear_step, future in futures.items():
        #     results[shear_step] = future.result()

    # for shear_step, result in results.items():
    #     print(f"{shear_step}: {result}")
