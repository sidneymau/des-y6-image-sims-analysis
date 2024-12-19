from pathlib import Path
import functools
import os

import tqdm
import h5py
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

import smatch
import healsparse

import tomography
import util


TOMOGRAPHIC_BINS = [0, 1, 2, 3]
MDET_STEPS = ["noshear", "1p", "1m", "2p", "2m"]
SHEAR_STEPS = [
    'g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1',
    'g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8',
    'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0',
]

tomo_colors = {
    0: "blue",
    1: "gold",
    2: "green",
    3: "red",
}


NBINS = 100

imsim_base = "/global/cfs/cdirs/des/y6-image-sims/fiducial-400/"
imsim_catalogs = {
    shear_step: os.path.join(
        imsim_base,
        shear_step,
        "metadetect_cutsv6_all.h5",
    )
    for shear_step in SHEAR_STEPS
}

redshift_base = "/global/cfs/cdirs/des/y6-redshift/imsim_400Tile/fidbin_S005/"
redshift_catalogs = {
    shear_step: os.path.join(
        redshift_base,
        f"{shear_step}_sompz_unblind_fidbin.h5"
    )
    for shear_step in SHEAR_STEPS
}


imsim_constant_pair = (
    imsim_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
    imsim_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
)
redshift_constant_pair = (
    redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
    redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
)


# def accumulate_pair(*, pdict, mdict, tile, edges, mdet_mask):
def accumulate_pair(dset_plus, dset_minus, bhat_plus, bhat_minus, edges, mdet_mask, tomographic_bin=None, tile=None):
    # plus
    in_tile_p = dset_plus["tilename"][:] == tile
    in_tomo_p = bhat_plus["noshear"] == tomographic_bin
    sel_p = in_tile_p & in_tomo_p

    matcher_p = smatch.Matcher(dset_plus["ra"][sel_p], dset_plus["dec"][sel_p])
    indices_p, distances_p = matcher_p.query_knn(matcher_p.lon, matcher_p.lat, k=2, return_distances=True)
    dnn_p = distances_p[:, 1] * 60 * 60
    del matcher_p, indices_p, distances_p

    hist_p, _ = np.histogram(dnn_p, bins=edges)
    del dnn_p

    tile_map_p = util.get_tile_mask(
        tile,
        "r",
        shear="plus",
        mdet_mask=mdet_mask,
    )
    tile_area_p = tile_map_p.get_valid_area(degrees=True)

    n_sample_p = np.sum(sel_p)

    rand_ra_p, rand_dec_p = healsparse.make_uniform_randoms(tile_map_p, n_sample_p)
    del tile_map_p

    rand_matcher_p = smatch.Matcher(rand_ra_p, rand_dec_p)
    rand_indices_p, rand_distances_p = rand_matcher_p.query_knn(rand_matcher_p.lon, rand_matcher_p.lat, k=2, return_distances=True)
    rand_dnn_p = rand_distances_p[:, 1] * 60 * 60
    del rand_matcher_p, rand_indices_p, rand_distances_p

    rand_hist_p, _ = np.histogram(rand_dnn_p, bins=edges)
    del rand_dnn_p

    # minus
    in_tile_m = dset_minus["tilename"][:] == tile
    in_tomo_m = bhat_minus["noshear"] == tomographic_bin
    sel_m = in_tile_m & in_tomo_m

    matcher_m = smatch.Matcher(dset_minus["ra"][sel_m], dset_minus["dec"][sel_m])
    indices_m, distances_m = matcher_m.query_knn(matcher_m.lon, matcher_m.lat, k=2, return_distances=True)
    dnn_m = distances_m[:, 1] * 60 * 60
    del matcher_m, indices_m, distances_m

    hist_m, _ = np.histogram(dnn_m, bins=edges)
    del dnn_m

    tile_map_m = util.get_tile_mask(
        tile,
        "r",
        shear="minus",
        mdet_mask=mdet_mask,
    )
    tile_area_m = tile_map_m.get_valid_area(degrees=True)

    n_sample_m = np.sum(sel_m)

    rand_ra_m, rand_dec_m = healsparse.make_uniform_randoms(tile_map_m, n_sample_m)
    del tile_map_m

    rand_matcher_m = smatch.Matcher(rand_ra_m, rand_dec_m)
    rand_indices_m, rand_distances_m = rand_matcher_m.query_knn(rand_matcher_m.lon, rand_matcher_m.lat, k=2, return_distances=True)
    rand_dnn_m = rand_distances_m[:, 1] * 60 * 60
    del rand_matcher_m, rand_indices_m, rand_distances_m

    rand_hist_m, _ = np.histogram(rand_dnn_m, bins=edges)
    del rand_dnn_m

    return hist_p, rand_hist_p, tile_area_p, hist_m, rand_hist_m, tile_area_m


def main():

    hf = h5py.File(
        "/dvs_ro/cfs/projectdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6/metadetect_cutsv6_all_blinded.h5",
        # "/global/cfs/projectdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6/metadetect_cutsv6_all_blinded.h5",
        mode="r",
        locking=False
    )
    dset = hf["mdet"]["noshear"]
    hf_redshift = h5py.File(
        "/dvs_ro/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/sompz_v6_10000Tile_final_unblind_24-11-05.h5",
        mode="r",
        locking=False,
    )

    bhat = {
        mdet_step: tomography.get_tomography(
            hf,
            hf_redshift,
            mdet_step,
        )
        for mdet_step in MDET_STEPS
    }

    catalog_p, catalog_m = imsim_constant_pair
    redshift_catalog_p, redshift_catalog_m = redshift_constant_pair

    hf_plus = h5py.File(
        catalog_p,
        mode="r",
        locking=False,
    )
    hf_redshift_plus = h5py.File(
        redshift_catalog_p,
        mode="r",
        locking=False,
    )

    hf_minus = h5py.File(
        catalog_m,
        mode="r",
        locking=False,
    )
    hf_redshift_minus = h5py.File(
        redshift_catalog_m,
        mode="r",
        locking=False,
    )

    bhat_plus = {
        mdet_step: tomography.get_tomography(
            hf_plus,
            hf_redshift_plus,
            mdet_step,
        )
        for mdet_step in MDET_STEPS
    }
    bhat_minus = {
        mdet_step: tomography.get_tomography(
            hf_minus,
            hf_redshift_minus,
            mdet_step,
        )
        for mdet_step in MDET_STEPS
    }

    dset_plus = hf_plus["mdet"]["noshear"]
    dset_minus = hf_minus["mdet"]["noshear"]

    tilenames_p = np.unique(hf_plus["mdet"]["noshear"]["tilename"][:])
    tilenames_m = np.unique(hf_minus["mdet"]["noshear"]["tilename"][:])
    tilenames = np.intersect1d(tilenames_p, tilenames_m)
    # tilenames = tilenames[:10]  # FIXME
    ntiles = len(tilenames)

    mdet_mask = util.load_mdet_mask()
    mdet_area = mdet_mask.get_valid_area()

    # edges = np.linspace(0, 1, NBINS + 1)  # arcmin
    # edges = np.geomspace(1e-2, 1, NBINS + 1)  # arcmin
    edges = np.geomspace(1e-2, 1e1, NBINS + 1) * 60  # arcsec

    do_mdet = True

    fig, axs = plt.subplots(1, len(TOMOGRAPHIC_BINS), sharex=True, sharey=True, constrained_layout=True)

    for tomographic_bin in TOMOGRAPHIC_BINS:
        i = tomographic_bin

        hist = np.zeros(NBINS)

        if do_mdet:
            sel = bhat["noshear"] == tomographic_bin
            matcher = smatch.Matcher(dset["ra"][sel], dset["dec"][sel])
            indices, distances = matcher.query_knn(matcher.lon, matcher.lat, k=2, return_distances=True)
            dnn = distances[:, 1] * 60 * 60
            del matcher, indices, distances  # forecfully cleanup
            _hist, _ = np.histogram(dnn, bins=edges)
            hist += _hist

            hist /= mdet_area


        hist_p = np.zeros(NBINS)
        rand_hist_p = np.zeros(NBINS)
        area_p = 0
        hist_m = np.zeros(NBINS)
        rand_hist_m = np.zeros(NBINS)
        area_m = 0

        for res in tqdm.tqdm(
            map(
                functools.partial(accumulate_pair, dset_plus, dset_minus, bhat_plus, bhat_minus, edges, mdet_mask, tomographic_bin),
                tilenames,
            ),
            total=len(tilenames),
            ncols=80,
        ):
            hist_p += res[0]
            rand_hist_p += res[1]
            area_p += res[2]
            hist_m += res[3]
            rand_hist_m += res[4]
            area_m += res[5]

        # hist_sims = np.nanmean([hist_p, hist_m], axis=0)
        # hist_sims /= np.nanmean([area_p, area_m])
        hist_sims = np.nanmean([hist_p / area_p, hist_m / area_m], axis=0)
        hist_rand = np.nanmean([rand_hist_p / area_p, rand_hist_m / area_m], axis=0)


        if do_mdet:
            axs[i].stairs(
                hist,
                edges=edges,
                # color=plotting.mdet_color,
                color="black",
                label="mdet",
            )
        axs[i].stairs(
            hist_rand,
            edges=edges,
            color="gray",
            label="rand",
        )
        axs[i].stairs(
            hist_sims,
            edges=edges,
            # color=plotting.sims_color,
            color=tomo_colors[tomographic_bin],
            label="sims",
        )
        axs[i].legend(loc="upper left")
        # axs[i].grid()

    axs[0].set_xlabel("nearest neighbor distance [arcsec]")
    axs[0].set_ylabel("source density [$counts / deg^2$]")
    axs[0].set_xscale("log")
    axs[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # axs[0].xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    # axs[0].set_yscale("log")
    axs[0].set_ylim(0, None)

    fig.savefig("clustering-tomographic.pdf")


if __name__ == "__main__":
    main()
