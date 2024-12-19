from pathlib import Path
import functools
import os

import tqdm
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import weights
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

imsim_base = "/global/cfs/cdirs/des/y6-image-sims/fiducial-400/"
imsim_catalogs = {
    shear_step: os.path.join(
        imsim_base,
        shear_step,
        "metadetect_cutsv6_all.h5",
    )
    for shear_step in SHEAR_STEPS
}

imsim_constant_pair = (
    imsim_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
    imsim_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
)


redshift_base = "/global/cfs/cdirs/des/y6-redshift/imsim_400Tile/fidbin_S005/"
redshift_catalogs = {
    shear_step: os.path.join(
        redshift_base,
        f"{shear_step}_sompz_unblind_fidbin.h5"
    )
    for shear_step in SHEAR_STEPS
}

redshift_constant_pair = (
    redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
    redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
)


def compute_count_grid(fname, bins):
    print(f"Computing count grid for {fname}")
    with h5py.File(fname, mode="r", locking=False) as hf:

        size_ratio = hf["mdet"]["noshear"]["gauss_T_ratio"][:]
        snr = hf["mdet"]["noshear"]["gauss_s2n"][:]
        count, _, _, _ = stats.binned_statistic_2d(
            size_ratio,
            snr,
            None,
            statistic="count",
            bins=bins,
        )

    return count


def compute_count_grid_tomographic(fname, fname_redshift, bins, tomographic_bin):
    print(f"Computing count grid for {fname}")
    with h5py.File(fname, mode="r", locking=False) as hf, h5py.File(fname_redshift, mode="r", locking=False) as hf_redshift:
        bhat = tomography.get_tomography(
            hf,
            hf_redshift,
            "noshear",
        )
        sel = bhat == tomographic_bin

        size_ratio = hf["mdet"]["noshear"]["gauss_T_ratio"][sel]
        snr = hf["mdet"]["noshear"]["gauss_s2n"][sel]
        count, _, _, _ = stats.binned_statistic_2d(
            size_ratio,
            snr,
            None,
            statistic="count",
            bins=bins,
        )

    return count

def compute_response_grid(fname, bins):
    print(f"Computing response grid for {fname}")
    with h5py.File(fname, mode="r", locking=False) as hf:

        size_ratio_1p = hf["mdet"]["1p"]["gauss_T_ratio"][:]
        snr_1p = hf["mdet"]["1p"]["gauss_s2n"][:]
        values_1p = hf["mdet"]["1p"]["gauss_g_1"][:]
        e1_1p, _, _, _ = stats.binned_statistic_2d(
            size_ratio_1p,
            snr_1p,
            values_1p,
            statistic="mean",
            # statistic="sum",
            bins=bins,
        )
        # count_1p, _, _, _ = stats.binned_statistic_2d(
        #     size_ratio_1p,
        #     snr_1p,
        #     None,
        #     statistic="count",
        #     bins=bins,
        # )

        size_ratio_1m = hf["mdet"]["1m"]["gauss_T_ratio"][:]
        snr_1m = hf["mdet"]["1m"]["gauss_s2n"][:]
        values_1m = hf["mdet"]["1m"]["gauss_g_1"][:]
        e1_1m, _, _, _ = stats.binned_statistic_2d(
            size_ratio_1m,
            snr_1m,
            values_1m,
            statistic="mean",
            # statistic="sum",
            bins=bins,
        )
        # count_1m, _, _, _ = stats.binned_statistic_2d(
        #     size_ratio_1m,
        #     snr_1m,
        #     None,
        #     statistic="count",
        #     bins=bins,
        # )

        # e1_1p = e1_1p / count_1p
        # e1_1m = e1_1m / count_1m
        responsivity = (e1_1p - e1_1m) / (2 * 0.01)

    return responsivity


def compute_response_grid_tomographic(fname, fname_redshift, bins, tomographic_bin):
    print(f"Computing response grid for {fname}")
    with h5py.File(fname, mode="r", locking=False) as hf, h5py.File(fname_redshift, mode="r", locking=False) as hf_redshift:
        bhat_1p = tomography.get_tomography(
            hf,
            hf_redshift,
            "1p",
        )
        sel_1p = bhat_1p == tomographic_bin

        size_ratio_1p = hf["mdet"]["1p"]["gauss_T_ratio"][sel_1p]
        snr_1p = hf["mdet"]["1p"]["gauss_s2n"][sel_1p]
        values_1p = hf["mdet"]["1p"]["gauss_g_1"][sel_1p]
        e1_1p, _, _, _ = stats.binned_statistic_2d(
            size_ratio_1p,
            snr_1p,
            values_1p,
            statistic="mean",
            # statistic="sum",
            bins=bins,
        )
        # count_1p, _, _, _ = stats.binned_statistic_2d(
        #     size_ratio_1p,
        #     snr_1p,
        #     None,
        #     statistic="count",
        #     bins=bins,
        # )

        bhat_1m = tomography.get_tomography(
            hf,
            hf_redshift,
            "1m",
        )
        sel_1m = bhat_1m == tomographic_bin

        size_ratio_1m = hf["mdet"]["1m"]["gauss_T_ratio"][sel_1m]
        snr_1m = hf["mdet"]["1m"]["gauss_s2n"][sel_1m]
        values_1m = hf["mdet"]["1m"]["gauss_g_1"][sel_1m]
        e1_1m, _, _, _ = stats.binned_statistic_2d(
            size_ratio_1m,
            snr_1m,
            values_1m,
            statistic="mean",
            # statistic="sum",
            bins=bins,
        )
        # count_1m, _, _, _ = stats.binned_statistic_2d(
        #     size_ratio_1m,
        #     snr_1m,
        #     None,
        #     statistic="count",
        #     bins=bins,
        # )

        # e1_1p = e1_1p / count_1p
        # e1_1m = e1_1m / count_1m
        responsivity = (e1_1p - e1_1m) / (2 * 0.01)

    return responsivity



def main():

    NBINS = 10  # 20
    bins = (
        np.geomspace(0.5, 5, NBINS +  1),
        np.geomspace(10, 1000, NBINS +  1)
    )

    mdet_fname = "/dvs_ro/cfs/projectdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6/metadetect_cutsv6_all_blinded.h5"
    redshift_fname = "/dvs_ro/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/sompz_v6_10000Tile_final_unblind_24-11-05.h5"

    count = compute_count_grid(mdet_fname, bins)
    count_p = compute_count_grid(imsim_constant_pair[0], bins)
    count_m = compute_count_grid(imsim_constant_pair[1], bins)
    count_sims = np.nanmean([count_p, count_m], axis=0)

    responsivity = compute_response_grid(mdet_fname, bins)
    responsivity_p = compute_response_grid(imsim_constant_pair[0], bins)
    responsivity_m = compute_response_grid(imsim_constant_pair[1], bins)
    responsivity_sims = np.nanmean([responsivity_p, responsivity_m], axis=0)

    responsivity_label = "$\\langle R \\rangle$"
    responsivity_norm = mpl.colors.Normalize()

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout=True)

    # count

    pcm = axs[0, 0].pcolormesh(
        bins[0],
        bins[1],
        count.T,
        alpha=0.5,
    )
    fig.colorbar(pcm)

    pcm = axs[0, 1].pcolormesh(
        bins[0],
        bins[1],
        count_sims.T,
        alpha=0.5,
    )
    fig.colorbar(pcm)

    # pcm = axs[0, 2].pcolormesh(
    #     bins[0],
    #     bins[1],
    #     count_sims.T - count.T,
    #     norm=mpl.colors.CenteredNorm(),
    #     cmap="RdBu_r",
    #     alpha=0.5,
    # )
    # fig.colorbar(pcm, label=count_label)

    # responsivity

    pcm = axs[1, 0].pcolormesh(
        bins[0],
        bins[1],
        responsivity.T,
        norm=responsivity_norm,
        alpha=0.5,
    )
    # fig.colorbar(pcm, cax=axs[0], label=responsivity_label)

    pcm = axs[1, 1].pcolormesh(
        bins[0],
        bins[1],
        responsivity_sims.T,
        norm=responsivity_norm,
        alpha=0.5,
    )
    fig.colorbar(pcm, label=responsivity_label)

    pcm = axs[1, 2].pcolormesh(
        bins[0],
        bins[1],
        responsivity_sims.T - responsivity.T,
        norm=mpl.colors.CenteredNorm(),
        cmap="RdBu_r",
        alpha=0.5,
    )
    fig.colorbar(pcm, label="sims - mdet")

    axs[1, 0].set_xlabel("gauss_T_ratio")
    axs[1, 1].set_xlabel("gauss_T_ratio")
    axs[1, 2].set_xlabel("gauss_T_ratio")
    axs[0, 0].set_ylabel("gauss_s2n")
    axs[1, 0].set_ylabel("gauss_s2n")

    axs[0, 0].set_title("mdet")
    axs[0, 1].set_title("sims")
    # axs[0, 2].set_title("sims - mdet")

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")

    axs[0, 0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axs[0, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[0, 0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto"))


    fig.savefig("responsivity.pdf")

    print("mdet", np.average(responsivity, weights=count))
    print("sims", np.average(responsivity_sims, weights=count_sims))
    # print("diff", np.sum(responsivity_sims * count_sims - responsivity * count) / np.sum(count_sims + count))


    responsivity_norm = mpl.colors.Normalize()

    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, constrained_layout=True)
    for tomographic_bin in TOMOGRAPHIC_BINS:

        responsivity = compute_response_grid_tomographic(mdet_fname, redshift_fname, bins, tomographic_bin)
        responsivity_p = compute_response_grid_tomographic(imsim_constant_pair[0], redshift_constant_pair[0], bins, tomographic_bin)
        responsivity_m = compute_response_grid_tomographic(imsim_constant_pair[1], redshift_constant_pair[1], bins, tomographic_bin)
        responsivity_sims = np.nanmean([responsivity_p, responsivity_m], axis=0)

        # responsivity

        pcm = axs[0, tomographic_bin].pcolormesh(
            bins[0],
            bins[1],
            responsivity.T,
            norm=responsivity_norm,
            alpha=0.5,
        )
        # fig.colorbar(pcm, cax=axs[0], label=responsivity_label)

        pcm = axs[1, tomographic_bin].pcolormesh(
            bins[0],
            bins[1],
            responsivity_sims.T,
            norm=responsivity_norm,
            alpha=0.5,
        )

    fig.colorbar(pcm, label=responsivity_label)

    axs[1, 0].set_xlabel("gauss_T_ratio")
    axs[1, 1].set_xlabel("gauss_T_ratio")
    axs[1, 2].set_xlabel("gauss_T_ratio")
    axs[1, 3].set_xlabel("gauss_T_ratio")
    axs[0, 0].set_ylabel("gauss_s2n")
    axs[1, 0].set_ylabel("gauss_s2n")

    axs[0, 0].set_title("0")
    axs[0, 1].set_title("1")
    axs[0, 2].set_title("2")
    axs[0, 3].set_title("3")
    # axs[0, 2].set_title("sims - mdet")

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")

    axs[0, 0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    axs[0, 0].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[0, 0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins="auto"))


    fig.savefig("responsivity-tomographic.pdf")


if __name__ == "__main__":
    main()
