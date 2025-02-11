import h5py
import numpy as np
import matplotlib.pyplot as plt

import lib


onecolumn_kwargs = {
    "width": 2,
    "height": 2,
    "horizontal_margin": 8/12,
    "vertical_margin": 6/12,
    "gutter": 1,
    "fig_width": 3 + 4/12,
    "fig_height": 3,
}

twocolumn_kwargs = {
    "width": 6,
    "height": 2,
    "margin": 1,
    "gutter": 1,
    "fig_width": 8,
    "fig_height": 4,
}



def main():
    lib.plotting.setup()

    shear_step = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    assert shear_step in lib.const.SHEAR_STEPS
    print(shear_step)

    truth_match_table = {}

    with h5py.File(
        lib.const.TRUTH_MATCH_CATALOG,
        mode="r",
    ) as hf:
        for k, v in hf.items():
            truth_match_table[k] = v[:]

    wide_table = {}

    with h5py.File(
        lib.const.IMSIM_CATALOGS[shear_step],
        mode="r",
    ) as hf:
        for k, v in hf["mdet"]["noshear"].items():
            wide_table[k] = v[:]

    match_table = {}

    with h5py.File(
        lib.const.MATCH_CATALOGS[shear_step],
        mode="r",
    ) as hf:
        for k, v in hf["mdet"]["noshear"].items():
            match_table[k] = v[:]

    _, wide_index, match_index = np.intersect1d(
        wide_table["uid"][:],
        match_table["uid"][:],
        return_indices=True,
    )

    truth_kwargs = {
        # "ec": "k",
        "color": "k",
        "ls": ":",
        "label": "truth",
    }
    match_kwargs = {
        # "ec": "b",
        "color": "b",
        "ls": "--",
        "label": "wide–truth",
    }
    wide_kwargs = {
        # "ec": "b",
        "color": "b",
        "ls": "-",
        "label": "wide",
    }

    NBINS = 100
    bins = np.linspace(0, 1_000, NBINS + 1)

    fig, axs = lib.plotting.make_axes(
        1, 1,
        **onecolumn_kwargs,
    )

    # axs.hist(
    #     truth_match_table["DEEP:flux_r"],
    #     bins=bins,
    #     histtype="step",
    #     density=True,
    #     **truth_kwargs,
    # )
    # axs.hist(
    #     match_table["DEEP:flux_r"],
    #     bins=bins,
    #     histtype="step",
    #     density=True,
    #     **match_kwargs,
    # )
    truth_match_hist, _ = np.histogram(
        truth_match_table["DEEP:flux_r"],
        bins=bins,
        density=True
    )
    match_hist, _ = np.histogram(
        match_table["DEEP:flux_r"],
        bins=bins,
        density=True
    )
    lib.plotting.contour1d(
        axs,
        truth_match_hist,
        bins,
        **truth_kwargs,
    )
    lib.plotting.contour1d(
        axs,
        match_hist,
        bins,
        **match_kwargs,
    )
    axs.set_yscale("log")
    axs.set_xlabel("$r$ [deep flux]")
    axs.legend(loc="lower right")

    lib.plotting.watermark(fig)

    fig.savefig("flux_recovery.pdf")

    # --

    bins = np.linspace(0, 1_000, NBINS + 1)

    fig, axs = lib.plotting.make_axes(
        1, 1,
        **onecolumn_kwargs,
    )

    # axs.hist(
    #     match_table["WIDE:pgauss_flux_r"],
    #     bins=bins,
    #     histtype="step",
    #     **match_kwargs,
    # )
    # axs.hist(
    #     wide_table["pgauss_band_flux_r"],
    #     bins=bins,
    #     histtype="step",
    #     **wide_kwargs,
    # )
    match_hist, _ = np.histogram(
        match_table["WIDE:pgauss_flux_r"],
        bins=bins,
    )
    wide_hist, _ = np.histogram(
        wide_table["pgauss_band_flux_r"],
        bins=bins,
    )
    lib.plotting.contour1d(
        axs,
        match_hist,
        bins,
        **match_kwargs,
    )
    lib.plotting.contour1d(
        axs,
        wide_hist,
        bins,
        **wide_kwargs,
    )
    axs.set_yscale("log")
    axs.set_xlabel("$r$ [pgauss flux]")
    axs.legend(loc="lower right")

    lib.plotting.watermark(fig)

    fig.savefig("flux_match.pdf")


if __name__ == "__main__":
    main()
