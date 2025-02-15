import random

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


def main():
    lib.plotting.setup()

    shear_step = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    assert shear_step in lib.const.SHEAR_STEPS
    print(shear_step)

    truth_ids = lib.truth.get_truth_ids()
    deepfield_ids = lib.deepfield.get_deepfield_ids()
    _, truth_indices, deepfield_indices = np.intersect1d(
        truth_ids,
        deepfield_ids,
        return_indices=True,
    )
    id_diff = np.setdiff1d(
        truth_ids,
        deepfield_ids,
    )
    _, truth_diff_indices, _  = np.intersect1d(
        truth_ids,
        id_diff,
        return_indices=True,
    )

    knn = lib.deepfield.get_knn()
    deepfield_table = lib.deepfield.get_deepfield_table()
    truth_table = lib.truth.get_truth_table()

    _mag = deepfield_table["MAG_r"]
    _mag_err = deepfield_table["ERR_MAG_r"]

    _flux, _flux_err = lib.util.mag_to_flux_with_error(_mag, _mag_err)

    _med_flux_err = np.median(_flux_err)

    # n_train = len(truth_ids) // 2
    # train_indices = random.choices(range(len(truth_ids)), k=n_train)
    # test_indices = np.setdiff1d(range(len(truth_ids)), train_indices)

    X_train = np.array(
        [
            truth_table[f"flux_{band}"][truth_indices]
            # truth_table[f"flux_{band}"][:n_train]
            # truth_table[f"flux_{band}"][train_indices]
            for band in lib.const.TRUTH_BANDS
        ]
    ).T

    y_train = knn.predict(X_train)

    X_test = np.array(
        [
            truth_table[f"flux_{band}"][truth_diff_indices]
            # truth_table[f"flux_{band}"][n_train:]
            # truth_table[f"flux_{band}"][test_indices]
            for band in lib.const.TRUTH_BANDS
        ]
    ).T

    y_test = knn.predict(X_test)

    NBINS = 100
    bins = [
        np.geomspace(
            # _flux.min(),
            # _flux.max(),
            np.quantile(_flux, 0.01),
            np.quantile(_flux, 0.99),
            NBINS + 1,
        ),
        np.geomspace(
            # _flux_err.min(),
            # _flux_err.max(),
            np.quantile(_flux_err, 0.01),
            np.quantile(_flux_err, 0.99),
            NBINS + 1,
        ),
    ]

    deep_hist, _, _ = np.histogram2d(_flux, _flux_err, bins=bins)
    train_hist, _, _ = np.histogram2d(X_train[:, 2], y_train[:, 2], bins=bins)
    test_hist, _, _ = np.histogram2d(X_test[:, 2], y_test[:, 2], bins=bins)

    percentiles = 1.0 - np.exp(-0.5 * np.array([1.5, 2.0, 2.5, 3.0]) ** 2)
    levels = lib.util.get_levels(deep_hist, percentiles=percentiles)
    train_levels = lib.util.get_levels(train_hist, percentiles=percentiles)
    test_levels = lib.util.get_levels(test_hist, percentiles=percentiles)

    fig, axs = lib.plotting.make_axes(
        1, 1,
        **onecolumn_kwargs,
        # 1, 3,
        # sharex="row",
        # sharey="row",
        # width=1.5,
        # height=1.5,
        # horizontal_margin=8/12,
        # vertical_margin=6/12,
        # gutter=7/12,
        # fig_width=7,
        # fig_height=2.5,
    )

    artists = []
    labels = []

    contours = lib.plotting.contour(
        axs,
        deep_hist,
        bins,
        levels=levels,
        linestyles=":",
        colors="gray",
    )
    _artists, _labels = contours.legend_elements()
    artists.append(_artists[0])
    labels.append("Deep Field")

    contours = lib.plotting.contour(
        axs,
        train_hist,
        bins,
        levels=train_levels,
        linestyles="--",
        colors="k",
    )
    _artists, _labels = contours.legend_elements()
    artists.append(_artists[0])
    labels.append("KNN (inform)")

    contours = lib.plotting.contour(
        axs,
        test_hist,
        bins,
        levels=test_levels,
        linestyles="-",
        colors="r",
    )
    _artists, _labels = contours.legend_elements()
    artists.append(_artists[0])
    labels.append("KNN (predict)")

    axs.set_xscale("log")
    axs.set_yscale("log")

    axs.set_xlabel("$r$ [flux]")
    axs.set_ylabel("$\\sigma_r$ [flux]")

    axs.legend(artists, labels, loc="upper left")

    lib.plotting.watermark(fig)

    fig.savefig("flux_knn.pdf")

    # fig, axs = lib.plotting.make_axes(
    #     1, 3,
    #     sharex="row",
    #     sharey="row",
    #     width=1.5,
    #     height=1.5,
    #     horizontal_margin=8/12,
    #     vertical_margin=6/12,
    #     gutter=7/12,
    #     fig_width=7,
    #     fig_height=2.5,
    # )

    # axs[0].hist2d(
    #     _flux,
    #     _flux_err,
    #     bins=bins,
    # )
    # axs[0].set_title("Deep Field")

    # axs[1].hist2d(
    #     X_train[:, 2],
    #     y_train[:, 2],
    #     bins=bins,
    # )
    # axs[1].tick_params("y", labelleft=False)
    # axs[1].set_title("KNN (training)")

    # axs[2].hist2d(
    #     X_test[:, 2],
    #     y_test[:, 2],
    #     bins=bins,
    # )
    # axs[2].tick_params("y", labelleft=False)
    # axs[2].set_title("KNN (validation)")

    # axs[0].set_xscale("log")
    # axs[0].set_yscale("log")

    # fig.supxlabel("$r$ [flux]")
    # fig.supylabel("$\\sigma_r$ [flux]")


    # lib.plotting.watermark(fig)

    # fig.savefig("flux_knn.pdf")


if __name__ == "__main__":
    main()
