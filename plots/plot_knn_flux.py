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


    X_train = np.array(
        [
            truth_table[f"flux_{band}"][truth_indices]
            for band in lib.const.TRUTH_BANDS
        ]
    ).T

    y_train = knn.predict(X_train)

    X_test = np.array(
        [
            truth_table[f"flux_{band}"][truth_diff_indices]
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

    fig, axs = lib.plotting.make_axes(
        1, 3,
        sharex="row",
        sharey="row",
        width=1.5,
        height=1.5,
        horizontal_margin=8/12,
        vertical_margin=6/12,
        gutter=7/12,
        fig_width=7,
        fig_height=2.5,
    )

    axs[0].hist2d(
        _flux,
        _flux_err,
        bins=bins,
    )
    axs[0].set_title("Deep Field")

    axs[1].hist2d(
        X_train[:, 2],
        y_train[:, 2],
        bins=bins,
    )
    axs[1].tick_params("y", labelleft=False)
    axs[1].set_title("KNN (training)")

    axs[2].hist2d(
        X_test[:, 2],
        y_test[:, 2],
        bins=bins,
    )
    axs[2].tick_params("y", labelleft=False)
    axs[2].set_title("KNN (validation)")

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    fig.supxlabel("$r$ [flux]")
    fig.supylabel("$\\sigma_r$ [flux]")


    lib.plotting.watermark(fig)

    fig.savefig("flux_knn.pdf")


if __name__ == "__main__":
    main()
