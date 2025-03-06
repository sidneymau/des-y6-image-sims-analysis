import os

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

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

    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    assert shear_step_plus in lib.const.SHEAR_STEPS
    assert shear_step_minus in lib.const.SHEAR_STEPS

    with (
        h5py.File(lib.const.Y6_SHEAR_CATALOG) as shear_y6,
        h5py.File(lib.const.Y6_REDSHIFT_CATALOG) as redshift_y6,
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus]) as shear_sim,
        h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step_plus]) as redshift_sim,
    ):

        cell_y6 = lib.tomography.get_assignment(shear_y6, redshift_y6, "noshear")

        count_grid_y6, _, _ = stats.binned_statistic(
            cell_y6,
            None,
            statistic="count",
            bins=lib.const.CELL_IDS,
        )
        occupancy_grid_y6 = count_grid_y6 / sum(count_grid_y6)
            
        cell_sim = lib.tomography.get_assignment(shear_sim, redshift_sim, "noshear")
        # weights_sim = lib.weight.get_shear_weights(shear_sim["mdet/noshear"])
        
        count_grid_sim, _, _ = stats.binned_statistic(
            cell_sim,
            None,
            statistic="count",
            bins=lib.const.CELL_IDS,
        )
        occupancy_grid_sim = count_grid_sim / sum(count_grid_sim)

    fig, axs = lib.plotting.make_axes(
        1, 1,
        **onecolumn_kwargs,
    )


    im = axs.imshow(
        (occupancy_grid_y6 / occupancy_grid_sim).reshape(lib.const.SOM_SHAPE),
        origin="lower",
        vmin=0,
    )
    lib.plotting.add_colorbar(axs, im)

    axs.set_title("Occupancy Ratio")
    
    axs.set_xticks([])
    axs.set_yticks([])

    lib.plotting.watermark(fig)

    fig.savefig("occupancy-ratio.pdf")

    # plt.show()


if __name__ == "__main__":
    main()
