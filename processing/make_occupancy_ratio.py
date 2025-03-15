import os
import pickle

import h5py
import numpy as np
from scipy import stats

import lib

def main():

    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    with (
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus], "r") as shear_sim_plus,
        h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step_plus], "r") as redshift_sim_plus,
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_minus], "r") as shear_sim_minus,
        h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step_minus], "r") as redshift_sim_minus,
        h5py.File(lib.const.Y6_SHEAR_CATALOG, "r") as shear_y6,
        h5py.File(lib.const.Y6_REDSHIFT_CATALOG, "r") as redshift_y6,
    ):
        mdet_step = "noshear"
            
        cell_sim_plus = lib.tomography.get_assignment(shear_sim_plus, redshift_sim_plus, mdet_step)
        weight_sim_plus = lib.weight.get_shear_weights(shear_sim_plus["mdet"][mdet_step])

        cell_sim_minus = lib.tomography.get_assignment(shear_sim_minus, redshift_sim_minus, mdet_step)
        weight_sim_minus = lib.weight.get_shear_weights(shear_sim_minus["mdet"][mdet_step])
        
        cell_y6 = lib.tomography.get_assignment(shear_y6, redshift_y6, mdet_step)
        weight_y6 = lib.weight.get_shear_weights(shear_y6["mdet"][mdet_step])

        _som_weight_sim_plus, _, _ = stats.binned_statistic(
            cell_sim_plus,
            weight_sim_plus,
            statistic="sum",
            bins=lib.const.CELL_IDS,
        )
        som_occupancy_sim_plus = _som_weight_sim_plus / sum(_som_weight_sim_plus)

        _som_weight_sim_minus, _, _ = stats.binned_statistic(
            cell_sim_minus,
            weight_sim_minus,
            statistic="sum",
            bins=lib.const.CELL_IDS,
        )
        som_occupancy_sim_minus = _som_weight_sim_minus / sum(_som_weight_sim_minus)

        som_occupancy_sim = (som_occupancy_sim_plus + som_occupancy_sim_minus) / 2

        _som_weight_y6, _, _ = stats.binned_statistic(
            cell_y6,
            weight_y6,
            statistic="sum",
            bins=lib.const.CELL_IDS,
        )
        som_occupancy_y6 = _som_weight_y6 / sum(_som_weight_y6)

        occupancy_ratio = som_occupancy_y6 / som_occupancy_sim

        weights_file = f"occupancy_ratio.pickle"
        with open(weights_file, "wb") as handle:
            pickle.dump(occupancy_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return 0


if __name__ == "__main__":
    main()
