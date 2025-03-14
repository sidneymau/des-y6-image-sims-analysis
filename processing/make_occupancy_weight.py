import os
import pickle

import h5py
import numpy as np
from scipy import stats

import lib

COLUMN = "occupancy_weight"

def main():

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step], "r") as redshift_sim,
            h5py.File(lib.const.Y6_SHEAR_CATALOG, "r") as shear_y6,
            h5py.File(lib.const.Y6_REDSHIFT_CATALOG, "r") as redshift_y6,
            h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step], "a") as weights_sim,
        ):

            _mdet_group = weights_sim.get("mdet")
            if _mdet_group is None:
                _mdet_group = weights_sim.create_group("mdet")

            for mdet_step in lib.const.MDET_STEPS:


                _shear_group = _mdet_group.get(mdet_step)
                if _shear_group is None:
                    _shear_group = _mdet_group.create_group(mdet_step)
                
                _dataset = _shear_group.get(COLUMN)
                if _dataset is None:
                    _n = shear_sim["mdet"][mdet_step]["uid"].len()
                    _data = np.full(_n, np.nan)
                    _shear_group.create_dataset(COLUMN, data=_data)
                    _dataset = _shear_group[COLUMN]

                    del _data
                
                cell_sim = lib.tomography.get_assignment(shear_sim, redshift_sim, mdet_step)
                weight_sim = lib.weight.get_shear_weights(shear_sim["mdet"][mdet_step])
                
                cell_y6 = lib.tomography.get_assignment(shear_y6, redshift_y6, mdet_step)
                weight_y6 = lib.weight.get_shear_weights(shear_y6["mdet"][mdet_step])

                _som_weight_sim, _, _ = stats.binned_statistic(
                    cell_sim,
                    weight_sim,
                    statistic="sum",
                    bins=lib.const.CELL_IDS,
                )
                som_occupancy_sim = _som_weight_sim / sum(_som_weight_sim)

                _som_weight_y6, _, _ = stats.binned_statistic(
                    cell_y6,
                    weight_y6,
                    statistic="sum",
                    bins=lib.const.CELL_IDS,
                )
                som_occupancy_y6 = _som_weight_y6 / sum(_som_weight_y6)

                occupancy_ratio = som_occupancy_y6 / som_occupancy_sim

                _sel = np.isfinite(cell_sim)

                _dataset[_sel] = occupancy_ratio[cell_sim[_sel].astype(int)]


    return 0


if __name__ == "__main__":
    main()
