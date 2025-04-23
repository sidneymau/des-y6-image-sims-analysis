import os
import pickle

import h5py
import numpy as np
from scipy import stats

import lib

COLUMN = "nz_weight"

def main():

    weights_file = f"nz-match_weight.pickle"
    with open(weights_file, "rb") as handle:
        nz_weight = pickle.load(handle)

    
    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step], "r") as redshift_sim,
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

                _sel = np.isfinite(cell_sim)

                _dataset[:] = np.nan
                _dataset[_sel] = nz_weight[cell_sim[_sel].astype(int)]


    return 0


if __name__ == "__main__":
    main()
