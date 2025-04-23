import os
import pickle

import h5py
import numpy as np
from scipy import stats

import lib

def main():

    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    weight_response_catalogs = {
        shear_step: f"SOM_weight_response_{shear_step}.hdf5"
        for shear_step in [shear_step_plus, shear_step_minus]
    }

    for shear_step in [shear_step_plus, shear_step_minus]:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step], "r") as weights_sim,
            h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step], "r") as tomo_sim,
            h5py.File(weight_response_catalogs[shear_step], "a") as out,
        ):

            _mdet_group = out.get("mdet")
            if _mdet_group is None:
                _mdet_group = out.create_group("mdet")

            for mdet_step in lib.const.MDET_STEPS:

                _shear_group = _mdet_group.get(mdet_step)
                if _shear_group is None:
                    _shear_group = _mdet_group.create_group(mdet_step)

                _n = 1024

                _dataset = _shear_group.get("cell_wide")
                if _dataset is None:
                    _shear_group.create_dataset("cell_wide", data=list(range(_n)))
                
                _dataset = _shear_group.get("weight_response")
                if _dataset is None:
                    _data = np.full(_n, np.nan)
                    _shear_group.create_dataset("weight_response", data=_data)
                    _dataset = _shear_group["weight_response"]

                    del _data

                _w = weights_sim["mdet"][mdet_step]["statistical_weight"][:]
                _R = lib.response.get_shear_response(shear_sim["mdet"][mdet_step])
                _cell = tomo_sim["sompz"][mdet_step]["cell_wide"][:]

                _sum, _, _ = stats.binned_statistic(_cell, _w * _R, statistic="sum", bins=lib.const.CELL_IDS)
                _dataset[:] = _sum




    return 0


if __name__ == "__main__":
    main()
