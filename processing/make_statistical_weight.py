import os
import pickle

import h5py
import numpy as np

import lib

COLUMN = "statistical_weight"

def main():

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
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

                _w = lib.weight.get_shear_weights(shear_sim["mdet"][mdet_step])
                _dataset[:] = _w



    return 0


if __name__ == "__main__":
    main()
