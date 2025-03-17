import os

import h5py
import numpy as np

import lib


def main():

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step], "r") as sompz_sim,
            h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step], "a") as tomography_sim,
        ):

            _sompz_group = tomography_sim.get("mdet")
            if _sompz_group is None:
                _sompz_group = tomography_sim.create_group("sompz")

            for mdet_step in lib.const.MDET_STEPS:


                _shear_group = _sompz_group.get(mdet_step)
                if _shear_group is None:
                    _shear_group = _sompz_group.create_group(mdet_step)

                COLUMN = "bhat"
                _dataset = _shear_group.get(COLUMN)
                if _dataset is None:
                    _n = shear_sim["mdet"][mdet_step]["uid"].len()
                    _data = np.full(_n, np.nan)
                    _shear_group.create_dataset(COLUMN, data=_data)
                    _dataset = _shear_group[COLUMN]

                    del _data

                _bhat = lib.tomography.get_tomography(shear_sim, sompz_sim, mdet_step)
                _dataset[:] = _bhat

                COLUMN = "cell_wide"
                _dataset = _shear_group.get(COLUMN)
                if _dataset is None:
                    _n = shear_sim["mdet"][mdet_step]["uid"].len()
                    _data = np.full(_n, np.nan)
                    _shear_group.create_dataset(COLUMN, data=_data)
                    _dataset = _shear_group[COLUMN]

                    del _data

                _cell = lib.tomography.get_assignment(shear_sim, sompz_sim, mdet_step)
                _dataset[:] = _cell



    return 0


if __name__ == "__main__":
    main()
