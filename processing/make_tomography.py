import os

import h5py
import numpy as np

import lib

def _main(shear_catalog, redshift_catalog, tomography_catalog):
    with (
        h5py.File(shear_catalog, "r") as shear,
        h5py.File(redshift_catalog, "r") as sompz,
        h5py.File(tomography_catalog, "a") as tomography,
    ):
    
        _sompz_group = tomography.get("mdet")
        if _sompz_group is None:
            _sompz_group = tomography.create_group("sompz")
    
        for mdet_step in lib.const.MDET_STEPS:
    
    
            _shear_group = _sompz_group.get(mdet_step)
            if _shear_group is None:
                _shear_group = _sompz_group.create_group(mdet_step)
    
            COLUMN = "bhat"
            _dataset = _shear_group.get(COLUMN)
            if _dataset is None:
                _n = shear["mdet"][mdet_step]["uid"].len()
                _data = np.full(_n, np.nan)
                _shear_group.create_dataset(COLUMN, data=_data)
                _dataset = _shear_group[COLUMN]
    
                del _data
    
            _bhat = lib.tomography.get_tomography(shear, sompz, mdet_step)
            _dataset[:] = _bhat
    
            COLUMN = "cell_wide"
            _dataset = _shear_group.get(COLUMN)
            if _dataset is None:
                _n = shear["mdet"][mdet_step]["uid"].len()
                _data = np.full(_n, np.nan)
                _shear_group.create_dataset(COLUMN, data=_data)
                _dataset = _shear_group[COLUMN]
    
                del _data
    
            _cell = lib.tomography.get_assignment(shear, sompz, mdet_step)
            _dataset[:] = _cell


def main():

    print("Y6")
    _main(
        lib.const.Y6_SHEAR_CATALOG,
        lib.const.Y6_REDSHIFT_CATALOG,
        lib.const.Y6_TOMOGRAPHY_CATALOG,
    )

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)
        _main(
            lib.const.SIM_SHEAR_CATALOGS[shear_step],
            lib.const.SIM_REDSHIFT_CATALOGS[shear_step],
            lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step],
        )

    return 0


if __name__ == "__main__":
    main()
