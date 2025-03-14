import os
import pickle

import h5py
import numpy as np

import lib

COLUMN = "neighbor_weight"

def main():

    weights = {}

    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        weights_filename = f"kmeans_weights_{tomographic_bin}.pickle"
        with open(weights_filename, "rb") as handle:
            weights[tomographic_bin] = pickle.load(handle)

    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step], "r") as redshift_sim,
            h5py.File(f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_{shear_step}.hdf5", "r") as neighbors_sim,
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
                
                bhat_sim = lib.tomography.get_tomography(shear_sim, redshift_sim, mdet_step)
    
                for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
    
                    _weights = weights[tomographic_bin]
                    _labels = _weights["labels"]
                    _w = _weights["weights"]
                    _pipeline = _weights["pipeline"]
    
                    sel_sim = (bhat_sim == tomographic_bin)
    
                    X_sim = np.stack(
                        [
                            np.log10(neighbors_sim["mdet"][mdet_step]["neighbor_mag"][sel_sim]),
                            neighbors_sim["mdet"][mdet_step]["mag"][sel_sim],
                            neighbors_sim["mdet"][mdet_step]["neighbor_distance"][sel_sim],
                        ],
                        axis=-1,
                    )
                    y_sim = _pipeline.predict(X_sim)
    
                    _ind = np.digitize(
                        y_sim,
                        _labels,
                        right=True,
                    )
    
                    _dataset[sel_sim] = _w[_ind]



    return 0


if __name__ == "__main__":
    main()
