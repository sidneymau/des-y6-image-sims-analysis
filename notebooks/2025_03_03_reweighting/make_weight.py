import pickle

import h5py
import numpy as np

import lib


N_CLUSTERS = 200

def main():

    rng = np.random.default_rng(0)

    w_bins = np.arange(N_CLUSTERS)
    w_vals_y6 = {}
    w_vals_sim = {}
    weights = {}

    out = {}

    scalers = {}
    kmeans = {}

    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:

        scaler_file = f"scaler_{tomographic_bin}.pickle"
        with open(scaler_file, "rb") as handle:
            scalers[tomographic_bin] = pickle.load(handle)

        kmeans_file = f"kmeans_{tomographic_bin}.pickle"
        with open(kmeans_file, "rb") as handle:
            kmeans[tomographic_bin] = pickle.load(handle)

    # first, y6
    print("Y6")
    with (
        h5py.File(lib.const.Y6_SHEAR_CATALOG) as shear_y6,
        h5py.File(lib.const.Y6_REDSHIFT_CATALOG) as redshift_y6,
        h5py.File(
            f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_y6.hdf5",
        ) as neighbors_y6,
    ):
        bhat_y6 = lib.tomography.get_tomography(shear_y6, redshift_y6, "noshear")

        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            _kmeans = kmeans[tomographic_bin]
            _scaler = scalers[tomographic_bin]
            
            sel_y6 = (bhat_y6 == tomographic_bin)

            X_y6 = np.stack(
                [
                    np.log10(neighbors_y6["mdet"]["noshear"]["neighbor_mag"][sel_y6]),
                    neighbors_y6["mdet"]["noshear"]["mag"][sel_y6],
                    neighbors_y6["mdet"]["noshear"]["neighbor_distance"][sel_y6],
                ],
                axis=-1,
            )

            y_y6 = _kmeans.predict(
                _scaler.transform(X_y6)
            )

            _w = np.bincount(y_y6)
            w_vals_y6[tomographic_bin] = _w / np.mean(_w)


    # next, sims
    for shear_step in lib.const.SHEAR_STEPS:
        print(shear_step)

        weights[shear_step] = {}
        w_vals_sim[shear_step] = {}

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step]) as shear_sim,
            h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_step]) as redshift_sim,
            h5py.File(
                f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_{shear_step}.hdf5",
            ) as neighbors_sim,
        ):
            out[shear_step] = np.full(shear_sim["mdet/noshear"]["uid"].shape, np.nan)
            bhat_sim = lib.tomography.get_tomography(shear_sim, redshift_sim, "noshear")

            for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:

                _scaler = scalers[tomographic_bin]
                _kmeans = kmeans[tomographic_bin]

                sel_sim = (bhat_sim == tomographic_bin)

                X_sim = np.stack(
                    [
                        np.log10(neighbors_sim["mdet"]["noshear"]["neighbor_mag"][sel_sim]),
                        neighbors_sim["mdet"]["noshear"]["mag"][sel_sim],
                        neighbors_sim["mdet"]["noshear"]["neighbor_distance"][sel_sim],
                    ],
                    axis=-1,
                )
                y_sim = _kmeans.predict(_scaler.transform(X_sim))

                _w = np.bincount(y_sim)
                w_vals_sim[shear_step][tomographic_bin] = _w / np.mean(_w)

                # TODO renoramlize weights here
                _weights = w_vals_y6[tomographic_bin] / w_vals_sim[shear_step][tomographic_bin]
                _weights = np.ma.masked_invalid(_weights)


                _ind = np.digitize(
                    y_sim,
                    w_bins,
                    right=True,
                )

                weights[shear_step][tomographic_bin] = _weights[_ind]
                out[shear_step][sel_sim] = _weights[_ind]

    for shear_step in lib.const.SHEAR_STEPS:
        weight_file = f"weight_{shear_step}.pickle"
        with open(weight_file, "wb") as handle:
            pickle.dump(out[shear_step], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == "__main__":
    main()
