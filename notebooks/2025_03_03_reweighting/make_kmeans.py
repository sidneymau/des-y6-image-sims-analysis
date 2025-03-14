import pickle

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lib


shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

N_CLUSTERS = 200
N_SUBSAMPLE = 200_000


def main():

    rng = np.random.default_rng(0)

    shear_sim_plus = h5py.File(
        lib.const.SIM_SHEAR_CATALOGS[shear_step_plus],
    )
    redshift_sim_plus = h5py.File(
        lib.const.SIM_REDSHIFT_CATALOGS[shear_step_plus],
    )
    neighbors_sim_plus = h5py.File(
        f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_{shear_step_plus}.hdf5",
    )

    shear_sim_minus = h5py.File(
        lib.const.SIM_SHEAR_CATALOGS[shear_step_minus],
    )
    redshift_sim_minus = h5py.File(
        lib.const.SIM_REDSHIFT_CATALOGS[shear_step_minus],
    )
    neighbors_sim_minus = h5py.File(
        f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_{shear_step_minus}.hdf5",
    )

    shear_y6 = h5py.File(lib.const.Y6_SHEAR_CATALOG)
    redshift_y6 = h5py.File(lib.const.Y6_REDSHIFT_CATALOG)
    neighbors_y6 = h5py.File(f"/pscratch/sd/s/smau/fiducial-neighbors/neighbors_y6.hdf5")

    bhat_sim_plus = lib.tomography.get_tomography(shear_sim_plus, redshift_sim_plus, "noshear")
    weight_sim_plus = lib.weight.get_shear_weights(shear_sim_plus["mdet/noshear"])

    bhat_sim_minus = lib.tomography.get_tomography(shear_sim_minus, redshift_sim_minus, "noshear")
    weight_sim_minus = lib.weight.get_shear_weights(shear_sim_minus["mdet/noshear"])

    bhat_y6 = lib.tomography.get_tomography(shear_y6, redshift_y6, "noshear")
    weight_y6 = lib.weight.get_shear_weights(shear_y6["mdet/noshear"])

    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        sel_sim_plus = (bhat_sim_plus == tomographic_bin)
        sel_sim_minus = (bhat_sim_minus == tomographic_bin)
        sel_y6 = (bhat_y6 == tomographic_bin)

        _sim_subsample_plus = rng.choice(sum(sel_sim_plus), N_SUBSAMPLE, replace=False)
        _sim_subsample_minus = rng.choice(sum(sel_sim_minus), N_SUBSAMPLE, replace=False)
        _y6_subsample = rng.choice(sum(sel_y6), N_SUBSAMPLE, replace=False)

        sim_subsample_plus = np.isin(
            np.arange(sum(sel_sim_plus)),
            _sim_subsample_plus,
        )
        sim_subsample_minus = np.isin(
            np.arange(sum(sel_sim_minus)),
            _sim_subsample_minus,
        )
        y6_subsample = np.isin(
            np.arange(sum(sel_y6)),
            _y6_subsample,
        )

        X_sim_plus = np.stack(
            [
                np.log10(neighbors_sim_plus["mdet"]["noshear"]["neighbor_mag"][sel_sim_plus][sim_subsample_plus]),
                neighbors_sim_plus["mdet"]["noshear"]["mag"][sel_sim_plus][sim_subsample_plus],
                neighbors_sim_plus["mdet"]["noshear"]["neighbor_distance"][sel_sim_plus][sim_subsample_plus],
            ],
            axis=-1,
        )
        weights_sim_plus = weight_sim_plus[sel_sim_plus][sim_subsample_plus]

        X_sim_minus = np.stack(
            [
                np.log10(neighbors_sim_minus["mdet"]["noshear"]["neighbor_mag"][sel_sim_minus][sim_subsample_minus]),
                neighbors_sim_minus["mdet"]["noshear"]["mag"][sel_sim_minus][sim_subsample_minus],
                neighbors_sim_minus["mdet"]["noshear"]["neighbor_distance"][sel_sim_minus][sim_subsample_minus],
            ],
            axis=-1,
        )
        weights_sim_minus = weight_sim_minus[sel_sim_minus][sim_subsample_minus]

        X_sim = np.concatenate([X_sim_plus, X_sim_minus])
        W_sim = np.concatenate([weights_sim_plus, weights_sim_minus])

        X_y6 = np.stack(
            [
                np.log10(neighbors_y6["mdet"]["noshear"]["neighbor_mag"][sel_y6][y6_subsample]),
                neighbors_y6["mdet"]["noshear"]["mag"][sel_y6][y6_subsample],
                neighbors_y6["mdet"]["noshear"]["neighbor_distance"][sel_y6][y6_subsample],
            ],
            axis=-1,
        )
        W_y6 = weight_y6[sel_y6][y6_subsample]

        X = np.concatenate([X_sim, X_y6])
        W = np.concatenate([W_sim, W_y6])

        scaler = StandardScaler()
        scaler.fit(X)

        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=0,
        )
        kmeans.fit(
            scaler.transform(X),
            sample_weight=W,
        )

        scaler_file = f"scaler_{tomographic_bin}.pickle"
        with open(scaler_file, "wb") as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        kmeans_file = f"kmeans_{tomographic_bin}.pickle"
        with open(kmeans_file, "wb") as handle:
            pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # pipe = Pipeline(
        #     [
        #         ("scaler", scaler),
        #         ("kmeans", kmeans),
        #     ]
        # )
        # pipe.set_params(kmeans__sample_weights=W)
        # pipe.fit(X)

    return 0

# def predict():
#     for shear_step in lib.const.SHEAR_STEPS:
#         out = np.full(shear_sim_plus["mdet/noshear"]["uid"].shape, np.nan)
#         for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:

#             # TODO need to load correct data for the sims now...
#             X_sim_plus = np.stack([np.log10(neighbor_distance_sim_plus), source_magnitude_sim_plus, neighbor_magnitude_sim_plus], axis=-1)
#             y_sim_plus = kmeans.predict(scaler.transform(X_sim_plus))

#             X_sim_minus = np.stack([np.log10(neighbor_distance_sim_minus), source_magnitude_sim_minus, neighbor_magnitude_sim_minus], axis=-1)
#             y_sim_minus = kmeans.predict(scaler.transform(X_sim_minus))

#             y_y6 = kmeans.predict(scaler.transform(X_y6))

#             w_bins = np.arange(N_CLUSTERS)

#             w_plus = np.bincount(y_y6) / np.bincount(y_sim_plus)
#             w_plus /= np.mean(w_plus)

#             w_minus = np.bincount(y_y6) / np.bincount(y_sim_minus)
#             w_minus /= np.mean(w_minus)


#             _ind = np.digitize(
#                 y_sim_plus,
#                 w_bins,
#                 right=True,
#             )

#             out[ind_sim_plus[:, 0]][_ind] = w_plus[_ind]

#     return 0

if __name__ == "__main__":
    main()
