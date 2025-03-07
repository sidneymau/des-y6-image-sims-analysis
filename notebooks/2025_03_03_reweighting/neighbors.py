import concurrent.futures
import os

import h5py
import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors

import lib


COLUMNS = [
    "uid",
    "mag",
    "neighbor_mag",
    "neighbor_distance",
]

def _done(future):
    print(f"future {id(future)} exited with status {future.result()}")

    return 0


def _pre_main(hf_shear_file, hf_redshift_file, hf_neighbor_file):
    with (
        h5py.File(hf_shear_file, "r", locking=False) as hf_shear,
        h5py.File(hf_redshift_file, "r", locking=False) as hf_redshift,
        h5py.File(hf_neighbor_file, "w", locking=False) as hf_neighbor,
    ):
        mdet_group = hf_neighbor.create_group("mdet")

        for mdet_step in lib.const.MDET_STEPS:

            shear_group = mdet_group.create_group(mdet_step)

            _n = hf_shear["mdet"][mdet_step]["uid"].len()
            _data = np.full(_n, np.nan)

            for COLUMN in COLUMNS:
                shear_group.create_dataset(COLUMN, data=_data)

            del _data

    return 0
    

def _main(hf_shear_file, hf_redshift_file, hf_neighbor_file):
    with (
        h5py.File(hf_shear_file, "r", locking=False) as hf_shear,
        h5py.File(hf_redshift_file, "r", locking=False) as hf_redshift,
        h5py.File(hf_neighbor_file, "r+", locking=False) as hf_neighbor,
    ):

        for mdet_step in lib.const.MDET_STEPS:

            bhat = lib.tomography.get_tomography(hf_shear, hf_redshift, mdet_step)
            mag = lib.util.flux_to_mag(
                np.mean(
                    [
                        hf_shear["mdet"][mdet_step]["pgauss_band_flux_r"][:],
                        hf_shear["mdet"][mdet_step]["pgauss_band_flux_i"][:],
                        hf_shear["mdet"][mdet_step]["pgauss_band_flux_z"][:],
                    ],
                    axis=0,
                )
            )
            points = np.deg2rad(
                np.stack(
                    [
                        hf_shear["mdet"][mdet_step]["ra"][:],
                        hf_shear["mdet"][mdet_step]["dec"][:],
                    ],
                    axis=-1
                )
            )

            for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
                print(os.path.basename(hf_neighbor_file), mdet_step, tomographic_bin)

                # sel = np.isin(
                #     cell,
                #     lib.const.CELL_ASSIGNMENTS[tomographic_bin],
                # )
                sel = (bhat == tomographic_bin)
                _mag = mag[sel]
                _points = points[sel]
                # tree = BallTree(
                #     _points,
                #     metric="haversine",
                #     leaf_size=4_000,
                # )
                # dist, ind = tree.query(
                #     _points,
                #     k=2,
                #     return_distance=True,  # default, but we depend on this behavior
                #     sort_results=True,  # default, but we depend on this behavior
                # )
                neighbors = NearestNeighbors(
                    n_neighbors=2,
                    metric="haversine",
                    n_jobs=4,
                )
                neighbors.fit(_points)
                dist, ind = neighbors.kneighbors(
                    _points,
                    n_neighbors=2,
                    return_distance=True,
                )

                hf_neighbor["mdet"][mdet_step]["uid"][sel] = hf_shear["mdet"][mdet_step]["uid"][sel]
                hf_neighbor["mdet"][mdet_step]["mag"][sel] = _mag[ind[:, 0]]
                hf_neighbor["mdet"][mdet_step]["neighbor_mag"][sel] = _mag[ind[:, 1]]
                hf_neighbor["mdet"][mdet_step]["neighbor_distance"][sel] = dist[:, 1]

    return 0

def main():
    neighbor_dir = "/pscratch/sd/s/smau/fiducial-neighbors"
    os.makedirs(neighbor_dir, exist_ok=True)

    futures = []

    # n_jobs = len(lib.const.SHEAR_STEPS) + 1
    n_jobs = 16
    with concurrent.futures.ProcessPoolExecutor(n_jobs) as executor:

        # first, y6
        neighbor_filename_y6 = f"neighbors_y6.hdf5"
        hf_neighbor_file_y6 = os.path.join(
            neighbor_dir,
            neighbor_filename_y6,
        )

        hf_shear_file_y6 = lib.const.Y6_SHEAR_CATALOG
        hf_redshift_file_y6 = lib.const.Y6_REDSHIFT_CATALOG

        _pre_main(hf_shear_file_y6, hf_redshift_file_y6, hf_neighbor_file_y6)

        _future = executor.submit(
            _main,
            hf_shear_file_y6,
            hf_redshift_file_y6,
            hf_neighbor_file_y6,
        )
        _future.add_done_callback(_done)
        futures.append(_future)

        # next, sims
        for shear_step in lib.const.SHEAR_STEPS:
            neighbor_filename_sim = f"neighbors_{shear_step}.hdf5"
            hf_neighbor_file_sim = os.path.join(
                neighbor_dir,
                neighbor_filename_sim,
            )

            hf_shear_file_sim = lib.const.SIM_SHEAR_CATALOGS[shear_step]
            hf_redshift_file_sim = lib.const.SIM_REDSHIFT_CATALOGS[shear_step]

            _pre_main(hf_shear_file_sim, hf_redshift_file_sim, hf_neighbor_file_sim)

            _future = executor.submit(
                _main,
                hf_shear_file_sim,
                hf_redshift_file_sim,
                hf_neighbor_file_sim,
            )
            _future.add_done_callback(_done)
            futures.append(_future)

    concurrent.futures.wait(futures)
    for future in futures:
        print(f"future {id(future)} exited with status {future.result()}")

    return 0

if __name__ == "__main__":
    main()
