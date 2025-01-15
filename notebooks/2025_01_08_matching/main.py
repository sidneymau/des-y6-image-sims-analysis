import concurrent.futures
import os

import numpy as np
import h5py
import fitsio

import lib

COLUMNS = [
    "uid",
    # "tilename",
    "gauss_s2n",
    "gauss_T_ratio",
    "ra",
    "dec",
    "z",
]
        
for band in lib.const.DEEPFIELD_BANDS:
    COLUMNS.append(f"DEEP:flux_{band}")
    COLUMNS.append(f"DEEP:flux_err_{band}")

for band in lib.const.BANDS:
    COLUMNS.append(f"WIDE:pgauss_flux_{band}")
    COLUMNS.append(f"WIDE:pgauss_flux_err_{band}")


def _main(catalog, shear_step):
    with h5py.File(catalog, mode="r") as hf_imsim:

        tilenames = np.unique(hf_imsim["mdet"]["noshear"]["tilename"][:].astype(str))[:3]
        
        _n = hf_imsim["mdet"]["noshear"]["tilename"].len()

        # prepare hdf5 file
        _data = np.full(_n, np.nan)
        match_filename = f"match_{shear_step}.hdf5"
        match_file = os.path.join(
            "/pscratch/sd/s/smau/y6-image-sims-matches",
            match_filename,
        )
        with h5py.File(match_file, "w") as hf:
            mdet_group = hf.create_group("mdet")

            for COLUMN in COLUMNS:
                mdet_group.create_dataset(COLUMN, data=_data)

        del _data

        print(f"matching {shear_step}")
        shear_args = lib.util.parse_shear_arguments(shear_step)

        for i, tilename in enumerate(tilenames):
            print(f"{tilename} ({i + 1} / {len(tilenames)})", end="\r", flush=True)

            observed_matched_indices, truth_matched_table = lib.instance.match(
                tilename,
                hf_imsim,
                mdet_step="noshear",
                **shear_args,
            )

            truth_ids = truth_matched_table["des_id"]
            deepfield_ids = lib.deepfield.get_deepfield_ids()
            _, truth_indices, deepfield_indices = np.intersect1d(
                truth_ids,
                deepfield_ids,
                return_indices=True,
            )

            # deepfield_table = lib.deepfield.get_deepfield_table()
            knn = lib.deepfield.get_knn()

            _X = np.array(
                [
                    truth_matched_table[f"flux_{band}"]
                    for band in lib.const.TRUTH_BANDS
                ]
            ).T
            y = knn.predict(_X)
            
            with h5py.File(match_file, "r+") as hf:
                hf["mdet"]["uid"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["uid"][observed_matched_indices]
                # hf["mdet"]["tilename"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["tilename"][observed_matched_indices]
                hf["mdet"]["gauss_s2n"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["gauss_s2n"][observed_matched_indices]
                hf["mdet"]["gauss_T_ratio"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["gauss_T_ratio"][observed_matched_indices]
                hf["mdet"]["ra"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["ra"][observed_matched_indices]
                hf["mdet"]["dec"][observed_matched_indices] = hf_imsim["mdet"]["noshear"]["dec"][observed_matched_indices]

                hf["mdet"]["z"][observed_matched_indices] = truth_matched_table["photoz"]
            

                for i, band in enumerate(lib.const.DEEPFIELD_BANDS):
                    hf["mdet"][f"DEEP:flux_{band}"][observed_matched_indices] = _X[:, i]
                    hf["mdet"][f"DEEP:flux_err_{band}"][observed_matched_indices] = y[:, i]

                for band in lib.const.BANDS:
                    hf["mdet"][f"WIDE:pgauss_flux_{band}"][observed_matched_indices] = hf_imsim["mdet"]["noshear"][f"pgauss_band_flux_{band}"][observed_matched_indices]
                    hf["mdet"][f"WIDE:pgauss_flux_err_{band}"][observed_matched_indices] = hf_imsim["mdet"]["noshear"][f"pgauss_band_flux_err_{band}"][observed_matched_indices]

    return 0


def main():
    futures = {}
    
    n_jobs = len(lib.const.SHEAR_STEPS)
    with concurrent.futures.ProcessPoolExecutor(n_jobs) as executor:
        for shear_step, catalog in lib.const.IMSIM_CATALOGS.items():
            future = executor.submit(_main, catalog, shear_step)
            futures[shear_step] = future  

    for shear_step, future in futures.items():
        print(f"{shear_step}: exited with status {future.result()}")


if __name__ == "__main__":
    main()
