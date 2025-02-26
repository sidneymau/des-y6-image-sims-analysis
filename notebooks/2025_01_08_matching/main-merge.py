import os

import numpy as np
import h5py

import lib


def main():
    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    match_filename_plus = f"match_{shear_step_plus}.hdf5"
    match_filename_minus = f"match_{shear_step_minus}.hdf5"
    match_filename_merge = f"match_merge.hdf5"

    match_file_plus = os.path.join(
        "/global/cfs/cdirs/des/y6-image-sims/fiducial-matches",
        match_filename_plus,
    )
    match_file_minus = os.path.join(
        "/global/cfs/cdirs/des/y6-image-sims/fiducial-matches",
        match_filename_minus,
    )

    match_file_merge = os.path.join(
        "/global/cfs/cdirs/des/y6-image-sims/fiducial-matches",
        match_filename_merge,
    )

    with (
        h5py.File(match_file_plus, "r") as hf_plus,
        h5py.File(match_file_minus, "r") as hf_minus,
        h5py.File(match_file_merge, "w") as hf_merge,
    ):
        columns_plus = hf_plus["mdet"]["noshear"].keys()
        columns_minus = hf_minus["mdet"]["noshear"].keys()
        assert columns_plus == columns_minus
        columns = columns_plus

        n_rows_plus = hf_plus["mdet"]["noshear"]["uid"].len()
        n_rows_minus = hf_minus["mdet"]["noshear"]["uid"].len()

        n_rows_merge = n_rows_plus + n_rows_minus

        _data = np.full(n_rows_merge, np.nan)
        _data_str = np.full(n_rows_merge, "".encode("ascii"), dtype="S8")

        mdet_group = hf_merge.create_group("mdet")
        shear_group = mdet_group.create_group("noshear")

        for column in columns:
            shear_group.create_dataset(column, data=_data)
        shear_group.create_dataset("shear_step", data=_data_str)

        for column in columns:
            shear_group[column][:n_rows_plus] = hf_plus["mdet"]["noshear"][column][:]
            shear_group[column][n_rows_plus:] = hf_minus["mdet"]["noshear"][column][:]

        shear_group["shear_step"][:n_rows_plus] = "plus".encode("ascii")
        shear_group["shear_step"][n_rows_plus:] = "minus".encode("ascii")




if __name__ == "__main__":
    main()
