import os
import concurrent.futures
import functools

import h5py
import numpy as np
from sklearn.neighbors import BallTree
import fitsio
import galsim
import coord

import lib


def _main(shear_step, mdet_step, tilename):
    shear_args = lib.util.parse_shear_arguments(shear_step)

    g1_slice = shear_args["g1_slice"]
    g2_slice = shear_args["g2_slice"]
    g1_other = shear_args["g1_other"]
    g2_other = shear_args["g2_other"]
    zlow = shear_args["zlow"]
    zhigh = shear_args["zhigh"]

    shear_filename = lib.const.SIM_SHEAR_CATALOGS[shear_step]
    match_filename = os.path.join(
        "/pscratch/sd/s/smau/fiducial-matches-noshear",
        f"match_{shear_step}.hdf5",
    )
    pure_filename = os.path.join(
        "/pscratch/sd/s/smau/fiducial-matches-noshear",
        f"pure_{shear_step}.hdf5"
    )


    with (
        h5py.File(shear_filename, mode="r") as hf_shear,
        # h5py.File(lib.const.SIM_MATCH_CATALOGS[shear_step]) as hf_match,
        h5py.File(match_filename, mode="r") as hf_match,
        h5py.File(pure_filename, "r+", locking=False) as hf_pure,
    ):

        in_tile = (hf_shear["mdet"][mdet_step]["tilename"][:].astype(str) == tilename)

        if sum(in_tile) > 0:
            catalog_indices = np.indices(
                in_tile.shape,
            ).ravel()

            fits = fitsio.FITS(lib.const.TRUTH_CATALOGS[tilename])
            wcs = lib.util.load_wcs(tilename)
            truth_table = fits[1]

            shear_slice = galsim.Shear(g1=g1_slice, g2=g2_slice)
            shear_other = galsim.Shear(g1=g1_other, g2=g2_other)

            sheared_radec = []
            for i in range(truth_table.get_nrows()):
                ra = truth_table["ra_sim"][i]
                dec = truth_table["dec_sim"][i]
                redshift = truth_table["photoz"][i]

                world_pos = coord.CelestialCoord(ra=ra * coord.degrees, dec=dec * coord.degrees)
                u, v = wcs.center.project(world_pos, projection="gnomonic")
                pos = galsim.PositionD(u.rad, v.rad)

                if (redshift > zlow) & (redshift <= zhigh):
                    sheared_pos = pos.shear(shear_slice)
                else:
                    sheared_pos = pos.shear(shear_other)

                u2 = sheared_pos.x * coord.radians
                v2 = sheared_pos.y * coord.radians
                sheared_world_pos = wcs.center.deproject(u2, v2, projection="gnomonic")

                sheared_ra = sheared_world_pos.ra.deg
                sheared_dec = sheared_world_pos.dec.deg

                sheared_radec.append((sheared_ra, sheared_dec))

            truth_points = np.deg2rad(np.array(sheared_radec))

            _, shear_match_index, truth_match_index = np.intersect1d(
                hf_match["mdet"][mdet_step]["cat_index_sim"][:][in_tile],
                truth_table["cat_index_sim"][:],
                return_indices=True,
            )

            np.testing.assert_equal(
                hf_match["mdet"][mdet_step]["cat_index_sim"][:][in_tile][shear_match_index],
                truth_table["cat_index_sim"][:][truth_match_index],
            )
            np.testing.assert_equal(
                hf_match["mdet"][mdet_step]["z"][:][in_tile][shear_match_index],
                truth_table["photoz"][:][truth_match_index],
            )

            if len(truth_points[truth_match_index]) > 0:

                bt = BallTree(
                    truth_points[truth_match_index],
                    metric="haversine",
                )

                query_radius_arcsec = 3
                query_radius = np.deg2rad(query_radius_arcsec / 60 / 60)

                selection = (
                    (truth_table["x_sim"][truth_match_index] >= 250)
                    & (truth_table["x_sim"][truth_match_index] < 9750)
                    & (truth_table["y_sim"][truth_match_index] >= 250)
                    & (truth_table["y_sim"][truth_match_index] < 9750)
                    & (truth_table["mag_i_red_sim"][truth_match_index] >= 15)
                    & (truth_table["mag_i_red_sim"][truth_match_index] < 25.4)
                    & (truth_table["bdf_hlr"][truth_match_index] >= 0)
                    & (truth_table["bdf_hlr"][truth_match_index] < 5)
                    & (truth_table["isgal"][truth_match_index] == 1)
                    & (truth_table["mask_flags"][truth_match_index] == 0)
                )

                match_indices = bt.query_radius(
                    truth_points[truth_match_index],
                    r=query_radius,
                )

                nmatches = np.array(list(map(len, match_indices)))
                # contaminated_sample = nmatches > 1
                # pure_sample = (nmatches == 1)
                pure_sample = np.where(
                    nmatches == 1,
                    1,
                    0,
                )

                _indices = catalog_indices[in_tile][shear_match_index][selection]
                _indices_sort = np.argsort(_indices)
                indices = _indices[_indices_sort]

                # _data[:][in_tile][shear_match_index][selection] = pure_sample[selection]
                hf_pure["mdet"][mdet_step]["pure"][indices] = pure_sample[selection][_indices_sort]

    return 0


def main():
    futures = {}

    tilenames =  open("/global/homes/s/smau/projects/des/y6-image-sims/tiles-y6.txt").read().split()

    with concurrent.futures.ProcessPoolExecutor(64) as executor:
        # for shear_step in lib.const.SHEAR_STEPS:
        for shear_step in ["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0", "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7"]:

            # mdet_group = hf_pure.create_group("mdet")

            # for mdet_step in lib.const.MDET_STEPS:
            #     shear_group = mdet_group.create_group(mdet_step)

            #     _n = hf_shear["mdet"][mdet_step]["uid"].len()

            #     # prepare hdf5 file
            #     _data = np.full(_n, np.nan)

            #     shear_group.create_dataset("pure", data=_data)

            # del _data

            futures[shear_step] = {}
            for mdet_step in lib.const.MDET_STEPS:
                # _n = hf_shear["mdet"][mdet_step]["uid"].len()
                # _data = np.full(_n, np.nan)

                futures[shear_step][mdet_step] = {}
                for tilename in tilenames:
                    future = executor.submit(_main, shear_step, mdet_step, tilename)
                    futures[shear_step][mdet_step][tilename] = future



    for shear_step, _futures in futures.items():
        for mdet_step, __futures in _futures.items():
            for tilename, ___future in __futures.items():
                print(f"{shear_step}, {mdet_step}, {tilename} exited with status {___future.result()}")


if __name__ == "__main__":
    main()
