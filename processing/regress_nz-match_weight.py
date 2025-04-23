import functools
import os
import pickle

import h5py
import numpy as np
from scipy import optimize, stats, special

import lib

COLUMN = "shape_weight"

_PZDIR_SIM = "/global/cfs/cdirs/des/boyan/sompz_output/y6_imsim_1000Tile"
# _DZ = 0.05

def main():

    shear_step_plus = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    shear_step_minus = "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"

    weight_response_catalogs = {
        shear_step: f"SOM_weight_response_{shear_step}.hdf5"
        for shear_step in [shear_step_plus, shear_step_minus]
    }

    nz_match_weights = np.zeros(1024)
    zedges = np.arange(0.01, 3.02, 0.05)

    with (
        h5py.File(lib.const.Y6_REDSHIFT_CATALOG, "r") as shear_y6,
    ):
        zbinsc = shear_y6["sompz"]["pzdata_weighted_S005"]["zbinsc"][:]

        nz_y6 = {}
        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            _nz = shear_y6["sompz"]["pzdata_weighted_S005"][f"bin{tomographic_bin}"][:]
            # nz_y6[tomographic_bin] *= _DZ
            nz_y6[tomographic_bin] = _nz / np.sum(_nz)

            np.testing.assert_allclose(np.sum(nz_y6[tomographic_bin]), 1)

    nz_sim = np.zeros((len(nz_match_weights), len(zedges) - 1))
    for shear_step in [shear_step_plus, shear_step_minus]:
        print(shear_step)

        with (
            h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step], "r") as shear_sim,
            h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step], "r") as tomo_sim,
            h5py.File(lib.const.SIM_MATCH_CATALOGS[shear_step], "r") as truth_sim,
            h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step], "r") as weight_sim,
            h5py.File(weight_response_catalogs[shear_step], "r") as weight_response_sim,
        ):

            # pzchat_sim = np.load(
            #     os.path.join(
            #         _PZDIR_SIM,
            #         shear_step,
            #         "noshear/weighted_pile4_oldtomo/pzchat.npy",
            #     )
            # )
            cell_sim = tomo_sim["sompz"]["noshear"]["cell_wide"][:]
            redshift_sim = truth_sim["mdet"]["noshear"]["z"][:]
            weight_sim = weight_sim["mdet"]["noshear"]["statistical_weight"][:]
            response_sim = lib.response.get_shear_response(shear_sim["mdet"]["noshear"])

            weight_response = weight_response_sim["mdet"]["noshear"]["weight_response"][:]

            wR_a = weight_response[:, np.newaxis]
            # n_a = pzchat_sim[:, :len(zbinsc)]
            # n_a[:, -1] += np.sum(pzchat_sim[:, len(zbinsc):], axis=1)  # pileup in last bin (at z=3)
            # n_a = n_a / np.sum(n_a, axis=1, keepdims=True)
            # # n_a *= _DZ
            _zs = np.copy(redshift_sim)
            _zs[_zs < zbinsc[0]] = zbinsc[0] + 0.001
            _zs[_zs > zbinsc[-1]] = zbinsc[-1] - 0.001

            _nz, _, _, _ = stats.binned_statistic_2d(
                cell_sim,
                _zs,
                weight_sim * response_sim,
                statistic="sum",
                bins=[lib.const.CELL_IDS, zedges],
            )
            n_a = _nz / np.sum(_nz, axis=1, keepdims=True)

            np.testing.assert_allclose(np.sum(n_a, axis=1), 1)

            nz_sim += 0.5 * n_a

    # weights per SOM cell to regress
    b_a = np.ones(len(wR_a))
    log_ba = np.log(b_a)

    # B = 0
    # B = 1e-5  # l2 penalty for high weights
    B = 1e-1
    # B = 0

    def objective(log_x, l2reg=0):
        b_a = np.exp(log_x[:, np.newaxis])
        _loss = 0
        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            _cells = lib.const.CELL_ASSIGNMENTS[tomographic_bin]

            # _model = np.sum(
            #     (b_a * n_a * wR_a)[_cells],
            #     axis=0,
            # ) / np.sum(
            #     (b_a * n_a * wR_a)[_cells],
            # )
            # _model = np.sum(
            # 	(b_a * n_a * wR_a)[_cells],
            # 	axis=0,
            # ) / np.sum((b_a * wR_a)[_cells]) / np.sum(n_a, axis=1, keepdims=True)[_cells]
            _model = np.sum(
                (b_a * (n_a / np.sum(n_a, axis=1, keepdims=True)) * wR_a)[_cells],
                axis=0,
            ) / np.sum((b_a * wR_a)[_cells])

            np.testing.assert_almost_equal(np.sum(_model), 1)
            # np.testing.assert_allclose(np.sum(_model, axis=1), 1)

            # relative difference
            # _loss += np.sum(np.square(nz_y6[tomographic_bin] - _model))
            # _loss += np.sum(
            #     np.abs(
            #         np.cumsum(nz_y6[tomographic_bin]) - np.cumsum(_model)
            #     )
            # )

            # fractional difference
            _loss += np.sum(np.square(_model / nz_y6[tomographic_bin] - 1))
            # _loss += np.sum(
            #     np.abs(
            #         np.cumsum(nz_y6[tomographic_bin]) / np.cumsum(_model) - 1
            #     )
            # )
            # _loss += np.sum(
            #     np.abs(
            #         np.cumsum(nz_y6[tomographic_bin] /_model - 1)
            #     )
            # )

            # _loss += np.sum(special.kl_div(nz_y6[tomographic_bin], _model))

        # _loss +=  l2reg * np.square(np.sum(np.abs(log_x)))
        # _loss +=  l2reg * np.sum(np.square(log_x))
        _loss +=  l2reg * np.sum(np.square(np.exp(log_x) - 1))

        return _loss

    print("optimizing weights...")
    res = optimize.minimize(
        objective,
        log_ba,
        (B, ),
        # method="nelder-mead",
        # bounds=[(0, None) for _ in b_a],
    )
    print(res)

    # nz_match_weights += 0.5 * np.exp(res.x)
    nz_match_weights = np.exp(res.x)

    weights_file = f"nz-match_weight.pickle"
    with open(weights_file, "wb") as handle:
        pickle.dump(nz_match_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return 0


if __name__ == "__main__":
    main()
