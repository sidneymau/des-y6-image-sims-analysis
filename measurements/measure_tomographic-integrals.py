import concurrent.futures
import functools
import itertools
import operator
import os

import numpy as np
import tqdm
import h5py
from scipy import stats

import lib


ALPHA = {
    -1: (0.0, 6.0),
    0: (0.0, 0.3),
    1: (0.3, 0.6),
    2: (0.6, 0.9),
    3: (0.9, 1.2),
    4: (1.2, 1.5),
    5: (1.5, 1.8),
    6: (1.8, 2.1),
    7: (2.1, 2.4),
    8: (2.4, 2.7),
    9: (2.7, 6.0),
}
ALPHA_BINS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# from Boyan
ZBINSC = np.arange(0.035, 4, 0.05)
ZEDGES = np.arange(0.01, 4.02, 0.05)


def _compute_nz(cells, zs, weights, responses, zedges=ZEDGES, zbinsc=ZBINSC):
    _zs = np.copy(zs)
    _zs[_zs < zbinsc[0]] = zbinsc[0] + 0.001
    _zs[_zs > zbinsc[-1]] = zbinsc[-1] - 0.001

    _nz, _, _, _ = stats.binned_statistic_2d(
        cells,
        _zs,
        weights * responses,
        statistic="sum",
        bins=[lib.const.CELL_IDS, zedges],
    )

    nz = {}
    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        nz[tomographic_bin] = np.sum(
            _nz[lib.const.CELL_ASSIGNMENTS[tomographic_bin]],
            axis=0,
        )

        # renormalize
        nz[tomographic_bin] = nz[tomographic_bin] / np.sum(nz[tomographic_bin]) / np.diff(zedges)

    return nz

def compute_nz(shear_step_plus, shear_step_minus, weight_keys, zedges=ZEDGES, zbinsc=ZBINSC):
    with (
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus]) as shear_plus,
        h5py.File(lib.const.SIM_MATCH_CATALOGS[shear_step_plus]) as truth_plus,
        h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_plus]) as tomo_plus,
        h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_plus]) as weight_plus,
    ):

        c_plus = tomo_plus["sompz"]["noshear"]["cell_wide"][:]
        z_plus = truth_plus["mdet"]["noshear"]["z"][:]
        w_plus = get_weight(weight_plus["mdet"]["noshear"], weight_keys=weight_keys)
        response_plus = lib.response.get_shear_response(shear_plus["mdet"]["noshear"])

        nz_plus = _compute_nz(
            c_plus,
            z_plus,
            w_plus,
            response_plus,
            zedges=zedges,
            zbinsc=zbinsc,
        )

    with (
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_minus]) as shear_minus,
        h5py.File(lib.const.SIM_MATCH_CATALOGS[shear_step_minus]) as truth_minus,
        h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_minus]) as tomo_minus,
        h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_minus]) as weight_minus,
    ):

        c_minus = tomo_minus["sompz"]["noshear"]["cell_wide"][:]
        z_minus = truth_minus["mdet"]["noshear"]["z"][:]
        w_minus = get_weight(weight_minus["mdet"]["noshear"], weight_keys=weight_keys)
        response_minus = lib.response.get_shear_response(shear_minus["mdet"]["noshear"])

        nz_minus = _compute_nz(
            c_minus,
            z_minus,
            w_minus,
            response_minus,
            zedges=zedges,
            zbinsc=zbinsc,
        )

    nz = {}
    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        nz[tomographic_bin] = (nz_plus[tomographic_bin] + nz_minus[tomographic_bin]) / 2.

    return nz, zedges, zbinsc


def get_weight(weight_dataset, weight_keys=["statistical_weight"]):
    return functools.reduce(
        operator.mul,
        [
            weight_dataset[weight_key][:]
            for weight_key in weight_keys
        ],
    )


def concatenate_catalogs(data):
    _dp, _dm = np.stack(data, axis=1)
    dp = {
        mdet_step: np.hstack(
            [_d[mdet_step] for _d in _dp],
        )
        for mdet_step in lib.const.MDET_STEPS
    }
    dm = {
        mdet_step: np.hstack(
            [_d[mdet_step] for _d in _dm],
        )
        for mdet_step in lib.const.MDET_STEPS
    }
    return dp, dm


def process_file(shear, tomo, weight, tile, weight_keys, tomographic_bin=None):
    res = {}
    for mdet_step in lib.const.MDET_STEPS:
        mdet_cat = shear["mdet"][mdet_step]

        # _w = lib.weight.get_shear_weights(mdet_cat, sel=sel)
        # _w = weight_function(mdet_cat, sel=sel)
        weight_dataset = weight["mdet"][mdet_step]
        w = get_weight(weight_dataset, weight_keys)

        in_tile = mdet_cat["tilename"][:] == tile

        # in_tomo = bhat[mdet_step] == tomographic_bin
        in_tomo = (tomo["sompz"][mdet_step]["bhat"][:] == tomographic_bin)
        sel = in_tile & in_tomo

        n = np.sum(w[sel])
        if n > 0:
            g1 = np.average(mdet_cat["gauss_g_1"][sel], weights=w[sel])
            g2 = np.average(mdet_cat["gauss_g_2"][sel], weights=w[sel])
        else:
            g1 = np.nan
            g2 = np.nan

        res[mdet_step] = np.array(
            [(n, g1, g2)],
            dtype=[
                ("n", "f8"),
                ("g1", "f8"),
                ("g2", "f8"),
            ],
        )
    return res


def process_file_pair(shear_plus, shear_minus, tomo_plus, tomo_minus, weight_plus, weight_minus, weight_keys, *, tile, tomographic_bin=None):
    dp = process_file(shear_plus, tomo_plus, weight_plus, tile, weight_keys, tomographic_bin=tomographic_bin)
    dm = process_file(shear_minus, tomo_minus, weight_minus, tile, weight_keys, tomographic_bin=tomographic_bin)

    return dp, dm


def compute_shear_pair(dp, dm):
    g1_p = np.nansum(dp["noshear"]["g1"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    g1p_p = np.nansum(dp["1p"]["g1"] * dp["1p"]["n"]) / np.nansum(dp["1p"]["n"])
    g1m_p = np.nansum(dp["1m"]["g1"] * dp["1m"]["n"]) / np.nansum(dp["1m"]["n"])
    R11_p = (g1p_p - g1m_p) / 0.02

    g1_m = np.nansum(dm["noshear"]["g1"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    g1p_m = np.nansum(dm["1p"]["g1"] * dm["1p"]["n"]) / np.nansum(dm["1p"]["n"])
    g1m_m = np.nansum(dm["1m"]["g1"] * dm["1m"]["n"]) / np.nansum(dm["1m"]["n"])
    R11_m = (g1p_m - g1m_m) / 0.02

    # g2_p = np.nansum(dp["noshear"]["g2"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    # g2p_p = np.nansum(dp["2p"]["g2"] * dp["2p"]["n"]) / np.nansum(dp["2p"]["n"])
    # g2m_p = np.nansum(dp["2m"]["g2"] * dp["2m"]["n"]) / np.nansum(dp["2m"]["n"])
    # R22_p = (g2p_p - g2m_p) / 0.02

    # g2_m = np.nansum(dm["noshear"]["g2"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    # g2p_m = np.nansum(dm["2p"]["g2"] * dm["2p"]["n"]) / np.nansum(dm["2p"]["n"])
    # g2m_m = np.nansum(dm["2m"]["g2"] * dm["2m"]["n"]) / np.nansum(dm["2m"]["n"])
    # R22_m = (g2p_m - g2m_m) / 0.02

    return (
        (g1_p / R11_p - g1_m / R11_m), # dg_obs
        2 * 0.02,      # dg_true
    )


def process_pair(shear_step_plus, shear_step_minus, weight_keys, seed=None, resample="jackknife"):
    shear_plus = h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus])
    tomo_plus = h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_plus])
    weight_plus = h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_plus])

    shear_minus = h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_minus])
    tomo_minus = h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_minus])
    weight_minus = h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_minus])

    # bhat_plus = {
    #     mdet_step: lib.tomography.get_tomography(
    #         hf_plus,
    #         hf_redshift_plus,
    #         mdet_step,
    #     )
    #     for mdet_step in lib.const.MDET_STEPS
    # }
    # bhat_minus = {
    #     mdet_step: lib.tomography.get_tomography(
    #         hf_minus,
    #         hf_redshift_minus,
    #         mdet_step,
    #     )
    #     for mdet_step in lib.const.MDET_STEPS
    # }

    tilenames_p = np.unique(shear_plus["mdet"]["noshear"]["tilename"][:])
    tilenames_m = np.unique(shear_minus["mdet"]["noshear"]["tilename"][:])
    tilenames = np.intersect1d(tilenames_p, tilenames_m)
    # tilenames = tilenames[:10]  # FIXME
    ntiles = len(tilenames)

    results = {}
    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        data = [
            process_file_pair(shear_plus, shear_minus, tomo_plus, tomo_minus, weight_plus, weight_minus, weight_keys, tile=tile, tomographic_bin=tomographic_bin)
            for tile in tqdm.tqdm(
                tilenames,
                total=ntiles,
                desc=f"processing {tomographic_bin}",
                ncols=80,
            )
        ]
        data = np.array(data)

        if resample == "bootstrap":
            ns = 1000  # number of bootstrap resamples
            rng = np.random.RandomState(seed=seed)

            bootstrap = []
            for i in tqdm.trange(ns, desc="bootstrap", ncols=80):
                rind = rng.choice(len(data), size=len(data), replace=True)
                _bootstrap = data[rind]
                _dp, _dm = concatenate_catalogs(_bootstrap)
                _dg_obs, _dg_true = compute_shear_pair(_dp, _dm)
                bootstrap.append(_dg_obs / _dg_true)

            results[tomographic_bin] = np.array(bootstrap)

        elif resample == "jackknife":
            jackknife = []
            for i in tqdm.trange(len(data), desc="jackknife", ncols=80):
                _pre = data[:i]
                _post = data[i + 1:]
                # _jackknife = _pre + _post
                _jackknife = np.concatenate([_pre, _post])
                _dp, _dm = concatenate_catalogs(_jackknife)
                # jackknife.append(compute_shear_pair(_dp, _dm))
                _dg_obs, _dg_true = compute_shear_pair(_dp, _dm)
                jackknife.append(_dg_obs / _dg_true)

            results[tomographic_bin] = np.array(jackknife)

    return results


def main():

    kwargs = {"seed": None, "resample": "jackknife"}
    # kwargs = {"seed": 42, "resample": "bootstrap"}

    shear_constant_step_pair = (
        "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
        "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
    )

    shear_step_pairs = [
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
        (
            "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0",
            "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"
        ),
    ]

    weight_keys = ["statistical_weight"]

    results = {}
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(32, len(shear_step_pairs) + 1)) as executor:
        # constant shear
        _future = executor.submit(
            process_pair,
            *shear_constant_step_pair,
            weight_keys,
            **kwargs,
        )
        # we let alpha=-1 represent the constant shear simulation
        futures[-1] = _future

        # redshift-dependent shear
        for alpha, shear_step_pair in enumerate(shear_step_pairs):
            _future = executor.submit(
                process_pair,
                *shear_step_pair,
                weight_keys,
                **kwargs,
            )
            futures[alpha] = _future

        for alpha, future in futures.items():
            results[alpha] = future.result()

    xi = list(itertools.product(ALPHA_BINS, lib.const.TOMOGRAPHIC_BINS))
    mean_params = np.array(xi)
    cov = list(itertools.product(xi, xi))
    cov_params = np.array(cov).reshape(len(xi), len(xi), 2, 2)
    cov = np.full((len(xi), len(xi)), np.nan)
    mean = np.full(len(xi), np.nan)

    for i in range(mean_params.shape[0]):
        mean_alpha = mean_params[i][0]
        mean_tomo = mean_params[i][1]
        mean_value = np.mean(results[mean_alpha][mean_tomo])
        mean[i] = mean_value

    if kwargs["resample"] == "jackknife":
        print("jackknifing covariance")
        n_jackknife = None
        for i in range(cov_params.shape[0]):
            for j in range(cov_params.shape[1]):
                cov_1_alpha = cov_params[i, j][0][0]
                cov_1_tomo = cov_params[i, j][0][1]
                cov_2_alpha = cov_params[i, j][1][0]
                cov_2_tomo = cov_params[i, j][1][1]
                if n_jackknife is None:
                    n_jackknife_1 = len(results[cov_1_alpha][cov_1_tomo])
                    n_jackknife_2 = len(results[cov_2_alpha][cov_2_tomo])
                    assert n_jackknife_1 == n_jackknife_2
                    n_jackknife = n_jackknife_1
                cov_value = (n_jackknife - 1) / n_jackknife * np.sum(
                    (results[cov_1_alpha][cov_1_tomo] - mean[i]) * (results[cov_2_alpha][cov_2_tomo] - mean[j])
                )
                cov[i, j] = cov_value

    elif kwargs["resample"] == "bootstrap":
        print("bootstrapping covariance")
        n_bootstrap = None
        for i in range(cov_params.shape[0]):
            for j in range(cov_params.shape[1]):
                cov_1_alpha = cov_params[i, j][0][0]
                cov_1_tomo = cov_params[i, j][0][1]
                cov_2_alpha = cov_params[i, j][1][0]
                cov_2_tomo = cov_params[i, j][1][1]
                if n_bootstrap is None:
                    n_bootstrap_1 = len(results[cov_1_alpha][cov_1_tomo])
                    n_bootstrap_2 = len(results[cov_2_alpha][cov_2_tomo])
                    assert n_bootstrap_1 == n_bootstrap_2
                    n_bootstrap = n_bootstrap_1
                cov_value =  1 / (n_bootstrap - 1) * np.sum(
                    (results[cov_1_alpha][cov_1_tomo] - mean[i]) * (results[cov_2_alpha][cov_2_tomo] - mean[j])
                )
                cov[i, j] = cov_value

    with h5py.File("N_gamma_alpha.hdf5", "w") as hf:
        shear_group = hf.create_group("shear")
        shear_group.create_dataset("mean_params", data=mean_params)
        shear_group.create_dataset("mean", data=mean)
        shear_group.create_dataset("cov_params", data=cov_params)
        shear_group.create_dataset("cov", data=cov)

        alpha_group = hf.create_group("alpha")
        for alpha_k, alpha_v in ALPHA.items():
            groupname = f"bin{alpha_k}"
            alpha_group.create_dataset(groupname, data=alpha_v)

    # ---

    # zbinsc = lib.const.ZVALS

    # with (
    #     h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_constant_step_pair[0]]) as hf_redshift_plus,
    #     h5py.File(lib.const.SIM_REDSHIFT_CATALOGS[shear_constant_step_pair[1]]) as hf_redshift_minus,
    # ):
    #     # _zbinsc_plus = hf_redshift_plus["sompz"]["pzdata_weighted_sompz_dz005"]["zbinsc"][:]
    #     # _zbinsc_minus = hf_redshift_minus["sompz"]["pzdata_weighted_sompz_dz005"]["zbinsc"][:]

    #     # np.testing.assert_array_equal(_zbinsc_plus, _zbinsc_minus)
    #     # zbinsc = _zbinsc_plus

    #     nz_sompz = {}
    #     nz_true = {}
    #     for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
    #         _nz_sompz_plus = hf_redshift_plus["sompz"]["pzdata_weighted_sompz_dz005"][f"bin{tomographic_bin}"][:]
    #         _nz_sompz_minus = hf_redshift_minus["sompz"]["pzdata_weighted_sompz_dz005"][f"bin{tomographic_bin}"][:]
    #         _nz_sompz = (_nz_sompz_plus + _nz_sompz_minus) / 2
    #         nz_sompz[f"bin{tomographic_bin}"] = _nz_sompz

    #         _nz_true_plus = hf_redshift_plus["sompz"]["pzdata_weighted_true_dz005"][f"bin{tomographic_bin}"][:]
    #         _nz_true_minus = hf_redshift_minus["sompz"]["pzdata_weighted_true_dz005"][f"bin{tomographic_bin}"][:]
    #         _nz_true = (_nz_true_plus + _nz_true_minus) / 2
    #         nz_true[f"bin{tomographic_bin}"] = _nz_true

    nz, zedges, zbinsc = compute_nz(shear_constant_step_pair[0], shear_constant_step_pair[1], weight_keys, zedges=ZEDGES, zbinsc=ZBINSC)

    # ---

    with h5py.File("N_gamma_alpha.hdf5", "r+") as hf:

        redshift_group = hf.create_group("redshift")
        # redshift_group.create_dataset("zbinsc", data=zbinsc)

        # sompz_redshift_group = redshift_group.create_group("sompz")
        # true_redshift_group = redshift_group.create_group("true")


        # sompz_redshift_group.create_dataset("zbinsc", data=zbinsc)
        # true_redshift_group.create_dataset("zbinsc", data=zbinsc)

        redshift_group.create_dataset("zedges", data=zedges)
        redshift_group.create_dataset("zbinsc", data=zbinsc)


        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            groupname = f"bin{tomographic_bin}"
            # redshift_group.create_dataset(groupname, data=nz_sompz[groupname])

            # sompz_redshift_group.create_dataset(groupname, data=nz_sompz[groupname])
            # true_redshift_group.create_dataset(groupname, data=nz_true[groupname])

            redshift_group.create_dataset(groupname, data=nz[tomographic_bin])


if __name__ == "__main__":
    main()
