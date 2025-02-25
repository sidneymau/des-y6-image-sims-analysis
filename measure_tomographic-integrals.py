import concurrent.futures
import itertools
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import h5py

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


def process_file(*, dset, bhat, tile, tomographic_bin=None):
    mdet = dset["mdet"]

    res = {}
    for mdet_step in lib.const.MDET_STEPS:
        mdet_cat = mdet[mdet_step]
        in_tile = mdet_cat["tilename"][:] == tile
        in_tomo = bhat[mdet_step] == tomographic_bin
        sel = in_tile & in_tomo

        _w = lib.weight.get_shear_weights(mdet_cat, sel=sel)
        n = np.sum(_w)
        if n > 0:
            g1 = np.average(mdet_cat["gauss_g_1"][sel], weights=_w)
            g2 = np.average(mdet_cat["gauss_g_2"][sel], weights=_w)
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


def process_file_pair(dset_plus, dset_minus, bhat_plus, bhat_minus, *, tile, tomographic_bin=None):
    dp = process_file(dset=dset_plus, bhat=bhat_plus, tile=tile, tomographic_bin=tomographic_bin)
    dm = process_file(dset=dset_minus, bhat=bhat_minus, tile=tile, tomographic_bin=tomographic_bin)

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


def process_pair(catalog_p, catalog_m, redshift_catalog_p, redshift_catalog_m, seed=None, resample="jackknife"):

    parts_p = Path(catalog_p).parts
    parts_m = Path(catalog_m).parts

    config_p = parts_p[-3]
    config_m = parts_m[-3]

    assert config_p == config_m

    shear_p = parts_p[-2].split("__")
    shear_m = parts_m[-2].split("__")

    # plus sim is shear slice
    # minus sim is constant shear
    zlow = float(shear_p[4].split("=")[1])
    zhigh = float(shear_p[5].split("=")[1])

    hf_plus = h5py.File(
        catalog_p,
        mode="r",
        locking=False,
    )
    hf_redshift_plus = h5py.File(
        redshift_catalog_p,
        mode="r",
        locking=False,
    )

    hf_minus = h5py.File(
        catalog_m,
        mode="r",
        locking=False,
    )
    hf_redshift_minus = h5py.File(
        redshift_catalog_m,
        mode="r",
        locking=False,
    )

    bhat_plus = {
        mdet_step: lib.tomography.get_tomography(
            hf_plus,
            hf_redshift_plus,
            mdet_step,
        )
        for mdet_step in lib.const.MDET_STEPS
    }
    bhat_minus = {
        mdet_step: lib.tomography.get_tomography(
            hf_minus,
            hf_redshift_minus,
            mdet_step,
        )
        for mdet_step in lib.const.MDET_STEPS
    }

    tilenames_p = np.unique(hf_plus["mdet"]["noshear"]["tilename"][:])
    tilenames_m = np.unique(hf_minus["mdet"]["noshear"]["tilename"][:])
    tilenames = np.intersect1d(tilenames_p, tilenames_m)
    # tilenames = tilenames[:10]  # FIXME
    ntiles = len(tilenames)

    results = {}
    for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
        data = [
            process_file_pair(hf_plus, hf_minus, bhat_plus, bhat_minus, tile=tile, tomographic_bin=tomographic_bin)
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
    shear_catalogs = lib.const.SIM_SHEAR_CATALOGS

    redshift_catalogs = lib.const.SIM_REDSHIFT_CATALOGS

    kwargs = {"seed": None, "resample": "jackknife"}
    # kwargs = {"seed": 42, "resample": "bootstrap"}

    shear_constant_catalog_pair = (
        shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
        shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
    )
    redshift_constant_catalog_pair = (
        redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
        redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
    )

    shear_catalog_pairs = [
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            shear_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0"],
            shear_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
    ]
    redshift_catalog_pairs = [
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            redshift_catalogs["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0"],
            redshift_catalogs["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
    ]

    # /// FIXME
    # process_pair(*shear_constant_pair, *redshift_constant_pair, **kwargs)
    # ///

    results = {}
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(32, len(shear_catalog_pairs) + 1)) as executor:
        # constant shear
        _future = executor.submit(process_pair, *shear_constant_catalog_pair, *redshift_constant_catalog_pair, **kwargs)
        # we let alpha=-1 represent the constant shear simulation
        futures[-1] = _future

        # redshift-dependent shear
        for alpha, (shear_catalog_pair, redshift_catalog_pair) in enumerate(zip(shear_catalog_pairs, redshift_catalog_pairs)):
            _future = executor.submit(process_pair, *shear_catalog_pair, *redshift_catalog_pair, **kwargs)
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

    # cov /= dg_true**2
    # mean /= dg_true

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

    zbinsc = lib.const.ZVALS

    with h5py.File(redshift_constant_catalog_pair[0], "r") as hf_redshift_plus, h5py.File(redshift_constant_catalog_pair[1], "r") as hf_redshift_minus:
        # _zbinsc_plus = hf_redshift_plus["sompz"]["pzdata_weighted_sompz_dz005"]["zbinsc"][:]
        # _zbinsc_minus = hf_redshift_minus["sompz"]["pzdata_weighted_sompz_dz005"]["zbinsc"][:]

        # np.testing.assert_array_equal(_zbinsc_plus, _zbinsc_minus)
        # zbinsc = _zbinsc_plus

        nz_sompz = {}
        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            _nz_sompz_plus = hf_redshift_plus["sompz"]["pzdata_weighted_sompz_dz005"][f"bin{tomographic_bin}"][:]
            _nz_sompz_minus = hf_redshift_minus["sompz"]["pzdata_weighted_sompz_dz005"][f"bin{tomographic_bin}"][:]
            _nz_sompz = (_nz_sompz_plus + _nz_sompz_minus) / 2
            nz_sompz[f"bin{tomographic_bin}"] = _nz_sompz

            _nz_true_plus = hf_redshift_plus["sompz"]["pzdata_weighted_true_dz005"][f"bin{tomographic_bin}"][:]
            _nz_true_minus = hf_redshift_minus["sompz"]["pzdata_weighted_true_dz005"][f"bin{tomographic_bin}"][:]
            _nz_true = (_nz_true_plus + _nz_true_minus) / 2
            nz_true[f"bin{tomographic_bin}"] = _nz_true

    # ---

    with h5py.File("N_gamma_alpha.hdf5", "r+") as hf:

        redshift_group = hf.create_group("redshift")
        # redshift_group.create_dataset("zbinsc", data=zbinsc)

        sompz_redshift_group = redshift_group.create_group("sompz")
        true_redshift_group = redshift_group.create_group("true")

        for tomographic_bin in lib.const.TOMOGRAPHIC_BINS:
            groupname = f"bin{tomographic_bin}"
            # redshift_group.create_dataset(groupname, data=nz_sompz[groupname])

            sompz_redshift_group.create_dataset(groupname, data=nz_sompz[groupname])
            sompz_redshift_group.create_dataset("zbinsc", data=zbinsc)

            true_redshift_group.create_dataset(groupname, data=nz_true[groupname])
            true_redshift_group.create_dataset("zbinsc", data=zbinsc)


if __name__ == "__main__":
    main()
