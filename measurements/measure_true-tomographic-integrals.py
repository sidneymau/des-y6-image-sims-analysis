import argparse
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

# tomographic_bins = lib.const.TOMOGRAPHIC_BINS

SIM_MATCH_CATALOGS = {
    shear_step: os.path.join(
        "/pscratch/sd/s/smau/fiducial-matches-noshear",
        f"match_{shear_step}.hdf5",
    )
    for shear_step in lib.const.SHEAR_STEPS
}
SIM_PURE_CATALOGS = {
    shear_step: os.path.join(
        "/pscratch/sd/s/smau/fiducial-matches-noshear",
        f"pure_{shear_step}.hdf5",
    )
    for shear_step in lib.const.SHEAR_STEPS
}


def _COLOR_CUTS(d, dc):
    return (
        np.abs(
            lib.const.GI_COLOR - (lib.util.flux_to_mag(d["pgauss_band_flux_g"]) - lib.util.flux_to_mag(d["pgauss_band_flux_i"]))
        ) < dc
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed; only used for bootstrap resampling",
    )
    parser.add_argument(
        "--resample",
        type=str,
        choices=["bootstrap", "jackknife"],
        default="jackknife",
        required=False,
        help="resample algorithm",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=["statistical", "neighbor", "occupancy", "nz"],
        nargs="+",
        default=["statistical"],
        required=False,
        help="weight keys to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--decontaminate",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--dc",
        type=float,
        required=False,
        default=None,
    )

    return parser.parse_args()



def _compute_nz(
    cells,
    zs,
    weights,
    responses,
    zedges=ZEDGES,
    zbinsc=ZBINSC,
):
    tomographic_bins = ALPHA_BINS

    _zs = np.copy(zs)
    _zs[_zs < zbinsc[0]] = zbinsc[0] + 0.001
    _zs[_zs > zbinsc[-1]] = zbinsc[-1] - 0.001

    nz = {}
    for tomographic_bin in tomographic_bins:
        sel = (
            (_zs > ALPHA[tomographic_bin][0])
            & (_zs <= ALPHA[tomographic_bin][1])
        )
        _nz, _, _ = stats.binned_statistic(
            _zs[sel],
            (weights * responses)[sel],
            statistic="sum",
            bins=zedges,
        )

        nz[tomographic_bin] = _nz

        # renormalize
        nz[tomographic_bin] = nz[tomographic_bin] / np.sum(nz[tomographic_bin]) / np.diff(zedges)

    return nz


def compute_nz(
    shear_step_plus,
    shear_step_minus,
    weight_keys,
    zedges=ZEDGES,
    zbinsc=ZBINSC,
    decontaminate=False,
    dc=None,
):
    tomographic_bins = ALPHA_BINS

    with (
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus]) as shear_plus,
        h5py.File(SIM_MATCH_CATALOGS[shear_step_plus]) as truth_plus,
        h5py.File(SIM_PURE_CATALOGS[shear_step_plus]) as pure_plus,
        h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_plus]) as tomo_plus,
        h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_plus]) as weight_plus,
    ):

        c_plus = tomo_plus["sompz"]["noshear"]["cell_wide"][:]
        z_plus = truth_plus["mdet"]["noshear"]["z"][:]
        w_plus = get_weight(weight_plus["mdet"]["noshear"], weight_keys=weight_keys)
        response_plus = lib.response.get_shear_response(shear_plus["mdet"]["noshear"])

        if decontaminate:
            sel = (pure_plus["mdet"]["noshear"]["pure"][:] == 1)
        elif dc is not None:
            sel = _COLOR_CUTS(shear_plus["mdet"]["noshear"], dc)
        else:
            sel = slice(None)

        nz_plus = _compute_nz(
            c_plus[sel],
            z_plus[sel],
            w_plus[sel],
            response_plus[sel],
            zedges=zedges,
            zbinsc=zbinsc,
        )

    with (
        h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_minus]) as shear_minus,
        h5py.File(SIM_MATCH_CATALOGS[shear_step_minus]) as truth_minus,
        h5py.File(SIM_PURE_CATALOGS[shear_step_minus]) as pure_minus,
        h5py.File(lib.const.SIM_TOMOGRAPHY_CATALOGS[shear_step_minus]) as tomo_minus,
        h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_minus]) as weight_minus,
    ):

        c_minus = tomo_minus["sompz"]["noshear"]["cell_wide"][:]
        z_minus = truth_minus["mdet"]["noshear"]["z"][:]
        w_minus = get_weight(weight_minus["mdet"]["noshear"], weight_keys=weight_keys)
        response_minus = lib.response.get_shear_response(shear_minus["mdet"]["noshear"])

        if decontaminate:
            sel = (pure_minus["mdet"]["noshear"]["pure"][:] == 1)
        elif dc is not None:
            sel = _COLOR_CUTS(shear_minus["mdet"]["noshear"], dc)
        else:
            sel = slice(None)

        nz_minus = _compute_nz(
            c_minus[sel],
            z_minus[sel],
            w_minus[sel],
            response_minus[sel],
            zedges=zedges,
            zbinsc=zbinsc,
        )

    nz = {}
    for tomographic_bin in tomographic_bins:
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


def process_file(
    shear,
    match,
    pure,
    weight,
    tile,
    weight_keys,
    tomographic_bin=None,
    decontaminate=False,
    dc=None,
):
    res = {}
    for mdet_step in lib.const.MDET_STEPS:
        mdet_cat = shear["mdet"][mdet_step]

        # _w = lib.weight.get_shear_weights(mdet_cat, sel=sel)
        # _w = weight_function(mdet_cat, sel=sel)
        weight_dataset = weight["mdet"][mdet_step]
        w = get_weight(weight_dataset, weight_keys)

        sel = (mdet_cat["tilename"][:] == tile)

        if decontaminate:
            sel &= (pure["mdet"][mdet_step]["pure"][:] == 1)
        elif dc is not None:
            sel &= _COLOR_CUTS(mdet_cat, dc)

        sel &= (
            (match["mdet"][mdet_step]["z"][:] > ALPHA[tomographic_bin][0])
            & (match["mdet"][mdet_step]["z"][:] <= ALPHA[tomographic_bin][1])
        )

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


def process_file_pair(
    shear_plus,
    shear_minus,
    match_plus,
    match_minus,
    pure_plus,
    pure_minus,
    weight_plus,
    weight_minus,
    weight_keys,
    *,
    tile,
    tomographic_bin=None,
    decontaminate=False,
    dc=None,
):
    dp = process_file(
        shear_plus,
        match_plus,
        pure_plus,
        weight_plus,
        tile,
        weight_keys,
        tomographic_bin=tomographic_bin,
        decontaminate=decontaminate,
        dc=dc,
    )
    dm = process_file(
        shear_minus,
        match_minus,
        pure_minus,
        weight_minus,
        tile,
        weight_keys,
        tomographic_bin=tomographic_bin,
        decontaminate=decontaminate,
        dc=dc,
    )

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

    g2_p = np.nansum(dp["noshear"]["g2"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    g2p_p = np.nansum(dp["2p"]["g2"] * dp["2p"]["n"]) / np.nansum(dp["2p"]["n"])
    g2m_p = np.nansum(dp["2m"]["g2"] * dp["2m"]["n"]) / np.nansum(dp["2m"]["n"])
    R22_p = (g2p_p - g2m_p) / 0.02

    g2_m = np.nansum(dm["noshear"]["g2"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    g2p_m = np.nansum(dm["2p"]["g2"] * dm["2p"]["n"]) / np.nansum(dm["2p"]["n"])
    g2m_m = np.nansum(dm["2m"]["g2"] * dm["2m"]["n"]) / np.nansum(dm["2m"]["n"])
    R22_m = (g2p_m - g2m_m) / 0.02

    R_p = 0.5 * (R11_p + R22_p)
    R_m = 0.5 * (R11_m + R22_m)

    return (
        (g1_p / R_p - g1_m / R_m), # dg_obs
        2 * 0.02,      # dg_true
    )


def process_pair(
    shear_step_plus,
    shear_step_minus,
    weight_keys,
    tomographic_bin,
    seed=None,
    resample="jackknife",
    debug=False,
    decontaminate=False,
    dc=None,
):
    shear_plus = h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_plus])
    match_plus = h5py.File(SIM_MATCH_CATALOGS[shear_step_plus])
    pure_plus = h5py.File(SIM_PURE_CATALOGS[shear_step_plus])
    weight_plus = h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_plus])

    shear_minus = h5py.File(lib.const.SIM_SHEAR_CATALOGS[shear_step_minus])
    match_minus = h5py.File(SIM_MATCH_CATALOGS[shear_step_minus])
    pure_minus = h5py.File(SIM_PURE_CATALOGS[shear_step_minus])
    weight_minus = h5py.File(lib.const.SIM_WEIGHT_CATALOGS[shear_step_minus])

    tilenames_p = np.unique(shear_plus["mdet"]["noshear"]["tilename"][:])
    tilenames_m = np.unique(shear_minus["mdet"]["noshear"]["tilename"][:])
    tilenames = np.intersect1d(tilenames_p, tilenames_m)
    if debug:
        tilenames = tilenames[:10]
    ntiles = len(tilenames)

    data = [
        process_file_pair(
            shear_plus,
            shear_minus,
            match_plus,
            match_minus,
            pure_plus,
            pure_minus,
            weight_plus,
            weight_minus,
            weight_keys,
            tile=tile,
            tomographic_bin=tomographic_bin,
            decontaminate=decontaminate,
            dc=dc,
        )
        for tile in tilenames
    ]
    data = np.array(data)

    if resample == "bootstrap":
        ns = 1000  # number of bootstrap resamples
        rng = np.random.RandomState(seed=seed)

        bootstrap = []
        for i in range(ns):
            rind = rng.choice(len(data), size=len(data), replace=True)
            _bootstrap = data[rind]
            _dp, _dm = concatenate_catalogs(_bootstrap)
            _dg_obs, _dg_true = compute_shear_pair(_dp, _dm)
            bootstrap.append(_dg_obs / _dg_true)

        results = np.array(bootstrap)

    elif resample == "jackknife":
        jackknife = []
        for i in range(len(data)):
            _pre = data[:i]
            _post = data[i + 1:]
            # _jackknife = _pre + _post
            _jackknife = np.concatenate([_pre, _post])
            _dp, _dm = concatenate_catalogs(_jackknife)
            # jackknife.append(compute_shear_pair(_dp, _dm))
            _dg_obs, _dg_true = compute_shear_pair(_dp, _dm)
            jackknife.append(_dg_obs / _dg_true)

        results = np.array(jackknife)

    print(f"done processing {shear_step_plus}, {shear_step_minus}, {tomographic_bin}")
    return results


def main():

    args = get_args()
    print(args)

    seed = args.seed
    resample = args.resample
    weights = args.weights
    debug = args.debug
    decontaminate = args.decontaminate
    dc = args.dc

    tomographic_bins = ALPHA_BINS

    weight_keys = [f"{weight}_weight" for weight in weights]

    output_template = "N_gamma_alpha_v3_{}.hdf5"
    output_tag = "-".join(weights)

    output_tag += "_true-tomo"

    if decontaminate:
        output_tag += "_pure"
    if dc is not None:
        output_tag += f"_dc={dc}"

    output_filename = output_template.format(output_tag)
    print(f"Will write {output_filename}")

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

    print(f"Processing with weights: {weights}")

    results = {}
    futures = {}
    futures[-1] = {}
    results[-1] = {}
    for alpha in range(len(shear_step_pairs)):
        futures[alpha] = {}
        results[alpha] = {}
        for tomographic_bin in tomographic_bins:
            futures[alpha][tomographic_bin] = {}
            results[alpha][tomographic_bin] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(64, (len(shear_step_pairs) + 1) * len(tomographic_bins))) as executor:
        # constant shear
        for tomographic_bin in tomographic_bins:
            _future = executor.submit(
                process_pair,
                *shear_constant_step_pair,
                weight_keys,
                tomographic_bin,
                seed=seed,
                resample=resample,
                debug=debug,
                decontaminate=decontaminate,
                dc=dc,
            )
            # we let alpha=-1 represent the constant shear simulation
            futures[-1][tomographic_bin] = _future

            # redshift-dependent shear
            for alpha, shear_step_pair in enumerate(shear_step_pairs):
                _future = executor.submit(
                    process_pair,
                    *shear_step_pair,
                    weight_keys,
                    tomographic_bin,
                    seed=seed,
                    resample=resample,
                    debug=debug,
                    decontaminate=decontaminate,
                    dc=dc,
                )
                futures[alpha][tomographic_bin] = _future

        for tomographic_bin, _futures in futures.items():
            for alpha, future in _futures.items():
                results[alpha][tomographic_bin] = future.result()

    xi = list(itertools.product(ALPHA_BINS, tomographic_bins))
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

    if resample == "jackknife":
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

    elif resample == "bootstrap":
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

    with h5py.File(output_filename, "w") as hf:
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

    nz, zedges, zbinsc = compute_nz(
        shear_constant_step_pair[0],
        shear_constant_step_pair[1],
        weight_keys,
        zedges=ZEDGES,
        zbinsc=ZBINSC,
        decontaminate=decontaminate,
        dc=dc,
    )

    with h5py.File(output_filename, "r+") as hf:
        redshift_group = hf.create_group("redshift")
        redshift_group.create_dataset("zedges", data=zedges)
        redshift_group.create_dataset("zbinsc", data=zbinsc)

        for tomographic_bin in tomographic_bins:
            groupname = f"bin{tomographic_bin}"
            redshift_group.create_dataset(groupname, data=nz[tomographic_bin])


if __name__ == "__main__":
    main()
