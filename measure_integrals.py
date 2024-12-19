import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import h5py

import weights


MDET_STEPS = ["noshear", "1p", "1m", "2p", "2m"]


def concatenate_catalogs(data):
    _dp, _dm = np.stack(data, axis=1)
    dp = {
        mdet_step: np.hstack(
            [_d[mdet_step] for _d in _dp],
        )
        for mdet_step in MDET_STEPS
    }
    dm = {
        mdet_step: np.hstack(
            [_d[mdet_step] for _d in _dm],
        )
        for mdet_step in MDET_STEPS
    }
    return dp, dm


def process_file(*, dset, tile):
    mdet = dset["mdet"]

    res = {}
    for mdet_step in MDET_STEPS:
        mdet_cat = mdet[mdet_step]
        in_tile = mdet_cat["tilename"][:] == tile
        _w = weights.get_shear_weights(mdet_cat, in_tile)
        n = np.sum(_w)
        if n > 0:
            g1 = np.average(mdet_cat["gauss_g_1"][in_tile], weights=_w)
            g2 = np.average(mdet_cat["gauss_g_2"][in_tile], weights=_w)
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


def process_file_pair(dset_plus, dset_minus, *, tile):
    dp = process_file(dset=dset_plus, tile=tile)
    dm = process_file(dset=dset_minus, tile=tile)

    return dp, dm


def compute_shear_pair(dp, dm):
    g1_p = np.nansum(dp["noshear"]["g1"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    # g1p_p = np.nansum(dp["1p"]["g1"] * dp["1p"]["n"]) / np.nansum(dp["1p"]["n"])
    # g1m_p = np.nansum(dp["1m"]["g1"] * dp["1m"]["n"]) / np.nansum(dp["1m"]["n"])
    # R11_p = (g1p_p - g1m_p) / 0.02

    g1_m = np.nansum(dm["noshear"]["g1"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    # g1p_m = np.nansum(dm["1p"]["g1"] * dm["1p"]["n"]) / np.nansum(dm["1p"]["n"])
    # g1m_m = np.nansum(dm["1m"]["g1"] * dm["1m"]["n"]) / np.nansum(dm["1m"]["n"])
    # R11_m = (g1p_m - g1m_m) / 0.02

    # g2_p = np.nansum(dp["noshear"]["g2"] * dp["noshear"]["n"]) / np.nansum(dp["noshear"]["n"])
    # g2p_p = np.nansum(dp["2p"]["g2"] * dp["2p"]["n"]) / np.nansum(dp["2p"]["n"])
    # g2m_p = np.nansum(dp["2m"]["g2"] * dp["2m"]["n"]) / np.nansum(dp["2m"]["n"])
    # R22_p = (g2p_p - g2m_p) / 0.02

    # g2_m = np.nansum(dm["noshear"]["g2"] * dm["noshear"]["n"]) / np.nansum(dm["noshear"]["n"])
    # g2p_m = np.nansum(dm["2p"]["g2"] * dm["2p"]["n"]) / np.nansum(dm["2p"]["n"])
    # g2m_m = np.nansum(dm["2m"]["g2"] * dm["2m"]["n"]) / np.nansum(dm["2m"]["n"])
    # R22_m = (g2p_m - g2m_m) / 0.02

    return (
        (g1_p - g1_m), # dg_obs
        2 * 0.02,      # dg_true
    )


def process_pair(catalog_p, catalog_m, seed=None, resample="jackknife"):

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

    hf_minus = h5py.File(
        catalog_m,
        mode="r",
        locking=False,
    )

    tilenames_p = np.unique(hf_plus["mdet"]["noshear"]["tilename"][:])
    tilenames_m = np.unique(hf_minus["mdet"]["noshear"]["tilename"][:])
    tilenames = np.intersect1d(tilenames_p, tilenames_m)
    ntiles = len(tilenames)

    data = [
        process_file_pair(hf_plus, hf_minus, tile=tile)
        for tile in tqdm.tqdm(
            tilenames,
            total=ntiles,
            desc="processing",
            ncols=80,
        )
    ]

    if resample == "bootstrap":
        ns = 1000  # number of bootstrap resamples
        rng = np.random.RandomState(seed=seed)

        dp, dm = concatenate_catalogs(data)
        dg_obs, dg_true = compute_shear_pair(dp, dm)

        bootstrap = []
        for i in tqdm.trange(ns, desc="bootstrap", ncols=80):
            rind = rng.choice(data.shape[0], size=data.shape[0], replace=True)
            _bootstrap = data[rind]
            _dp, _dm = concatenate_catalogs(_bootstrap)
            bootstrap.append(compute_shear_pair(_dp, _dm))

        bootstrap = np.array(bootstrap)
        dg_obs_std, dg_true_std = np.std(bootstrap, axis=0)

    elif resample == "jackknife":
        jackknife = []
        for i in tqdm.trange(len(data), desc="jackknife", ncols=80):
            _pre = data[:i]
            _post = data[i + 1:]
            _jackknife = _pre + _post
            _dp, _dm = concatenate_catalogs(_jackknife)
            jackknife.append(compute_shear_pair(_dp, _dm))

        _n = len(jackknife)
        # m_mean, c_mean_1, c_mean_2 = np.mean(jackknife, axis=0)
        jackknife_mean = np.mean(jackknife, axis=0)
        # jackknife_var = ((_n - 1) / _n) * np.sum(np.square(np.subtract(jackknife, jackknife_mean)), axis=0)
        jackknife_std = np.sqrt(((_n - 1) / _n) * np.sum(np.square(np.subtract(jackknife, jackknife_mean)), axis=0))

        dg_obs, dg_true = jackknife_mean
        dg_obs_std, dg_true_std = jackknife_std

    results = np.array(
        [(dg_obs, dg_obs_std, dg_true, dg_true_std, zlow, zhigh, ntiles)],
        dtype=[
            ("dg_obs_mean", "f8"),
            ("dg_obs_std", "f8"),
            ("dg_true_mean", "f8"),
            ("dg_true_std", "f8"),
            ("zlow", "f8"),
            ("zhigh", "f8"),
            ("ntiles", "i8"),
        ],
    )

    return results


def main():
    # input_dir = Path("/pscratch/sd/s/smau/y6-image-sims-cats")
    input_dir = Path("/global/cfs/cdirs/des/y6-image-sims/fiducial-400/")

    kwargs = {"seed": None, "resample": "jackknife"}

    sim_variants = [
        "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
        "g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7",
        "g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0",
    ]

    sims = {}
    for variant in sim_variants:
        # catalog_file = input_dir / "fiducial" / variant / "metadetect_cutsv6_all.h5"
        catalog_file = input_dir / variant / "metadetect_cutsv6_all.h5"
        assert catalog_file.is_file()
        sims[variant] = catalog_file

    constant_pair = (
        sims["g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"],
        sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
    )
    pairs = [
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
        (
            sims["g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0"],
            sims["g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"]
        ),
    ]

    results = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(32, len(pairs))) as executor:
        for pair in pairs:
            future = executor.submit(process_pair, *pair, **kwargs)
            futures.append(future)

        for future in futures:
            results.append(future.result())

    results = np.concatenate(results)

    zcenter = 0.5 * (results["zlow"] + results["zhigh"])
    zdiff = results["zhigh"] - results["zlow"]

    N = results["dg_obs_mean"] / results["dg_true_mean"]
    # N_error = 3 * results["dg_obs_std"] / results["dg_true_std"]
    N_error = 3 * np.abs(N) * np.hypot(results["dg_obs_std"] / results["dg_obs_mean"], results["dg_true_std"] / results["dg_true_mean"])

    print(f"alpha zlow zhigh N_mean N_cov")
    for i, (zl, zh, n, n_error) in enumerate(zip(results["zlow"], results["zhigh"], N, N_error)):
        # print(f"alpha={i}, z=[{zl}, {zh}), {n}, {n_error}")
        print(f"{i} {zl} {zh} {n} {(n_error/3)**2}")

    fig, axs = plt.subplots(1, 1)

    axs.axhline(0, ls="--")

    axs.errorbar(
        zcenter,
        N / zdiff,
        [
            N_error / zdiff,
            N_error / zdiff,
        ],
        [
            zcenter - results["zlow"],
            results["zhigh"] - zcenter,
        ],
        fmt="none",
    )
    for i, (zl, zh) in enumerate(zip(results["zlow"], results["zhigh"])):
        axs.axvline(zh, ls=":")
        # axs.text(
        #     zh,
        #     0,
        #     f"$\\alpha = {i}$",
        #     ha="right",
        #     va="center",
        # )

    secax = axs.secondary_xaxis("top")
    secax.set_ticks(
        zcenter,
        labels=[f"$\\alpha = {i}$" for i, z in enumerate(zcenter)],
        minor=False,
        ha="left",
        rotation=45,
    )
    secax.set_ticks(
        [],
        minor=True,
    )
    secax.tick_params(
        direction="out",
    )

    axs.set_xlabel("$z$")
    axs.set_ylabel("$N_{\\gamma}^{\\alpha} / \\Delta z^{\\alpha}$")
    axs.set_xlim(0, 6)

    plt.savefig("Ngammaalpha.pdf")
    plt.show()


if __name__ == "__main__":
    main()
