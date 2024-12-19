import functools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import h5py

logger = logging.getLogger(__name__)

TOMOGRAPHIC_BINS = [0, 1, 2, 3]
tomo_colors = {
    0: "blue",
    1: "gold",
    2: "green",
    3: "red",
}
ALPHA = {
    # -1: (0.0, 6.0),
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


if __name__ == "__main__":
    shear_steps = [
        'g1_slice=-0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.6__zhigh=0.9',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.8__zhigh=2.1',
        'g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.9__zhigh=1.2',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.1__zhigh=2.4',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.0__zhigh=0.3',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.2__zhigh=1.5',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.4__zhigh=2.7',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=0.3__zhigh=0.6',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=1.5__zhigh=1.8',
        # 'g1_slice=0.02__g2_slice=0.00__g1_other=-0.02__g2_other=0.00__zlow=2.7__zhigh=6.0',
    ]

    N_gamma_alpha = h5py.File("N_gamma_alpha_v0_bs.hdf5", "r")

    redshift_base = "/global/cfs/cdirs/des/y6-redshift/imsim_400Tile/fidbin_S005/"
    redshift_catalogs = {
        shear_step: os.path.join(
            redshift_base,
            f"{shear_step}_sompz_unblind_fidbin.h5"
        )
        for shear_step in shear_steps
    }

    for shear_step in shear_steps:
        hf_redshift = h5py.File(redshift_catalogs[shear_step], mode="r")

        # zlow = hf_redshift["sompz"]["pzdata_weighted_S005"]["zlow"][:]
        # zhigh = hf_redshift["sompz"]["pzdata_weighted_S005"]["zhigh"][:]
        # zcenter = (zlow + zhigh) / 2
        zcenter = hf_redshift["sompz"]["pzdata_weighted_S005"]["zbinsc"][:]

        nzs = {
            tomographic_bin: hf_redshift["sompz"]["pzdata_weighted_S005"][f"bin{tomographic_bin}"][:]
            for tomographic_bin in TOMOGRAPHIC_BINS
        }

        fig, axs = plt.subplots(
            len(TOMOGRAPHIC_BINS),
            1,
            sharex=True,
            sharey=True,
        )

        for tomographic_bin in TOMOGRAPHIC_BINS:
            nz = nzs[tomographic_bin]
            _nz_integral = np.trapz(nz, zcenter)
            nz /= _nz_integral
            axs[tomographic_bin].plot(
                zcenter,
                nz,
                c=tomo_colors[tomographic_bin],
                # label=f"$\\hat{{b}} = {tomographic_bin}$",
            )
            for alpha, (zl, zh) in ALPHA.items():
                delta_z = zh - zl
                alpha_indices = np.nonzero((zcenter >= zl) & (zcenter < zh))
                nz_integral = np.trapz(nz[alpha_indices], zcenter[alpha_indices])
                axs[tomographic_bin].errorbar(
                    (zh + zl) / 2,
                    nz_integral / delta_z,
                    0,
                    delta_z / 2,
                    fmt="none",
                    c=tomo_colors[tomographic_bin],
                    # label="$\\int n(z) / \\Delta z^{\\alpha}$",
                )

            # Ngammaalpha = N_gamma_alpha[f"bin{tomographic_bin}"]
            alpha_sel = N_gamma_alpha["shear/mean_params"][:, 0] >= 0
            tomo_sel = N_gamma_alpha["shear/mean_params"][:, 1] == tomographic_bin
            sel = alpha_sel & tomo_sel
            alphas = N_gamma_alpha["shear/mean_params"][sel, 0]
            Ngammaalpha_mean = N_gamma_alpha["shear/mean"][sel]
            Ngammaalpha_var = np.diag(N_gamma_alpha["shear/cov"])[sel]
            zl = np.array([ALPHA[alpha][0] for alpha in alphas])
            zh = np.array([ALPHA[alpha][1] for alpha in alphas])
            # zl = Ngammaalpha["zlow"]
            # zh = Ngammaalpha["zhigh"]
            zc = 0.5 * (zl + zh)
            zd = zh - zl
            axs[tomographic_bin].errorbar(
                zc,
                Ngammaalpha_mean / zd,
                [
                    Ngammaalpha_var**(1/2) * 3 / zd,
                    Ngammaalpha_var**(1/2) * 3 / zd,
                ],
                [
                    zc - zl,
                    zh - zc,
                ],
                fmt="none",
                c="k",
                # label="$N_{\\gamma}^{\\alpha} / \\Delta z^{\\alpha}$",
            )

        axs[0].set_xlim(0, 3)
        axs[0].set_ylim(0, None)
        axs[-1].set_xlabel("$z$")
        # for ax in axs[:]:
        #     ax.set_ylabel("$n(z)$")
        #     # ax.legend(loc="upper right")
        axs[0].set_title(" ".join(shear_step.split("__")))

        fig.savefig(f"nz-{shear_step}.pdf")

        # ---

        fig, axs = plt.subplots(1, 1)

        axs.axhline(0, ls="--")

        for tomographic_bin in TOMOGRAPHIC_BINS:
            # tomo_results = results[np.nonzero(results["tomographic_bin"] == tomographic_bin)]
            # zlow = tomo_results["zlow"]
            # zhigh = tomo_results["zhigh"]
            # zcenter = 0.5 * (zlow + zhigh)
            # zdiff = zhigh - zlow
            alpha_sel = N_gamma_alpha["shear/mean_params"][:, 0] >= 0
            tomo_sel = N_gamma_alpha["shear/mean_params"][:, 1] == tomographic_bin
            sel = alpha_sel & tomo_sel
            alphas = N_gamma_alpha["shear/mean_params"][sel, 0]
            Ngammaalpha_mean = N_gamma_alpha["shear/mean"][sel]
            Ngammaalpha_var = np.diag(N_gamma_alpha["shear/cov"])[sel]
            zl = np.array([ALPHA[alpha][0] for alpha in alphas])
            zh = np.array([ALPHA[alpha][1] for alpha in alphas])
            # zl = Ngammaalpha["zlow"]
            # zh = Ngammaalpha["zhigh"]
            zc = 0.5 * (zl + zh)
            zd = zh - zl

            # N = tomo_results["dg_obs_mean"] / tomo_results["dg_true_mean"]
            # # N_error = 3 * tomo_results["dg_obs_std"] / tomo_results["dg_true_std"]
            # N_error = 3 * np.abs(N) * np.hypot(tomo_results["dg_obs_std"] / tomo_results["dg_obs_mean"], tomo_results["dg_true_std"] / tomo_results["dg_true_mean"])

            axs.errorbar(
                zc,
                Ngammaalpha_mean / zd,
                [
                    Ngammaalpha_var**(1/2) * 3 / zd,
                    Ngammaalpha_var**(1/2) * 3 / zd,
                ],
                [
                    zc - zl,
                    zh - zc,
                ],
                fmt="none",
                c=tomo_colors[tomographic_bin],
                label=f"$\\hat{{b}} = {tomographic_bin}$",
            )

        for i, (_zl, _zh) in enumerate(zip(zl, zh)):
            axs.axvline(_zh, ls=":")
            # axs.text(
            #     zh,
            #     0,
            #     f"$\\alpha = {i}$",
            #     ha="right",
            #     va="center",
            # )

        secax = axs.secondary_xaxis("top")
        secax.set_ticks(
            zc,
            labels=[f"$\\alpha = {i}$" for i, z in enumerate(zc)],
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
        axs.legend(loc="upper right")

        fig.savefig("Ngammaalpha-tomographic.pdf")
