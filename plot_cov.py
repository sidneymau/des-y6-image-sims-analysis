import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="input file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with h5py.File(args.input_file, "r") as hf:
        cov = hf["shear/cov"]
        xi = hf["shear/mean_params"]
        alphas = np.unique(xi[:, 0])
        tomos = np.unique(xi[:, 1])

        fig, axs = plt.subplots(1, 1)

        im = axs.imshow(cov, norm=mpl.colors.CenteredNorm(), cmap="RdBu_r")
        fig.colorbar(im, ax=axs)

        # secax = axs.secondary_xaxis("top")
        # secax.set_ticks(
        #     np.arange(len(xi)),
        #     labels=[f"$\\alpha = {i}$, $\\hat{{b}} = {j}$" for (i, j) in xi],
        #     minor=False,
        #     ha="left",
        #     rotation=45,
        # )
        # secax.set_ticks(
        #     [],
        #     minor=True,
        # )
        # secax.tick_params(
        #     direction="out",
        # )

        axs.set_xticks(
            np.intersect1d(alphas, xi[:, 0], return_indices=True)[-1],
            labels=[f"$\\alpha = {alpha}$" for alpha in alphas],
            minor=False,
            ha="right",
            rotation=45,
        )
        axs.set_xticks(
            [],
            minor=True,
        )

        axs.set_yticks(
            np.intersect1d(tomos, xi[:, 1], return_indices=True)[-1],
            labels=[f"$\\hat{{b}} = {tomo}$" for tomo in tomos],
            minor=False,
            ha="right",
            rotation=45,
        )
        axs.set_yticks(
            [],
            minor=True,
        )

        axs.tick_params(
            direction="out",
            top=False,
            right=False,
        )

        fig.savefig("cov.pdf")

        #---

        corr = np.corrcoef(cov)

        fig, axs = plt.subplots(1, 1)

        im = axs.imshow(corr, norm=mpl.colors.CenteredNorm(), cmap="RdBu_r")
        fig.colorbar(im, ax=axs)

        # secax = axs.secondary_xaxis("top")
        # secax.set_ticks(
        #     np.arange(len(xi)),
        #     labels=[f"$\\alpha = {i}$, $\\hat{{b}} = {j}$" for (i, j) in xi],
        #     minor=False,
        #     ha="left",
        #     rotation=45,
        # )
        # secax.set_ticks(
        #     [],
        #     minor=True,
        # )
        # secax.tick_params(
        #     direction="out",
        # )
        # axs.set_xticks(
        #     [],
        # )
        # axs.set_yticks(
        #     [],
        # )

        axs.set_xticks(
            np.intersect1d(alphas, xi[:, 0], return_indices=True)[-1],
            labels=[f"$\\alpha = {alpha}$" for alpha in alphas],
            minor=False,
            ha="right",
            rotation=45,
        )
        axs.set_xticks(
            [],
            minor=True,
        )

        axs.set_yticks(
            np.intersect1d(tomos, xi[:, 1], return_indices=True)[-1],
            labels=[f"$\\hat{{b}} = {tomo}$" for tomo in tomos],
            minor=False,
            ha="right",
            rotation=45,
        )
        axs.set_yticks(
            [],
            minor=True,
        )

        axs.tick_params(
            direction="out",
            top=False,
            right=False,
        )

        fig.savefig("corr.pdf")
