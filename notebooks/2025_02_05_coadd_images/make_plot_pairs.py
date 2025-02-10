import os
import concurrent.futures

import matplotlib.pyplot as plt
from PIL import Image

Y6A1_COADD_DIR = "/pscratch/sd/s/smau/Y6A1_COADD"
Y6A1_COADD_FIDUCIAL_DIR = "/pscratch/sd/s/smau/Y6A1_COADD_fiducial"
OUT_DIR = "/pscratch/sd/s/smau/Y6A1_COADD_PAIRS"

SIZE = (1000, 1000)


def _main(tilename):
    _real_image = os.path.join(
        Y6A1_COADD_DIR,
        f"{tilename}-gri.jpg",
    )
    _sim_image = os.path.join(
        Y6A1_COADD_FIDUCIAL_DIR,
        f"{tilename}-coadd-gri.jpg",
    )
    outfile = os.path.join(
        OUT_DIR,
        f"{tilename}-pair-gri.png",
    )

    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    with (
        Image.open(_real_image) as real_image,
        Image.open(_sim_image) as sim_image
    ):
        real_image.thumbnail(SIZE)
        sim_image.thumbnail(SIZE)


    axs[0].imshow(real_image)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("real")

    axs[1].imshow(sim_image)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("sim")

    fig.suptitle(tilename)

    fig.savefig(outfile, dpi=200)

    fig.close()


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    futures = []

    with open("args-y6.txt", "r") as fp:
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            for i, line in enumerate(fp):
                if i < 525:
                    tilename = line.split(" ")[0]
                    _future = executor.submit(_main, tilename)
                    futures.append(_future)
                else:
                    break

    for future in futures:
        print(".", end="")

if __name__ == "__main__":
    main()
