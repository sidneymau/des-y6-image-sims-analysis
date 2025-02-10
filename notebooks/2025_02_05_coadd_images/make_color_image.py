# from https://github.com/beckermr/des-y6-analysis/blob/main/2022_09_28_color_images/make_color_image.py
import subprocess
import concurrent.futures
import glob
import os
import sys
import fitsio
import numpy as np
from pathlib import Path


MEDS_DIR = "/pscratch/sd/s/smau/fiducial_MEDS/"



def _main(tilename):
    bands = ["g", "r", "i"]
    coadds = {}

    for band in bands:
        fname_coadd = os.path.join(
            MEDS_DIR,
            tilename,
            f"{tilename}_{band}_des-pizza-slices-y6-v15_meds-pizza-slices.fits.fz",
        )

        assert os.path.isfile(fname_coadd)
        coadds[band] = fname_coadd

    # coadd_path = "/pscratch/sd/s/smau/coadds"
    image_path = "/pscratch/sd/s/smau/Y6A1_COADD_fiducial"

    # os.makedirs(coadd_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    for band, fname_coadd in coadds.items():
        subprocess.run(
            [
                f"make-coadd-image-from-slices",
                f"{fname_coadd}",
                f"--output-path={image_path}/{tilename}-{band}.fits.fz",
            ],
            check=True,
        )

    output_path = f"{image_path}/{tilename}-coadd-gri.jpg"
    # output_path_crop = f"{image_path}/{tilename}-coadd-gri-crop.jpg"

    subprocess.run(
        [
            f"des-make-image-fromfiles",
            output_path,
            f"{image_path}/{tilename}-g.fits.fz",
            f"{image_path}/{tilename}-r.fits.fz",
            f"{image_path}/{tilename}-i.fits.fz",
        ],
        check=True,
    )
    os.remove(f"{image_path}/{tilename}-g.fits.fz")
    os.remove(f"{image_path}/{tilename}-r.fits.fz")
    os.remove(f"{image_path}/{tilename}-i.fits.fz")

    # subprocess.run(
    #     [
    #         f"magick",
    #         output_path,
    #         f"-crop",
    #         f"1000x1000+4500+4500",
    #         output_path_crop,
    #     ],
    #     check=True,
    # )


def main():

    futures = []

    with open("args-y6.txt", "r") as fp:
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            for i, line in enumerate(fp):
                if i < 525:
                    tilename = line.strip(" ")[0]
                    _future = executor.submit(_main, tilename)
                    futures.append(_future)
                else:
                    break

    for future in futures:
        print(".", end="")


if __name__ == "__main__":
    main()

