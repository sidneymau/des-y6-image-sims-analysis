import os
import concurrent.futures

import fitsio
from matplotlib import cm
import numpy as np
from PIL import Image
import PIL.ImageOps


MASK_DIR = "/pscratch/sd/s/smau/fiducial_masks"
OUT_DIR = "/pscratch/sd/s/smau/Y6A1_COADD_MASKS"


def _main(tilename):
    _mask = os.path.join(
        MASK_DIR,
        f"{tilename}_metadetect-config_mdetcat_part0000-mask.fits.fz",
    )
    outfile = os.path.join(
        OUT_DIR,
        f"{tilename}-mask.png",
    )

    with fitsio.FITS(_mask) as fits:
        mask = fits[1].read()

    bitmask = (mask > 1)
    with Image.fromarray(bitmask) as im:
        im = PIL.ImageOps.flip(im)
        im = im.convert("L")
        im = PIL.ImageOps.invert(im)
        im.save(outfile)


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
