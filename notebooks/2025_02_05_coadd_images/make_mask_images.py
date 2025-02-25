import os
import concurrent.futures

import fitsio
import numpy as np
from PIL import Image, ImageOps


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

    # bitmask = (mask > 1)
    # with Image.fromarray(bitmask) as im:
    #     im = ImageOps.flip(im)
    #     im = im.convert("L")
    #     im = ImageOps.invert(im)
    #     im.save(outfile)

    with Image.fromarray((mask & (~2**2)) != 1) as _im_all:
        _im_all = ImageOps.invert(_im_all)
        # im_all = im_all.convert("L")
        _im_all = ImageOps.flip(_im_all)
        
    with Image.fromarray(~(mask & (2**2) == 0)) as _im_gaia:
        _im_gaia = ImageOps.invert(_im_gaia)
        # im_gaia = im_gaia.convert("L")
        _im_gaia = ImageOps.flip(_im_gaia)
        
    with Image.fromarray(~(mask & (2**2) == 0)) as im_alpha:
        # im_alpha = im_alpha.convert("L")
        im_alpha = ImageOps.flip(im_alpha)

    # this appears backwards -- this is because we are working in an inverted
    # color space with the masks
    im_all = _im_all.convert("LA")
    im_gaia = _im_gaia.convert("LA")
    im_all.putalpha(255 // 2)
    im_gaia.putalpha(im_alpha)
    
    im = Image.alpha_composite(
        im_gaia.convert("RGBA"),
        im_all.convert("RGBA"),
    ).convert("L")

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
