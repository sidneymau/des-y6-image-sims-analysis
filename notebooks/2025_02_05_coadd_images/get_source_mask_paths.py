import os


# OAK path:
OAK_PATH = "/oak/stanford/orgs/kipac/users/smau/y6-image-sims/fiducial/"
SHEAR_STEP = "g1_slice=0.02__g2_slice=0.00__g1_other=0.00__g2_other=0.00__zlow=0.0__zhigh=6.0"


def main():

    args = {}
    bands = ["g", "r", "i"]

    # we only have images for tiles 1--525
    with open("args-y6.txt") as args_file:
        for i, line in enumerate(args_file):
            if i < 525:
                _args = line.split(" ")
                _tile = _args[0]
                _seed = _args[1]
                args[_tile] = _seed
            else:
                break

    coadd_paths = []

    for tile, seed in args.items():
        for band in bands:
            _coadd_path = os.path.join(
                OAK_PATH,
                tile,
                seed,
                SHEAR_STEP,
                "des-pizza-slices-y6",
                tile,
                "metadetect",
                f"{tile}_metadetect-config_mdetcat_part0000-mask.fits.fz",
            )
            coadd_paths.append(_coadd_path)

    with open("source_mask_paths.txt", "w") as out_file:
        for coadd_path in coadd_paths:
            out_file.write(coadd_path)
            out_file.write("\n")


if __name__ == "__main__":
    main()

