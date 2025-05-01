# data

- v0: original test files based on 400-tile sample and using y6 transfer function
    - jk: jackknife resampling
    - bs: bootstrap resampling
- v1: 1000-tile sample and using sim transfer functions; with sompz n(z)s (data response grid)
- v2: 1000-tile sample and using sim transfer functions; with reocmputed n(z)s (sim response grid)
    - statistical: weights solely from statistical shear weight
    - statistical-neighbor: weights are product of statistical shear weight and neighbor weight
    - statistical-occupancy: weights are product of statistical shear weight and occupancy ratio weight
    - statistical-neighbor-occupancy: weights are product of statistical shear weight, neighbor weight, and occupancy ratio weight
    - statistical-nz: weights are product of statistical shear weight and nz-match weight
    - statistical-neighbor-nz: weights are product of statistical shear weight, neighbor weight, and nz-match weight
- v3: same as v2 but using averaged diagonal of response
    - y3: y3-like selection
    - y3-complement: complement of y3-like selection; treated as one tomographic bin (index `-1`)
    - i_X_Y: i-band magnitude limited to X < i <= Y; no tomographic binning applied
