# data

- v0: original test files based on 400-tile sample and using y6 transfer function
    - jk: jackknife resampling
    - bs: bootstrap resampling
- v1: 1000-tile sample and using sim transfer functions; with sompz n(z)s (data response grid)
- v2: 1000-tile sample and using sim transfer functions; with reocmputed n(z)s (sim response grid)
    - statistical: weights solely from statistical shear weight
    - statistical-neighbor: weights are product of statistical shear weight and neighbor weight
