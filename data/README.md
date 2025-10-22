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

## how to make the sompz-only Tz data

The file `combined_Tz_samples_y6_RU_ZPU_LHC_1e8_stdRUmethod_unblind_oldbinning_Nov5.npy` was made via
the following python code:

```python
import numpy as np
import h5py


fp = h5py.File("/global/cfs/cdirs/des/boyan/sompz_output/y6_data_10000Tile_final_unblind/nz_realizations/combined_Tz_samples_y6_RU_ZPU_LHC_1e8_stdRUmethod_unblind_oldbinning_Nov5.h5")

rng = np.random.RandomState(seed=42424242)
inds = np.sort(rng.choice(fp["bin0"].shape[0], size=10095, replace=False))

for i in range(4):
    final_data[:, i, :] = fp["bin%d" % i][inds, :]

np.save("combined_Tz_samples_y6_RU_ZPU_LHC_1e8_stdRUmethod_unblind_oldbinning_Nov5.npy", final_data, allow_pickle=False)
```
