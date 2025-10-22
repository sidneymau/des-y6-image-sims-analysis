## code env

Use the environment.yaml file and then install orthax via pip

```bash
conda env create --file environment.yaml
conda activate des-y6-imsim-analysis
pip install --no-deps --no-build-isolation orthax==0.2.3
pushd ../..
pip install --no-deps --no-build-isolation -e .
popd
```

## the final priors were pulled from the files

- `des_y6_nz_SOMPZ_imsim_v3.1_nonneg.h5`
- `des_y6_nz_SOMPZ_WZ_imsim_v{VERSION}_nonneg.h5`
