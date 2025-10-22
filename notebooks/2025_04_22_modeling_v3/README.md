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
