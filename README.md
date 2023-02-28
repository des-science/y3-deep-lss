# y3-deep-lss
This repo contains the codes for the DES Y3 data analysis with deep learning using the forward model from https://github.com/des-science/multiprobe-simulation-forward-model

### Setting up the conda environment on Perlmutter
```
module load python
conda create --prefix /global/common/software/des/athomsen/deep_lss python=3.9
conda activate /global/common/software/des/athomsen/deep_lss
module load tensorflow/2.9.0

# deepsphere
pip install scipy==1.8
conda install healpy
python -m pip install 'PyGSP @ git+https://github.com/jafluri/pygsp.git@sphere-graphs'
python -m pip install 'deepsphere @ git+https://github.com/deepsphere/deepsphere-cosmo-tf2.git'
```
And installing [`multiprobe-simulation-forward-model`](https://github.com/des-science/multiprobe-simulation-forward-model) from source.
