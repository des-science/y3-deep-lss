### Setting up the `conda` environment
- Locate the environment in `/global/common/software` instead of `$HOME`, see [here](https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware)
- Clone their conda environment containing tensorflow, see [here](https://docs.nersc.gov/development/languages/python/nersc-python/#using-conda-clone)
- Follow the best practices for using `pip` within a `conda` environment listed [here](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
  - First use conda, then only pip (don't go back and forth)
  - Don't use the `--user` flag in `pip install`
  - Use the `--upgrade-strategy only-if-needed` flag in `pip install`, but this is not necessary as it's the default
- The cluster support recommends to use the `--force-reinstall` and `--no-cache-dir` flags in `pip install`, see [here](https://docs.nersc.gov/development/languages/python/nersc-python/#installing-libraries-via-pip)

```
# load conda
module load python

# make the environments discoverable
conda config --append envs_dirs /global/common/software/des/athomsen

# you can get the location of the tensorflow environment by running `module show tensorflow`

# clone the NERSC provided tensorflow conda environment
conda create --prefix /global/common/software/des/athomsen/deep_lss --clone /global/common/software/nersc/pm-2022q4/sw/tensorflow/2.9.0

# install the jupyter kernel 
python -m ipykernel install --user --name deep_lss

# install packages, from now on try to only use pip (not conda)
pip install --force-reinstall --no-cache-dir scipy==1.8
pip install --force-reinstall --no-cache-dir healpy
pip install --force-reinstall --no-cache-dir tensorflow_probability==0.17.0
pip install --force-reinstall --no-cache-dir icecream

# TODO these don't work for some reason, clone the repo and `pip install -e .` instead
python -m pip install --force-reinstall --no-cache-dir 'PyGSP @ git+https://github.com/jafluri/pygsp.git@sphere-graphs'
python -m pip install --force-reinstall --no-cache-dir 'deepsphere @ git+https://github.com/deepsphere/deepsphere-cosmo-tf2.git'
```

### Setting up the `.bash_profile`
- Load the `CUDA` related modules separately, see [here](https://docs.nersc.gov/machinelearning/tensorflow/#customizing-environments). Without this, `tensorflow` does not recognize the GPUs.

```
module load python
module load cudatoolkit
module load cudnn
conda activate deep_lss
```

### Setting up the `jupyter` kernel
- Create a helper shell script to load the necessary modules before starting the kernel, see [here](https://docs.nersc.gov/services/jupyter/#customizing-kernels-with-a-helper-shell-script)
- The `kernel.json` file is located at `/global/u2/a/athomsen/.local/share/jupyter/kernels/deep_lss` and looks like 
```
{
 "argv": [
  "{resource_dir}/kernel-helper.sh",
  "/global/common/software/des/athomsen/deep_lss/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "deep_lss",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```
- `kernel-helper.sh` is located within the same directory and looks like 
```
#! /bin/bash
module load cudatoolkit
module load cudnn
exec "$@"
```
- `kernel-helper.sh` has to be made executable with `chmod u+x kernel-helper.sh`

