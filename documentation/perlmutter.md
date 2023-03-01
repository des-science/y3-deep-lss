### Setting up the `conda` environment
- Locate the environment in `/global/common/software` instead of `$HOME`, see [here](https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware)
- Clone their conda environment containing tensorflow, see [here](https://docs.nersc.gov/development/languages/python/nersc-python/#using-conda-clone)

```
# load conda
module load python

# make the environments discoverable
conda config --append envs_dirs /global/common/software/des/athomsen

# get the location of the tensorflow environment by running `module show tensorflow`
conda create --prefix /global/common/software/des/athomsen/deep_lss --clone /global/common/software/nersc/pm-2022q4/sw/tensorflow/2.9.0

# install the jupyter kernel 
python -m ipykernel install --user --name deep_lss

# install packages and modify their version
conda install healpy
pip install numpy==1.23.5
pip install tensorflow_probability==0.17.0

# TODO these don't work for some reason, clone the repo and `pip install -e .` instead
python -m pip install --user 'PyGSP @ git+https://github.com/jafluri/pygsp.git@sphere-graphs'
python -m pip install --user 'deepsphere @ git+https://github.com/deepsphere/deepsphere-cosmo-tf2.git'
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
exec "$@
```

