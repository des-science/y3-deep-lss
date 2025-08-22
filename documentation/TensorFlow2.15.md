# setup
```
module load python/3.9
module show tensorflow/2.15.0
conda create --prefix /global/common/software/des/athomsen/dlss15 --clone /global/common/software/nersc9/tensorflow/2.15.0

python -m ipykernel install --user --name dlss15

pip install --force-reinstall --no-cache-dir healpy
pip install --force-reinstall --no-cache-dir tensorflow_probability==0.23 
pip install --force-reinstall --no-cache-dir icecream
pip install --force-reinstall --no-cache-dir scipy==1.8
# install the local repos msfm, deep_lss, etc.
```

# .bash_profile
```
module load cpe/24.07
module load python/3.9
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load nccl/2.18.3-cu12
conda activate dlss15
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
```
