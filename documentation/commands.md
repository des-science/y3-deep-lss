- back up the `run_files`
  - to laptop:
    ```
    rsync -ahv --prune-empty-dirs \
    --include={"*/","*.yaml","*.h5","*.npy","*.pt"} \
    --exclude={"*","debug","wandb/","wandb/**"} \
    athomsen@perlmutter-p1.nersc.gov:/pscratch/sd/a/athomsen/run_files/v6 \
    /Users/arne/data/DESY3/models
    ```
  - Perlmutter internally:
    ```
    rsync -ahv --prune-empty-dirs \
    --include={"*/","*.yaml","*.h5","*.npy","*.pt"} \
    --exclude={"*","debug","wandb/","wandb/**"} \
    /pscratch/sd/a/athomsen/run_files/v6 \
    /global/cfs/cdirs/des/athomsen/deep_lss/run_files
    ```
