# PyTorch
Using a `uenv` following https://docs.cscs.ch/software/ml/pytorch/#running-pytorch-with-a-uenv.

## set up `uenv`
```
uenv image find

uenv image pull pytorch/v2.9.1:v2
uenv start --view=default pytorch/v2.9.1:v2

# https://docs.cscs.ch/build-install/python/#installing-venv-on-top-of-a-uenv-view
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
```

## set up virtual environment 
```
# uv setup
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"

# create and activate virtual environment
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy ~/dlss/torch_env
source ~/dlss/torch_env/bin/activate

# install from repos
uv pip install -e ~/dlss/repos/multiprobe-simulation-forward-model
uv pip install -e ~/dlss/repos/y3-deep-lss
uv pip install -e ~/dlss/repos/multiprobe-simulation-inference --override <(echo "pandas>=2.1.0")

# remove pypi version to use the ones from the uenv
uv pip uninstall torch sympy networkx mpmath

# for compatibility with sbi package
uv pip install "arviz<1"

# test GPUs
python -c "import torch; print(torch.cuda.device_count())"
```

## set up Jupyter kernel
```
uv pip install ipykernel
python -m ipykernel install ${VIRTUAL_ENV:+--env PATH $PATH --env VIRTUAL_ENV $VIRTUAL_ENV} --user --name="torch_env"
```
To launch JupyterLab, specify `pytorch/v2.9.1:v2` in "Custom uenv".

### stable kernel via standalone Jupyter server
VSCode often "forgets" the kernel over a tunnel session (fixed by reloading the window).
To make this more robust, run a standalone Jupyter server that outlives any
extension/tunnel hiccups, and connect VSCode to it as an "Existing Jupyter Server"
instead of letting VSCode launch the kernel itself:
```
# in the srun session, with torch_env active (e.g. inside tmux so it survives)
jupyter server --no-browser --port=8888 --ServerApp.token='' --ServerApp.ip=127.0.0.1
```
In VSCode: notebook kernel picker → "Select Another Kernel" → "Existing Jupyter Server" →
`http://localhost:8888`. Since `code tunnel` runs on the same compute node, this
resolves directly. After a "Reload Window", just reconnect to the same server/kernel
instead of relaunching one.

## VScode tunnel
### setup
```
# compute node
srun --uenv=pytorch/v2.9.1:v2 --view=default -A a0158 -t 00:10:00 -n 1 --pty bash

export VSCODE_CLI_DATA_DIR=$HOME/.vscode/cli
export VSCODE_CLI_USE_FILE_KEYCHAIN=1
export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=1
export HOSTNAME=clariden-fixed

code tunnel user login --provider github
```
### usage
```
# compute node
srun --uenv=pytorch/v2.9.1:v2 --view=default -A a0158 -t 00:10:00 -n 1 --pty code tunnel --name=$CLUSTER_NAME-tunnel

# login node
uenv run --view=default pytorch/v2.9.1:v2 -- code tunnel --name=$CLUSTER_NAME-tunnel
```

## activate environment session
```
# login node
uenv start --view=default pytorch/v2.9.1:v2
source ~/dlss/torch_env/bin/activate
```

## Tensorboard
```
uenv start --view=default pytorch/v2.9.1:v2
source ~/dlss/torch_env/bin/activate

tensorboard --logdir /users/athomsen/scratch/deep_lss/runs/v16/rot_in_place/maps --bind_all --port 6007
```