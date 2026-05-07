#!/bin/bash

CONDA_DIR="$HOME/miniconda3"
INSTALLER="$HOME/persistent/python/miniconda_installer.sh"
ENV_FILE="$HOME/persistent/python/lidarflow.yml"

bash "$INSTALLER" -b -p "$CONDA_DIR"
source "$CONDA_DIR/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f "$ENV_FILE" 2>/dev/null || conda env update -f "$ENV_FILE"
conda install -n base -c conda-forge screen -y

cat >> "$HOME/.bashrc" << 'BASHRC'
source "$HOME/miniconda3/etc/profile.d/conda.sh"
BASHRC

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if   [ "$CC" -ge 80 ]; then TORCH_IDX="cu124"; SPCONV="spconv-cu124==2.3.8"
else                        TORCH_IDX="cu118"; SPCONV="spconv-cu118==2.3.4"
fi

conda run -n lidarflow python -m pip install torch --index-url https://download.pytorch.org/whl/${TORCH_IDX}
conda run -n lidarflow python -m pip install av2
conda run -n lidarflow python -m pip install "${SPCONV}"

conda activate lidarflow

#sudo apt update
#sudo apt-get install screen
