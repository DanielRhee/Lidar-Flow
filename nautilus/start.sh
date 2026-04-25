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

conda activate lidarflow

sudo apt update
sudo apt-get install screen
