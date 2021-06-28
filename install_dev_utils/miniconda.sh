#!/bin/bash
set -e  # Exit on first failure

INSTALL_DIR=$HOME/.conda

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $INSTALL_DIR
rm install_miniconda.sh

echo 'Run "~/.conda/bin/conda init" or "~/.conda/bin/conda init zsh" and append "export CONDA_AUTO_ACTIVATE_BASE=false" to your .bashrc / .zshrc'
