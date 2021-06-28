#!/bin/bash
set -e  # Exit on first failure

wget https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -O get-poetry.py
python get-poetry.py
rm get-poetry.py

echo 'Append to your .zshrc / .bashrc or run: export PATH="$HOME/.poetry/bin:$PATH"'
