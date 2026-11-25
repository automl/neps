"""Fits the scaling law on the finished trials and predicts the loss at larger budgets."""

from generate_configs import ROOT_DIRECTORY

import scalefm

TARGET_FLOPS = [1e18, 1e19]

if __name__ == "__main__":
    scalefm.predict(ROOT_DIRECTORY, TARGET_FLOPS)