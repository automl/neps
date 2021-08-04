import subprocess
import sys


def install_torch():
    cuda_version = input("Please choose a cuda version e.g., cpu | cu101 | cu111]\n")
    torch_version = input("Please choose a pytorch version, e.g., 1.8.1 | 1.9.0\n")

    install_command = f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-{torch_version}+{cuda_version}.html"
    install_command = [sys.executable, "-m"] + install_command.split()
    subprocess.check_call(install_command)


if __name__ == "__main__":
    install_torch()
