import subprocess
import sys


def install_torch():
    torch_version = input("Please choose a PyTorch version, e.g., 1.8.1 | 1.9.0\n")
    assert torch_version in [
        "1.8.1",
        "1.9.0",
    ], "We only support PyTorch versions 1.8.1 or 1.9.0 for now!"
    cuda_version = input("Please choose a cuda version e.g., cpu | cu101 | cu111]\n")
    assert cuda_version in [
        "cpu",
        "cu101",
        "cu111",
    ], "We only support cpu, cu101 or cu111 for now!"

    install_command = f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-{torch_version}+{cuda_version}.html"
    install_command = [sys.executable, "-m"] + install_command.split()
    subprocess.check_call(install_command)


if __name__ == "__main__":
    install_torch()
