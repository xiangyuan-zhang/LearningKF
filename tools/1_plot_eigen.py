# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml
from matplotlib import pyplot as plt


def plot_eigen(PARAMS, A):
    """
    Plot the eigen spectrum of A matrix

    Args:
        A: complex 2D array
    """
    fig1 = plt.figure(figsize=[4, 4])
    ax1 = fig1.add_subplot(111)
    circle1 = plt.Circle((0, 0), 1, fc="white", ec="black")

    # Eliminate axes
    ax1.spines["right"].set_color("none")
    ax1.spines["top"].set_color("none")
    ax1.spines["left"].set_color("none")
    ax1.spines["bottom"].set_color("none")

    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.add_patch(circle1)
    X, Y = np.real(np.linalg.eig(A)[0]), np.imag(np.linalg.eig(A)[0])
    ax1.axvline(x=0, color="k", alpha=0.3)
    ax1.axhline(y=0, color="k", alpha=0.3)
    ax1.scatter(X, Y, color="tab:orange", marker="*", s=5)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.5, 0.5)

    # save the figure to test/eigen.png
    plt.savefig(PARAMS.log_dir + "eigen.png", dpi=300)
    plt.close(fig1)


if __name__ == "__main__":
    # Read the parameters of PDE from the yaml file in the test folder
    test_dir = "save/test/"
    with open(test_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))
    A = sio.loadmat(PARAMS.log_dir + "model.mat")["A"]
    plot_eigen(PARAMS, A)
