# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml
from matplotlib import pyplot as plt


def plot_test_kf(PARAMS, cost_mat, flag_list):
    """
    Generate the plot of 100 random trails

    Args:
        PARAMS: dictinary files keeping the PDE parameters
        cost_mat: the L2 norm between real traj and estimated traj
        flag_list: the list of ids that have been tested (6 of them)

    Returns:
        None. A figure will be saved to test/kf_test.png

    """
    color_list = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_subplot(111)
    for i in range(len(flag_list)):
        cost_i = cost_mat[:, i, :]
        mean_i = cost_i.mean(axis=0)
        std_i = cost_i.std(axis=0)

        # plot the mean
        ax1.plot(np.arange(PARAMS.n_save), mean_i, color=color_list[i], linewidth=2.5)

        # plot the shaded region with boundaries being mean+std and mean-std
        ax1.fill_between(
            np.arange(PARAMS.n_save),
            (mean_i - std_i),
            (mean_i + std_i),
            alpha=0.15,
            label="_nolegend_",
            color=color_list[i],
        )

    ax1.set_xlim(0, PARAMS.n_save)
    ax1.set_ylim(0, 3)
    for axis in ["top", "right"]:
        ax1.spines[axis].set_linewidth(0)
    for axis in ["bottom", "left"]:
        ax1.spines[axis].set_linewidth(2.5)
    plt.yticks([0, 1, 2, 3], fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(
        [r"$N=1$", r"$N=6$", r"$N=11$", r"$N=51$", r"$N=101$", r"KF($N=\infty$)"],
        fontsize=20,
        ncol=3,
    )

    # Save the figure to test/kf_test.png
    plt.savefig(PARAMS.log_dir + "kf_test.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    log_dir = "save/test/"
    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # plot the L2 norm of the estimation error against time
    # the plot will be saved to save/kf_test.png
    cost_mat = sio.loadmat(PARAMS.log_dir + "kf_test.mat")["cost_mat"]
    flag_list = sio.loadmat(PARAMS.log_dir + "kf_test.mat")["flag_list"]
    plot_test_kf(PARAMS, cost_mat, flag_list)
