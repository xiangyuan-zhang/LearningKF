# SPDX-License-Identifier: MIT

import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml
from matplotlib import colors
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from utils import pde_setup, truncate_colormap


def plot_real_traj(PARAMS, A, W, z0):
    """
    Plot the PDE trajectory starting from the mean of random initial state.

    Args:
        PARAMS: yaml files keeping the PDE parameters
        A, W: 2D arrays, linear system dynamics
        z0: 1D array, initial state

    Returns:
        None. A 3D plot will be saved at test/real_traj/real_traj.png

    """

    # Evolve the real trajectory forward
    rng = np.random.default_rng(seed=PARAMS.seed)  # set the random seed
    zz_ref = np.zeros((PARAMS.n_x, PARAMS.n_save))
    zz_ref[:, 0] = z0
    for step_num in range(PARAMS.n_save - 1):
        w_t = rng.multivariate_normal(np.zeros(A.shape[0]), W, size=(1,)).T
        zz_ref[:, step_num + 1] = A @ zz_ref[:, step_num] + w_t.flatten()

    # save the real trajectory
    sio.savemat(PARAMS.log_dir + "real_traj.mat", {"zz_ref": zz_ref})

    # Plot the real trajectory
    dx = PARAMS.Lx / PARAMS.n_x
    x = np.linspace(0, PARAMS.Lx - dx, PARAMS.n_x)
    tt_save = np.linspace(0, (PARAMS.n_save - 1) * PARAMS.dt_save, PARAMS.n_save)

    norm_center = colors.TwoSlopeNorm(
        vmin=0, vcenter=zz_ref.max() / 2, vmax=zz_ref.max()
    )
    colormap = truncate_colormap(plt.get_cmap("jet"), 0, 1)
    a, b = np.meshgrid(range(len(tt_save)), range(len(x)))
    fig = plt.figure(figsize=(10, 8))
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    ax = fig.add_subplot(111, projection="3d")
    cf1 = ax.plot_surface(
        a,
        b,
        zz_ref,
        cmap=colormap,
        norm=norm_center,
        rcount=PARAMS.n_x,
        ccount=PARAMS.n_save,
    )
    _ = fig.colorbar(
        cf1, format="%.1f", orientation="horizontal", shrink=0.45, pad=0.03, ax=ax
    )
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_zlim(0, 1)

    # Save the figure to test/real_traj.png
    plt.savefig(PARAMS.log_dir + "real_traj.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    log_dir = "save/test/"
    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # Setup the PDE variables
    # dx: spatial discretization size; x: spatial coordinate; z0: initial condition
    _, _, _, z0, _ = pde_setup(PARAMS)

    # Constuct the linear dynamical system and specify noise statistics
    A = sio.loadmat(log_dir + "model.mat")["A"]
    W = PARAMS.var_w * np.identity(PARAMS.n_x)

    # Visualize the system evolution as a 3D plot, which is saved in save/test/real_traj
    plot_real_traj(PARAMS, A, W, z0)
