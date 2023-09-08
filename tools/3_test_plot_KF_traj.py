# SPDX-License-Identifier: MIT

import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml

sys.path.append(os.getcwd())
from utils import evolve_kf, finite_kf, infinite_kf, pde_setup, visualize_kf

if __name__ == "__main__":
    log_dir = "save/test/"
    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # Setup the PDE variables
    # dx: spatial discretization size; x: spatial coordinate; z0: initial condition
    dx, x, _, z0, _ = pde_setup(PARAMS)

    # Constuct the linear dynamical system and specify noise statistics
    model = sio.loadmat(log_dir + "model.mat")
    A, C = model["A"], model["C"]
    mu_x0_noise = np.array([np.sin(2 * np.pi * x / PARAMS.Lx)]).T
    X0 = (PARAMS.var_x**2) * mu_x0_noise @ mu_x0_noise.T
    W = PARAMS.var_w * np.identity(PARAMS.n_x)
    V = PARAMS.var_v * np.identity(PARAMS.n_y)

    # Compute the infinite-horizon and finite-horizon Kalman filter
    S, L, AL = infinite_kf(A, C, W, V)
    L_f, AL_f = finite_kf(PARAMS, A, C, W, V, X0, PARAMS.n_save)

    # Visualize the estimated trajectory using infinite-horizon KF policy
    # where the initial state of the system starts from z0, the mean of the distribution
    evolve_kf(PARAMS, L, AL, "Inf_KF", "Inf_Horizon_KF")
    visualize_kf(PARAMS, "Inf_Horizon_KF")

    evolve_kf(PARAMS, L_f, AL_f, "Finite_KF", "Finite_Horizon_KF")
    visualize_kf(PARAMS, "Finite_Horizon_KF")
