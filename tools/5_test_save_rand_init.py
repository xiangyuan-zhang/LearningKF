# SPDX-License-Identifier: MIT

import os
import sys
from tqdm import tqdm
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml

sys.path.append(os.getcwd())
from utils import pde_setup


def test_kf(PARAMS, flag_list, X0, num):
    """
    Test several KF policy with (num) of trails starting from
    different initial state

    Args:
        PARAMS: dictinary files keeping the PDE parameters
        flag_list: list of ids to be loaded to test
        A, C, W, V, X0, z0: linear system dynamics
        num: number of samples for the experiment

    Returns:
        None. A cost matrix will be saved to test/kf_test.mat

    """
    _, _, _, z0, _ = pde_setup(PARAMS)

    # Constuct the linear dynamical system and specify noise statistics
    model = sio.loadmat(PARAMS.log_dir + "model.mat")
    A, C = model["A"], model["C"]
    W = PARAMS.var_w * np.identity(PARAMS.n_x)
    V = PARAMS.var_v * np.identity(PARAMS.n_y)
    rng = np.random.default_rng(seed=PARAMS.seed)

    # Initialize the cost matrix
    cost_mat = np.zeros((num, len(flag_list), PARAMS.n_save))
    AL_mat = np.zeros((len(flag_list), PARAMS.n_x, PARAMS.n_x))
    L_mat = np.zeros((len(flag_list), PARAMS.n_x, PARAMS.n_y))

    # load all the policies from different folders to test
    for i in range(len(flag_list)):
        policy_i = sio.loadmat(PARAMS.log_dir + flag_list[i] + "/policy.mat")
        AL_mat[i], L_mat[i] = policy_i["AL"], policy_i["L"]

    # Loop over the number of samples
    for i in tqdm(range(num)):
        # Sample a random initial state to start the trajectory
        x0_noise = rng.multivariate_normal(np.zeros(PARAMS.n_x), X0, size=(1,)).T
        x_0 = np.array([z0]).T + x0_noise
        est_x_0 = np.array([z0]).T

        # Initialize matrices to record both the real trajectory and the estimated trajectory
        x_mat_i = np.zeros((len(flag_list), PARAMS.n_save, PARAMS.n_x))
        est_x_mat_i = np.zeros((len(flag_list), PARAMS.n_save, PARAMS.n_x))

        # Loop over all the policies to test
        for j in range(len(flag_list)):
            # Compute the L2 norm of the state difference
            x_mat_i[j, 0] = x_0.flatten()
            est_x_mat_i[j, 0] = est_x_0.flatten()
            cost_mat[i, j, 0] = np.linalg.norm(x_mat_i[j, 0] - est_x_mat_i[j, 0])
            AL, L = AL_mat[j], L_mat[j]

            # Loop over the time horizon
            for k in range(PARAMS.n_save - 1):
                # Sample random system and measurement noises and evolve
                # both the system dyanmics and the estimator dynamics
                w_k = rng.multivariate_normal(np.zeros(PARAMS.n_x), W)
                v_k = rng.multivariate_normal(np.zeros(PARAMS.n_y), V)
                x_mat_i[j, k + 1] = A @ x_mat_i[j, k] + w_k
                y_k = C @ x_mat_i[j, k] + v_k
                est_x_mat_i[j, k + 1] = AL @ est_x_mat_i[j, k] + L @ y_k
                cost_mat[i, j, k + 1] = np.linalg.norm(
                    x_mat_i[j, k + 1] - est_x_mat_i[j, k + 1]
                )

    # save cost_mat to the log directory specified in PARAMS
    sio.savemat(
        PARAMS.log_dir + "kf_test.mat", {"cost_mat": cost_mat, "flag_list": flag_list}
    )


if __name__ == "__main__":
    log_dir = "save/test/"
    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # Check that N is large enough
    with open(log_dir + "PARAMS_RHPG.yaml", "r") as file:
        PARAMS_RHPG = SimpleNamespace(**yaml.safe_load(file))
    if PARAMS_RHPG.N < 102:
        raise ValueError("N needs to be at least 102 for this script to run.")

    # select a few representative case and sample 100 random initial states
    # (takes ~15 mins on Intel i9-13900K)
    flag_list = [
        "RHPG_(N=1)",
        "RHPG_(N=6)",
        "RHPG_(N=11)",
        "RHPG_(N=51)",
        "RHPG_(N=101)",
        "Inf_Horizon_KF",
    ]
    num_sample = 100

    dx = PARAMS.Lx / PARAMS.n_x
    x = np.linspace(0, PARAMS.Lx - dx, PARAMS.n_x)
    mu_x0_noise = np.array([np.sin(2 * np.pi * x / PARAMS.Lx)]).T
    X0 = (PARAMS.var_x**2) * mu_x0_noise @ mu_x0_noise.T

    test_kf(PARAMS, flag_list, X0, num_sample)
