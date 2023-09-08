# SPDX-License-Identifier: MIT

import os
import sys
from types import SimpleNamespace

import scipy.io as sio
import yaml

sys.path.append(os.getcwd())
from utils import evolve_kf, visualize_kf

if __name__ == "__main__":
    log_dir = "save/test/"
    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # Check that N is large enough
    with open(log_dir + "PARAMS_RHPG.yaml", "r") as file:
        PARAMS_RHPG = SimpleNamespace(**yaml.safe_load(file))
    if PARAMS_RHPG.N < 102:
        raise ValueError("N needs to be at least 102 for this script to run.")

    # Load the convergent policies from the save directory
    p_list = sio.loadmat(log_dir + "rhpg_policy.mat")["policy"]

    # Visualize the estimated trajectory of several frozen KF policies
    # where the initial state starts from the mean of the random distribution
    # the visualizations are saved into folders save/test/flag, where flag is
    # the id of the frozen policies
    for idx in [0, 1, 2, 5, 10, 20, 30, 50, 100]:
        evolve_kf(
            PARAMS,
            p_list[idx, :, PARAMS.n_x :],
            p_list[idx, :, : PARAMS.n_x],
            "Inf_KF",
            "RHPG_(N=" + str(idx + 1) + ")",
        )
        visualize_kf(PARAMS, "RHPG_(N=" + str(idx + 1) + ")")
