# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import yaml

from algorithm import rhpg_adam
from utils import pde_setup

if __name__ == "__main__":
    # Read the PDE parameter
    log_dir = "save/test/"

    with open(log_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))
    with open("config/PARAMS_RHPG.yaml", "r") as file:
        PARAMS_RHPG = SimpleNamespace(**yaml.safe_load(file))

    # Check that N is large enough for the scripts in tools/
    if PARAMS_RHPG.N < 102:
        raise ValueError("N needs to be at least 102 for the scripts in tools/ to run.")

    # Setup the PDE variables
    # dx: spatial discretization size; x: spatial coordinate; z0: initial condition
    dx, x, _, z0, _ = pde_setup(PARAMS)

    # Specify noise statistics
    mu_x0_noise = np.array([np.sin(2 * np.pi * x / PARAMS.Lx)]).T
    X0 = (PARAMS.var_x**2) * mu_x0_noise @ mu_x0_noise.T

    # Run RHPG to optimize for the KF policies (takes ~11 mins on Intel i9-13900K)
    rhpg_adam(PARAMS, X0, PARAMS_RHPG.N, z0, PARAMS_RHPG.lr)

    # Record the PARAMS file for the test
    with open(PARAMS.log_dir + "PARAMS_RHPG.yaml", "w") as file:
        yaml.dump(vars(PARAMS_RHPG), file, sort_keys=False)
