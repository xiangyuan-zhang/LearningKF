# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml


def check_obsv(A, C):
    """
    Test the observability of (A, C)
    Args:
        A, C: 2D array, representing linear system dynamics and weightings
    Returns:
        Rank of the observability matrix
    """
    obsv = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(0, A.shape[0])])
    return np.linalg.matrix_rank(obsv)


if __name__ == "__main__":
    test_dir = "save/test/"
    with open(test_dir + "PARAMS.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))
    model = sio.loadmat(PARAMS.log_dir + "model.mat")
    A, C = model["A"], model["C"]
    # Sanity check: plot the eigen spectrum of A matrix and check the observability
    obsv_rank = check_obsv(A, C)
    print("Rank of the observability matrix is %d" % (obsv_rank))
