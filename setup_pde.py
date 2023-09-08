# SPDX-License-Identifier: MIT

import os
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import yaml

from utils import compute_C, ft_matrix, ift_matrix

if __name__ == "__main__":
    # Read the parameters of PDE from the yaml file
    with open("config/PARAMS_HEAT.yaml", "r") as file:
        PARAMS = SimpleNamespace(**yaml.safe_load(file))

    # Create a folder to save data using the log_dir specified in PARAMS
    os.makedirs(PARAMS.log_dir, exist_ok=True)

    # Record the PARAMS file for the test
    with open(PARAMS.log_dir + "PARAMS.yaml", "w") as file:
        yaml.dump(vars(PARAMS), file, sort_keys=False)

    # Compute the analytical A matrix
    DFT = ft_matrix(PARAMS.n_x)  # discrete fourier transform matrix
    IDFT = ift_matrix(PARAMS.n_x)  # inverse fourier transform matrix
    kx_complex = (
        np.fft.fftfreq(PARAMS.n_x, PARAMS.Lx / PARAMS.n_x) * 2 * np.pi
    )  # wavenumbers
    A = np.real(
        IDFT
        @ np.diag(
            np.exp(
                (-PARAMS.velocity * 1j * kx_complex - PARAMS.nu * kx_complex**2)
                * PARAMS.dt_save
            )
        )
        @ DFT
    )

    # generate C matrix with n_y number of sensors
    spa_coor = np.linspace(0, PARAMS.Lx - (PARAMS.Lx / PARAMS.n_x), PARAMS.n_x)
    C, sensor_loc = compute_C(PARAMS.n_y, spa_coor, PARAMS.n_x)

    # save the linear dynamical system into the log_dir specified in the PARAMS file
    sio.savemat(
        PARAMS.log_dir + "model.mat", {"A": A, "C": C, "sensor_loc": sensor_loc}
    )
