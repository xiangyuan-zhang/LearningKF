# SPDX-License-Identifier: MIT

import os

import matplotlib.colors as mcolors
import numpy as np
import scipy.io as sio
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy.fft import rfftfreq


def pde_setup(PARAMS):
    """
    Set up the PDE

    Args:
        PARAMS: parameter file

    Returns:
        dx: spatial discretization size
        x: spatial coordinate
        kx: wavenumbers for FFT
        z0: initial condition
        z0h: rfft of z0
        tt_span: time span
        tt_save: discrete times
    """

    # Grid parameters
    Lx = PARAMS.Lx
    dx = Lx / PARAMS.n_x
    x = np.linspace(0, Lx - dx, PARAMS.n_x)
    kx = rfftfreq(PARAMS.n_x, dx) * 2 * np.pi

    # Initial condition
    z0 = np.cosh(10 * (x - 1 * Lx / 2)) ** (-1)

    # Time span and discrete times of trajectory
    # tt_span = (0, (PARAMS.n_save-1)*PARAMS.dt_save)
    tt_save = np.linspace(0, (PARAMS.n_save - 1) * PARAMS.dt_save, PARAMS.n_save)

    return dx, x, kx, z0, tt_save


def s_radius(A):
    """
    Compute the spectral radius of matrix A

    Args:
        A: complex 2D array
    """
    return np.real(np.linalg.eig(A)[0]).max()


def ft_matrix(n_x):
    """
    Computes the DFT matrix of size n_x with 'backward' normalization mode in numpy.fft.fft

    Args:
        n_x: number of discretization points

    Returns:
        DFT: discrete Fourier transform matrix
    """
    DFT = np.zeros((n_x, n_x), dtype=np.complex128)
    omega = np.exp(-2j * np.pi / n_x)  # Twiddle factor
    for i in range(n_x):
        for j in range(n_x):
            DFT[i, j] = omega ** (i * j)
    return DFT


def ift_matrix(n_x):
    """
    Computes the IDFT matrix of size n_x with 'backward' normalization mode in numpy.fft.fft

    Args:
        n_x: number of discretization points

    Returns:
        IDFT: inverse discrete Fourier transform matrix
    """
    DFT = ft_matrix(n_x)
    IDFT = np.conjugate(DFT.T) / n_x
    return IDFT


def compute_C(ny, spa_coor, nz):
    """
    Generates the observation matrix C of the DMD model, assuming that at the sensor location,
    perfect observations of the state could be measured.

    Args:
        ny: int
            Number of sensors (measurements)
        spa_coor: 1D array
            Coordinates of the spatial grid
        nz: int
            Dimension of the state vector z

    Returns:
        C: 2D array with ny rows and nz columns
            The measurement matrix of z, i.e., y_t = C*z_t
        x_sensors:

    """
    # Initialize C with a zero matrix
    # Initialize sensor_loc to be a zero vector
    C = np.zeros((ny, nz))
    sensors_loc = np.zeros(ny)
    # Loop over all sensors and assign entries of C
    for idx in range(ny):
        i_x = nz // ny * idx  # calculate the location of sensor No. idx
        C[idx, i_x] = 1  # set the corresponding entry in C to be 1
        sensors_loc[idx] = spa_coor[i_x]  # record the sensor coordinate in sensors_loc

    return C, sensors_loc


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    """
    Generate a truncated colormap
    """
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({name},{a:.2f},{b:.2f})".format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def visualize_kf(PARAMS, flag):
    """
    Plot the estimated trajectory from a given KF policy

    Args:
        PARAMS: dictinary files keeping the PDE parameters
        flag: id of the KF policy and the trajectory to be loaded

    Returns:
        None. A 3D plot will be saved at test/flag/est.png

    """
    test_dir = PARAMS.log_dir + flag + "/"
    data = sio.loadmat(test_dir + "trajectory.mat")
    (
        zz_ref,
        zz_est,
    ) = (
        data["zz_ref"],
        data["zz_est"],
    )
    dx = PARAMS.Lx / PARAMS.n_x
    x = np.linspace(0, PARAMS.Lx - dx, PARAMS.n_x)
    tt_save = np.linspace(0, (PARAMS.n_save - 1) * PARAMS.dt_save, PARAMS.n_save)

    norm_center = colors.TwoSlopeNorm(
        vmin=0,
        vcenter=max(zz_ref.max(), zz_est.max()) / 2,
        vmax=max(zz_ref.max(), zz_est.max()),
    )
    colormap = truncate_colormap(plt.get_cmap("jet"), 0, 1)
    a, b = np.meshgrid(range(len(tt_save)), range(len(x)))

    fig = plt.figure(figsize=(10, 8))
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    ax = fig.add_subplot(111, projection="3d")
    cf = ax.plot_surface(
        a,
        b,
        zz_est,
        cmap=colormap,
        norm=norm_center,
        rcount=PARAMS.n_x,
        ccount=PARAMS.n_save,
    )
    _ = fig.colorbar(
        cf, format="%.1f", orientation="horizontal", shrink=0.45, pad=0.03, ax=ax
    )
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_zlim(0, 1)
    plt.savefig(test_dir + "est_traj.png", dpi=300)
    plt.close(fig)


def evolve_kf(PARAMS, L, AL, setting, flag):
    """
    Visualize the KF policy by 3D plot

    Args:
        PARAMS: dictinary files keeping the PDE parameters
        L, AL: KF policies
        setting: either 'Inf_KF' or 'Finite_KF', where the first setting takes
                 a stationary KF policy and the second setting takes a list
                 of time-varying KF policies
        flag: id of the test case

    Returns:
        None. A 3D plot, trajectory, KF policy, and cost vector will be saved to test/flag/

    """
    _, _, _, z0, _ = pde_setup(PARAMS)
    model = sio.loadmat(PARAMS.log_dir + "model.mat")
    A, C = model["A"], model["C"]
    W = PARAMS.var_w * np.identity(PARAMS.n_x)
    V = PARAMS.var_v * np.identity(PARAMS.n_y)

    # Run both the PDE dynamics and the estimated ones
    rng = np.random.default_rng(seed=PARAMS.seed)  # fix the random seed
    zz_ref, zz_est = np.zeros((PARAMS.n_x, PARAMS.n_save)), np.zeros(
        (PARAMS.n_x, PARAMS.n_save)
    )
    zz_ref[:, 0] = zz_est[:, 0] = z0
    est_x = np.array([z0]).T
    cost_list = []
    cost = 0

    # Evolve the system dynamics and the estimator dynamics forward
    for step_num in range(PARAMS.n_save - 1):
        # Accumulate the running MMSE
        cost += (zz_ref[:, step_num] - est_x.flatten()).T @ (
            zz_ref[:, step_num] - est_x.flatten()
        )
        cost_list.append(cost)

        # Sample random system noise from N(0, W)
        w_t = rng.multivariate_normal(np.zeros(A.shape[0]), W, size=(1,)).T

        # Update the next system state
        zz_ref[:, step_num + 1] = A @ zz_ref[:, step_num] + w_t.flatten()

        # Evolve the state estimator
        x_t = np.array([zz_ref[:, step_num]]).T
        v_t = rng.multivariate_normal(
            np.zeros(C.shape[0]), V, size=(1,)
        ).T  # Sample measurement noise from N(0, V)
        y_t = C @ x_t + v_t  # update the measurement process

        if setting == "Inf_KF":
            est_x = AL @ est_x + L @ y_t  # KF equation with stationary policy
        elif setting == "Finite_KF":
            if step_num < L.shape[2]:  # KF equation with time-varying policies
                est_x = AL[step_num] @ est_x + L[step_num] @ y_t
            else:
                est_x = AL[-1] @ est_x + L[-1] @ y_t
        else:
            print("Flag Error!")
            return
        zz_est[:, step_num + 1] = est_x.flatten()

    # Accumulate the terminal cost
    cost += (zz_ref[:, -1] - est_x.flatten()).T @ (zz_ref[:, -1] - est_x.flatten())
    cost_list.append(cost)

    # Calculate the time-average cost
    cost = cost / PARAMS.n_save

    # Save the trajectories into test/flag/
    os.makedirs(PARAMS.log_dir, exist_ok=True)
    test_dir = PARAMS.log_dir + flag + "/"
    os.makedirs(test_dir, exist_ok=True)
    sio.savemat(
        test_dir + "trajectory.mat",
        {"zz_ref": zz_ref, "zz_est": zz_est, "cost": cost, "cost_list": cost_list},
    )  # save trajectory

    # save the KF policies
    test_dir = PARAMS.log_dir + flag + "/"
    sio.savemat(test_dir + "policy.mat", {"L": L, "AL": AL})
