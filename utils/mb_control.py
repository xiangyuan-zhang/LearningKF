# SPDX-License-Identifier: MIT

import numpy as np


def infinite_kf(A, C, W, V):
    """
    Generates optimal gains for KF by solving the DARE

    Args:
        A, C, W, V: 2D array, representing linear system dynamics and noise covarainces

    Returns:
        S: 2D array, solution to the DARE
        AL: 2D array, closed loop (of the estimator dynamics)
        L: 2D array, optimal Kalman gain matrix

    """
    # Initialize the Riccati recursion with W
    S = W

    # Iterate the Riccati recursion until convergence
    for _ in range(5000):
        S = A @ S @ A.T + W - A @ S @ C.T @ np.linalg.inv(V + C @ S @ C.T) @ C @ S @ A.T

    # Compute the Kalman gain and A_L
    L = A @ S @ C.T @ np.linalg.inv(V + C @ S @ C.T)
    AL = A - L @ C
    return S, L, AL


def finite_kf(PARAMS, A, C, W, V, X0, N):
    """
    Compute a list of filter gains and policies

    Args:
        A, C: linear system matrices
        W, V: noise covariances
        X0: covariance of x0
        N: Number of time steps

    Returns:
        L_list, AL_list: Kalman gains and "A" matrix of the estimator

    """
    # Initialize the time-varying KF policies to be zero matrices
    L_list = np.zeros((N - 1, PARAMS.n_x, PARAMS.n_y))
    AL_list = np.zeros((N - 1, PARAMS.n_x, PARAMS.n_x))

    # Start the Riccati recursion
    S_t = X0  # initialize S_0 = X0
    for i in range(N - 1):
        L_t = A @ S_t @ C.T @ np.linalg.inv(V + C @ S_t @ C.T)  # compute Kalman gain
        S_t = (
            A @ S_t @ A.T
            + W
            - A @ S_t @ C.T @ np.linalg.inv(V + C @ S_t @ C.T) @ C @ S_t @ A.T
        )
        L_list[i] = L_t
        AL_list[i] = A - L_t @ C

    return L_list, AL_list
