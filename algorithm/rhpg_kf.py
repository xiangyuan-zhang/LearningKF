# SPDX-License-Identifier: MIT

import numpy as np
import scipy.io as sio
import scipy.linalg


def adam_opt(p, grad, i, m_old, v_old, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Standard adam optimizer for computing the next iterate of GD

    Args:
        p: 2D array, current policy
        grad: 2D array, analytic gradient or gradient estimate of the current policy
        i: integer, iteration index
        m_old, v_old: 2D array, matrices for implementing adam update
        lr: float, learning rate for adam update
        beta1, beta2, eps: floats, variables using standard choices

    Returns:
        new_p: 2D array, updated policy
        m, v: 2D arrays, updated matraices for adam update

    """
    m = beta1 * m_old + (1 - beta1) * grad
    v = beta2 * v_old + (1 - beta2) * (grad.T @ grad)
    m_hat = m / (1 - (beta1**i))
    v_hat = v / (1 - (beta2**i))
    new_p = p - lr * m_hat @ np.linalg.inv(
        scipy.linalg.sqrtm(v_hat) + eps * np.identity(v_hat.shape[0])
    )
    return np.real(new_p), m, v


def rhpg_adam(PARAMS, X0, N, z0, eta, max_grad_norm=1e-4):
    """
    RHPG algorithm for learning the KF policies

    Args:
        PARAMS: dict file to track all the PDE parameters
        X0: 2D array, covariance matrix of the random initial state
        N: integer, number of steps as an user input
        z0: initial state of PDE
        eta: float, global learning rate (stepsize)
        max_grad_norm: float, threshold on gradient norm to stop GD update

    Returns:
        None, the convergent policies will be saved into test/RHPG_Adam/policy.mat
    """
    # Constuct the linear dynamical system and specify noise statistics
    model = sio.loadmat(PARAMS.log_dir + "model.mat")
    A, C = model["A"], model["C"]
    W = PARAMS.var_w * np.identity(PARAMS.n_x)
    V = PARAMS.var_v * np.identity(PARAMS.n_y)

    # 2D array, covariance matrix of the zero-mean random noise
    # to be injected into both the current state and current estimate
    # for convexification purpose
    Theta = PARAMS.var_theta * np.identity(PARAMS.n_x)

    # Initialize the KF policy list to be zero matrices
    p_list = np.zeros((N - 1, PARAMS.n_x, PARAMS.n_x + PARAMS.n_y))

    # Extra terms in Hessian and gradient due to the additional noise injection
    Delta = np.vstack(
        (np.hstack((Theta, Theta @ C.T)), np.hstack((C @ Theta, C @ Theta @ C.T)))
    )
    Xi = np.hstack((A @ Theta, A @ Theta @ C.T))

    # Set the mean and covaraince matrices
    # set the mean of estimated state to be identical to the mean of true state
    # for simplicity, but it is not required.
    mu_x = np.array([z0]).T
    mu_est_x = np.array([z0]).T

    var_x = mu_x @ mu_x.T + X0
    var_est_x = mu_est_x @ mu_est_x.T
    cov_x_est_x = mu_x @ mu_est_x.T
    cov_est_x_x = mu_est_x @ mu_x.T

    # Start the forward dynamic programming loop
    for h in range(0, N - 1):
        # arbitrary initialization to start GD
        p_h = p_list[h]

        # initialize m_h and v_h to be zero matrices for using adam update
        m_h, v_h = np.zeros_like(p_h), np.zeros_like(p_h.T @ p_h)

        # Compute the Hessian matrix and the constant term in gradient formula
        H = np.vstack(
            (
                np.hstack((var_est_x, cov_est_x_x @ C.T)),
                np.hstack((C @ cov_x_est_x, C @ var_x @ C.T + V)),
            )
        )
        G = np.hstack((A @ cov_x_est_x, A @ var_x @ C.T))
        grad_norm = np.linalg.norm(2 * (p_h @ (H + Delta) - (G + Xi)))
        idx = 0

        # Execute GD until convergence
        while grad_norm > max_grad_norm:
            grad = 2 * (p_h @ (H + Delta) - (G + Xi))
            grad_norm = np.linalg.norm(grad)
            if np.mod(idx, 100) == 0:
                print(
                    "- Solving h=%d, Iteration=%d, Grad_Norm=%.4f" % (h, idx, grad_norm)
                )

            # apply the standard adam update to calculate the updated policy
            p_h, m_h, v_h = adam_opt(p_h, grad, idx + 1, m_h, v_h, eta)
            idx += 1

        # Record the converged policy
        # If not the last step, use the converged policy as a warm start for the next iteration
        p_list[h] = p_h
        if h != N - 2:
            p_list[h + 1] = p_list[h]

        # Update the mean and convariance matrices
        mu_est_x = p_h[:, : PARAMS.n_x] @ mu_est_x + p_h[:, PARAMS.n_x :] @ C @ mu_x
        var_est_x = p_h @ H @ p_h.T

        cov_x_est_x = (
            A @ cov_x_est_x @ p_h[:, : PARAMS.n_x].T
            + A @ var_x @ C.T @ p_h[:, PARAMS.n_x :].T
        )
        cov_est_x_x = (
            p_h[:, : PARAMS.n_x] @ cov_est_x_x @ A.T
            + p_h[:, PARAMS.n_x :] @ C @ var_x @ A.T
        )

        mu_x = A @ mu_x
        var_x = A @ var_x @ A.T + W

    # Save the converged policies into a matrix file
    sio.savemat(PARAMS.log_dir + "rhpg_policy.mat", {"policy": p_list})
