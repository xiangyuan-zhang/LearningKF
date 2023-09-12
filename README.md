<!--
SPDX-License-Identifier: MIT
-->

# Receding-Horizon Policy Search for Learning Estimator Designs

This is the official implementation of the paper [Global Convergence of Receding-Horizon Policy Search in Learning Estimator Designs](https://arxiv.org/abs/2309.04831) authored by X. Zhang, S. Mowlavi, M. Benosman, and T. Başar.

<p align="center">
    <img src=gifs/3d_video_transparent.gif width="600" height="550" />
</p>
<p align="center">
<em1>3D visualizations of different filters in the estimation of the convection-diffusion equation.</em1>
</p>


<p align="center">
    <img src=gifs/2d_video_transparent.gif width="600" height="250" />
</p>
<p align="center">
<em2>2D visualizations of different filters compared with the ground truth.</em2>
</p>


## Usage

To run the experiments described in the paper, execute
```bash
cd LearningKF
python setup_pde.py
python train_kf.py
```

## Testing

To reproduce the plots presented in the paper, run
```bash
cd LearningKF/tools
# Plot eigen spectrum of the convection-diffusion model
python 1_plot_eigen.py

# Evolve the ground truth PDE trajectory
python 2_save_plot_pde_traj.py

# Visualize the trajectory estimated by model-based KF
python 3_test_plot_KF_traj.py

# Visualize trajectories estimated by RHPG filters with different horizon length N
python 4_test_plot_rhpg_traj.py

# Test the performances of different filters across different initial conditions
python 5_test_save_rand_init.py
python 6_plot_rand_init.py
```

The sturcture of the repository is as follows.

    LearningKF
    ├── setup_pde.py
    ├── train_kf.py
    ├── utils
        ├── helper.py
        └── mb_control.py
    └── tools
        ├── 0_check_obsv.py
        ├── 1_plot_eigen.py
        ├── 2_save_plot_pde_traj.py
        ├── 3_test_plot_KF_traj.py
        ├── 4_test_plot_rhpg_traj.py
        ├── 5_test_save_rand_init.py
        └── 6_plot_rand_init.py


## Citation

If you use the software, please cite the following [arXiv preprint](https://arxiv.org/abs/2309.04831):

```BibTeX
@article{zhang2023global,
    title = {Global Convergence of Receding-Horizon Policy Search in Learning Estimator Designs},
    author = {Zhang, Xiangyuan and Mowlavi, Saviz and Benosman, Mouhacine and Ba{\c{s}}ar, Tamer},
    journal = {arXiv preprint arXiv:2309.04831},
    year = {2023}
}
```

