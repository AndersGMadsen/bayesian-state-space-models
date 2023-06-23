# Advanced Bayesian Methods for State Space Models 

Anders Gjølbye Madsen (s194260) and William Lehn-Schiøler (s194272)

## Introduction

The course is structured around independent study and research, with a focus on the implementation and
understanding of advanced Bayesian methods for state-space models. It is offered for 2.5 ECTS credits during
the spring semester and extends to 5.0 ECTS credits over a three-week period in June. The examination is a
hand-in of the GitHub repository and an oral presentation of developed content.

## Installation

Clone the repository and run "pip install -r requirements.txt" to install dependencies

## File Structure

Here is a high-level overview of the file structure in this repository:


```
.
├── animations
├── demos
│   ├── normal.ipynb
│   ├── extended.ipynb
│   ├── unscented.ipynb
│   ├── particle.ipynb
│   └── parzen.ipynb
├── examples
│   ├── sampling.ipynb
│   ├── resampling.ipynb
│   └── unscented_transformation.ipynb
├── output
├── litterature
├── utils
│   ├── __init__.py
│   ├── cubic_spline_planner.py
│   ├── filter.py
│   ├── methods.py
│   ├── plots.py
│   ├── state_space_model.py
│   ├── systems.py
│   └── vehicle_simulation.py
├── .gitignore
├── Constrained_Notebook.ipynb
├── Filtering_Notebook.ipynb
└── README.md

```

### Root Directory

The root directory contains folders of animations in gif format, demonstrations of specific Bayesian methods in notebook format, examples of other interesting methods included in the project in notebook format, trajectory data for the specific tracks, books and papers containing relevant ideas and concepts, utility scripts for the project, two pedagogical explainer notebooks showcasing the primary work of the course, and a gitignore as well as a README file.

#### animations

Contains illustrative animations of the methods used.

#### demos

- normal.ipynb: Demonstration of the standard Kalman filter and the RTS smoother
- extended.ipynb: Demonstration of the extended Kalman filter and the extended RTS smoother
- unscented.ipynb: Demonstration of the unscented Kalman filter and the unscented RTS smoother
- particle.ipynb: Demonstration of the particle filter
- parzen.ipynb: Demonstration of the Parzen particle filter
- number_particles.ipynb: Experiment MSE vs. number of particles
- measurement_noise_illustration.ipynb: demo of how different methods work with different measurements distributions.
- tracks.ipynb: Shows the tracks used in the project.

#### examples

- sampling.ipynb: Examples of importance sampling.
- resampling.ipynb: Examples of resampling methods (systematic, residual, and stratefied)
- unscented_transformation.ipynb: Examples of the proporties of the unscented transformation and three ways of choosing Sigma points (Merwe, Julier, and Simplex).

#### output

Contains data from the MPC trajectory

#### literature
- Bayesian Filtering and Smoothing, Simo Sarkka
- a few relevant papers

#### utils

- __init__.py: imports from all files in utils
- cubic_spline_planner.py: contains classes Spline and Spline2D
- filter.py: contains the following classes for Bayesian and Monte Carlo methods for filtering and smooting: KF (Kalman filter), EKF (extended Kalman filter), UKF (unscented Kalman filter), PF (particle filtering), and PPF (Parzen particle filtering), as well as PF_CONSTRAINED (constrained particle filtering).
- methods.py: contains classes for sampling methods: systematic_resampling, residual resampling, stratefied_resampling, and sample_from_mixture (sampling from a Gaussian Micture Model).
- plots.py: contains functions rgba_to_rgb, conf_ellipse, plot_trajectory, as well as visualize_filter and visualize_filter_and_smoother. Contains also classes PlotAnimation and PlotParzenAnimation + show_animations function.
- state_space_model.py: contains class for the state space model used to describe the driving car. The Dynamics come from Ex. 4.3 of Bayesian Filtering and Smoothing. Information of both a linear and a nonlinear system can be extracted.
- systems.py: contains classes used to generate the systems (CarTrajectoryLinear, CarTrajectoryNonLinear, and MPCTrajectory). Information from the state space model can or cannot be used to generate the system. Contains also functions to generate specific tracks (track_example1, track_example2, and track_example3)
- utils.py: contains functions make_constraint, point_in_polygon, line_search, and nearest_point, as well as functions used in the unscented transformation
- vehicle_simulation: contains function for the car simulation, that is, function plot_car, and classes Vehicle, MPC, and Simulation

### Filtering_Notebook

Explainer notebook that goes through the theory of KF, RTS, EKF, UKF, PF (and PPF), and provides demonstration in forms of plots and animations of the methods.

### Constraint_Notebook

Explainer notebook that goes through the problem of constraining a state space model, and provides demonstration of different formulations as well as illustrations.


## Usage

This project is meant as an introduction as well as an exploration of existing Bayesian and Monte Carlo methods for filtering and smoothing accompanied with illustrative examples. Novel methods such as constrained particle filtering and constrained UKF are also presented and discussed.

## License

MIT License
