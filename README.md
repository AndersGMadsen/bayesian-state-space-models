# SP: Advanced Bayesian Methods for State Space Models

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


Certainly! Here's a beautifully formatted layout for the file structure in your README.md file:

markdown
Copy code
# bayesian-state-space-models

A project on Bayesian state space models.

## File Structure

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
├── papers
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

The root directory contains folders of animations in gif format, demonstrations of specific Bayesian methods in notebook format, examples of other interesting methods included in the project in notebook format, trajectory data for the specific tracks, papers containing relevant ideas and concepts, utility scripts for the project, two pedagogical explainer notebooks showcasing the primary work of the course, and a gitignore as well as a README file.

#### animations

Contains illustrative animations of the methods used.

#### demos

- normal.ipynb: Demonstration of the standard Kalman filter and the RTS smoother
- extended.ipynb: Demonstration of the extended Kalman filter and the extended RTS smoother
- unscented.ipynb: Demonstration of the unscented Kalman filter and the unscented RTS smoother
- particle.ipynb: Demonstration of the particle filter
- parzen.ipynb: Demonstration of the Parzen particle filter

#### examples

- sampling.ipynb: Examples of importance sampling.
- resampling.ipynb: Examples of resampling methods (systematic, residual, and stratefied)
- unscented_transformation.ipynb: Examples of the proporties of the unscented transformation and three ways of choosing Sigma points (Merwe, Julier, and Simplex).

#### output

Contains data from the MPC trajectory

#### papers


