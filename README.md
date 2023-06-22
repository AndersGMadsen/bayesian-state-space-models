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

| bayesian-state-space-models
|- animations (gif files with filtering and smoothing)
|- demos (demonstrations of methods)
|-- extended.ipynb
|-- normal.ipynb
|-- particle.ipynb
|-- unscented.ipynb
|- examples (theoretical illustrations)
|-- sampling.ipynb
|-- unscented_transformation.ipynb
|- output (trajectory data)
|- papers (litterature used in this project)
|- utils (utilities for the project)
|-- __init__.py
|-- cubic_spline_planner.py
|-- filter.py (classes for filters and smoothers)
|-- methods.py
|-- plots.py
|-- state_space_model.py
|-- systems.py
|-- vehicle_simulation.py
|- gitignore
|- Contrained_Notebook.ipynb (explainer containing theory)
|- Filtering_Notebook.ipynb (explainer containing theory)
|- README.md
