# Simulators
## Overview
This directory contains simulator's related code for MATLAB and Python.
Each "simulator" encapsulates the logic for an experiment and associated methods for handling query points and observations.
These examples should serve as a guidance on how to extend the current codebase for new problems, or changing the objective functiosn.

### MATLAB Simulators
Concerns problems in standard MATLAB and for the library Axonsim.
- main.m serves as the main entry point for all problems. This file needs to be updated in order to parametrize and call the currect problem, as well as post-processing the simulation results.
- circle_problem.m, rosenbrock_problem.m and multiobjective_problem.m are standard optimization problems for classification, regression and MOO cases.
- axonsim_problem.m and axonsim_call.m concer the Axonsim problems.

### Python Simulators
Concerns problems in Python and the Cajal/AxonML libraries.
- toy_problems.py contains illustrative examples similar to the ones mentioned for MATLAB (circle, rosenbrock and MOO).
- cajal_problems.py contains the related code for executing Cajal/AxonML simulations, post-processing the results and savingthe results.
- the folder python/pulses contains custom classes that extend the standard pulses present in Cajal/AxonML.
