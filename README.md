# Sim Interace

![Static Badge](https://img.shields.io/badge/Matlab-2023-green)

Optimized sampling for Matlab functions (or data acquisitions)

## Install requirements

```bash
# Install virtualenv
pip install virtualenv

# Create a virtual environment named venv
virtualenv venv

# Activate the virtual environment
# On Linux or macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

In addition, install the required matlab engine (the default pip wheel is currently Matlab 2024a). 

You can check the version by opening MATLAB and run the command: version

So for instance, if don't have 2024a version, but your engine for Matlab is 2023b you can do the following command:

```bash
# Install matlabengine for Matlab version 2023b
pip install matlabengine==23.2.3

# Or in the case of 2024a
pip install matlabengine==24.1.2
```

For more information: https://pypi.org/project/matlabengine/

## Usage

```py
TBD
```

### Using the graphic user interface.
 The bash scripts takes care of adding the required folders to PYTHONPATH and calls /frontend/app.py
```bash
$ sh run.sh
```
Then open within a browser 
[https:localhost:9000](https:localhost:9000)

## Configuration

The code is written in a way that minimum changes are required in the script.
All parameters (connection and optimization) are selected in the config.json

```json
{
  "connection_config": {
    "ip": "127.0.0.0",
    "port": 10000,
    "Timeout": 20,
    "ConnectTimeout": 30,
    "initiator": "Python",
    "target": "MATLAB"
  },
  "problem_config": {
    "name": "axonsim_nerve_block",
    "type": "classification",
    "config_path": "/config/experiment_axonsim_nerve_block.json"
  },
  "model_config": {
    "type": "GP",
    "classification": true,
    "scale_inputs": true,
    "constrains": false,
    "sparse": false,
    "variational": true,
    "noise_free": true,
    "trainable_likelihood": false,
    "init_samples":15,
    "batch_sampling": false,
    "num_query_points": 1
  },
  "path_config": {
    "save_path": "./simulations",
    "benchmark_path": "./benchmarks",
    "axonsim_path": "/$YOUR_PATH$/AxonSim/"
  }
}
```

- **Problem** The problem to be solved. Accepts 'circle', 'rose', and 'axon'.
- **Classification** Defines whether the problem is a classifier or a regression.
- **Sparse, Variational** Booleans that define the type of Gaussian Process
- **init_samples** Number of initial samples to request in the beginning/start model.
- **noise_free** It should be kept as true. (TODO add discussion as to why later)
- **batch_sampling** It should be kept as false. (TODO needs to be implemented).
- **trainable_likelihood** Whether the likelihood of the model is trainable. In most cases should be kept as false.

## Toy Problems

### Circle

A classification problem approximated sith SVGP and log-likelihood.

<img src="figures/circle.jpg" alt="circle" height="250"/> 
<img src="animations/animation_circle.gif" alt="circle_opt" height="250"/>

<!-- ![circle.jpg](circle.jpg) ![animation_circle.gif](animations%2Fanimation_circle.gif) -->

### Rosenbrock

A regression problem with smooth surfaces approximated with GP.

<img src="figures/rose.jpg" alt="rose" height="250"/> 
<img src="animations/animation_rose.gif" alt="rose_opt" height="250"/>

<!-- ![rose.jpg](rose.jpg)![animation1.gif](animations%2Fanimation_rose.gif) -->

### Multiobjective

A two objective problem with two inputs. Optimizing towards joined targets using Pareto Front.
% TODO is there a way to set up the constraints from matlab's end and pass them to Python?

<img src="figures/vlmop2.jpg" alt="vlmop2" width="450"/>
<img src="animations/animation_vlmop2.gif" alt="animation_vlmop2" width="450"/>

## Axonsim

A classification problem using SVG to detect the minimum currrent/pulse duration to create an AP.
%TODO need to add labels to the axis of these plots

<img src="animations/animation_threshold.gif" alt="Threshold searching" height="250"/>

## Other sources
