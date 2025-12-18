#### Explanation
## Configuration
Using the GUI there's no need of writing any code. However, you may want to modify some of the default configurations.
The code is written in a way that minimum changes are required in the script.

All parameters (connection and optimization) are selected in the config.json.
Bellow is the example of the config.json

```json
{
  "connection_config": {
    "ip": "127.0.0.1",
    "port": 9000,
    "Timeout": 30,
    "ConnectTimeout": 30,
    "target": "Python/MATLAB"
  },
  "problem_config": {
    "experimentParameters": {
      "problem_name": "EXPERIMENT",
      "type": "classification/regression",
      "config_file": "PATH"
    }
  },
  "model_config": {
    "type": "MODEL",
    "scale_inputs": true,
    "constrains": false,
    "sparse": false,
    "variational": true,
    "noise_free": true,
    "trainable_likelihood": false
  },
  "path_config": {
    "save_path": "./results/simulations",
    "benchmark_path": "./benchmarks",
    "axonsim_path": "AXONSIM_PATH",
    "cajal_path": "CAJAL_PATH",
    "axonml_path": "AXONML_PATH"
  },
  "optimizer_config": {
    "init_samples":5,
    "batch_sampling":false,
    "n_iters":3,
    "num_query_points":1
  }
}

```

- **Problem_config** The problem configuration.
  * Experiment/name Name of the problem. Accepts circle_classification, rose_regression, multiobjective_problem,
    axonsim_nerve_block, axonsim_regression.

    You can see a full list of implemented problems running ```python3 ```
- **Classification** Defines whether the problem is a classifier or a regression.
- **Sparse, Variational** Booleans that define the type of Gaussian Process
- **init_samples** Number of initial samples to request in the beginning/start model.
- **noise_free** It should be kept as true. (TODO add discussion as to why later)
- **batch_sampling** Boolean to indicate if batch sampling (specify number of sampels). Currently not implemented.
- **trainable_likelihood** Whether the likelihood of the model is trainable. In most cases should be kept as false.
