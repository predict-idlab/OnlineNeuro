#### Explanation

- **Problem_config** The problem configuration.
  * Experiment/name Name of the problem. Accepts circle_classification, rose_regression, multiobjective_problem,
    axonsim_nerve_block, axonsim_regression.

    You can see a full list of implemented problems running ```python3 ```
- **Classification** Defines whether the problem is a classifier or a regression.
- **Sparse, Variational** Booleans that define the type of Gaussian Process
- **init_samples** Number of initial samples to request in the beginning/start model.
- **noise_free** It should be kept as true. (TODO add discussion as to why later)
- **batch_sampling** It should be kept as false. (TODO needs to be implemented).
- **trainable_likelihood** Whether the likelihood of the model is trainable. In most cases should be kept as false.
