thor-magni-actions
==============================

Actions and motion analysis and modeling in MAGNI dataset.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Prepare Datasets

## THÖR-MAGNI dataset (via thor-magni-tools)


1. Prepare [thor-magni-tools](https://github.com/tmralmeida/thor-magni-tools).
2. Change [config file](https://github.com/tmralmeida/thor-magni-tools-new/blob/main/thor_magni_tools/preprocessing/cfg.yaml) to:
------------
    in_path: PATH_TO_CSVs/Scenario_{ID}
    out_path: PATH_TO/thor-magni-actions/data/external/thor_magni
    preprocessing_type: 3D-best_marker 
    max_nans_interpolate: 100 

    options: 
        resampling_rule: 400ms 
        average_window: 800ms 
------------
Change the config `in_path` and `out_path` settings accordingly. In this way, we obtain smoother and more consistent trajectories.

1. From `thor-magni-tools`, run for each scenario directory:
   ```
   python -m thor_magni_tools.run_preprocessing
   ```
2. Check your `data/external` directory.


## Install packages for thor-magni-actions

Install [miniconda](http://docs.conda.io/en/latest/miniconda.html). Then, you can install all packages required by running:

```
conda env create -f environment.yml && conda activate thor-magni-actions
```

3. To compute features, run:
   ```
   python -m thor_magni_actions.features.build_features data/external/thor_magni data/interim/thor_magni
   ```
4. To create a dataset of fixed-length tracklets, run:
    ```
    python -m thor_magni_actions.data.make_dataset thor_magni data/interim/thor_magni data/processed/thor_magni
    ```