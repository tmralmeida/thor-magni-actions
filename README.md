<h2 align="center">THÖR-MAGNI Act:&thinsp;Actions&thinsp;for&thinsp;Human&thinsp;Motion&thinsp;Modeling&thinsp;in&thinsp;Robot-Shared&thinsp;Industrial&thinsp;Spaces</h2>

<p align="center">
  <a href="https://dl.acm.org/doi/10.5555/3721488.3721626"><strong><code>Paper</code></strong></a>
  <a href="https://github.com/tmralmeida/thor-magni-actions/tree/main/data/processed/thor_magni"><strong><code>Dataset</code></strong></a>
  <a href="https://github.com/tmralmeida/thor-magni-tools"><strong><code>Dataset Tools</code></strong></a>
</p>


<div align='center'>
  <br><img src="doc/sample.gif" width=70%>
  <br>THÖR-MAGNI Act provides actions aligned with motion capture data and gazes from egocentric videos.
</div>

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


## Install packages for thor-magni-actions

Install [miniconda](http://docs.conda.io/en/latest/miniconda.html). Then, you can install all packages required by running:

```
conda env create -f environment.yml && conda activate thor-magni-actions && pip install -e .
```


## Prepare THÖR-MAGNI dataset (via thor-magni-tools)

0. Jump to [running thor-magni-actions modeling section](#running-thor-magni-actions-modeling) if you want to skip these preprocessing steps.
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
Change the config `in_path` and `out_path` settings accordingly. In this way, we obtain smoother and consistent trajectories.

1. From `thor-magni-tools`, run for each scenario directory:
   ```
   python -m thor_magni_tools.run_preprocessing
   ```
2. Check your `data/external` directory.
3. To align actions and trajectory data, run for each preprocessed scenario directory:
    ```
    unzip data/processed/thor_magni/QTM_frames_actions.zip -d data/processed/thor_magni/ && python -m thor_magni_tools.run_actions_merging --actions_path data/processed/thor_magni/QTM_frames_actions.csv --files_dir data/external/thor_magni/Scenario_{ID}/ --out_path data/interim/thor_magni/
    ```
6. To compute features, run:
   ```
   python -m thor_magni_actions.features.build_features data/interim/thor_magni data/interim/thor_magni
   ```
7. To create a dataset of fixed-length tracklets, run:
    ```
    python -m thor_magni_actions.data.make_dataset thor_magni data/interim/thor_magni data/processed/thor_magni
    ```


## Running thor-magni-actions modeling

To run the k-fold cross validation with the multi-task learning approach:
```
python -m thor_magni_actions.data_modeling.runners.k_fold_cv 5 thor_magni_actions/data_modeling/cfgs/mtl_tf.yaml
```


# Reference

Further details on the motivation and implementation of the THÖR-MAGNI *Act* dataset can be found in [the following paper](https://dl.acm.org/doi/10.5555/3721488.3721626):
```
@inproceedings{10.5555/3721488.3721626,
author = {Rodrigues de Almeida, Tiago and Schreiter, Tim and Rudenko, Andrey and Palmieri, Luigi and Stork, Johannes A. and Lilienthal, Achim J.},
title = {TH\"{O}R-MAGNI Act: Actions for Human Motion Modeling in Robot-Shared Industrial Spaces},
year = {2025},
publisher = {IEEE Press},
booktitle = {Proceedings of the 2025 ACM/IEEE International Conference on Human-Robot Interaction},
pages = {1083–1087},
numpages = {5},
keywords = {human activity prediction, human motion dataset, human motion modeling},
location = {Melbourne, Australia},
series = {HRI '25}
}
```

# Contact

The ``thor-magni-tools`` package has been developed by Tiago Rodrigues de Almeida [tiago.almeida@oru.se](tiago.almeida@oru.se).