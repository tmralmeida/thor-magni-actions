# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from .spatio_temporal_features import get_spatiotemporal_features
from ..io.create import create_dir


@click.command()
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def main(input_directory, output_directory):
    """Runs data feature extraction scripts to turn processed data from (data/external) into
    features extracted ready to be analyzed (saved in data/interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("extracting features from the dataset")
    scenarios = os.listdir(input_directory)
    for scenario in scenarios:
        out_dir = os.path.join(output_directory, scenario)
        files = os.listdir(os.path.join(input_directory, scenario))
        for csv_file in files:
            trajectories = pd.read_csv(
                os.path.join(input_directory, scenario, csv_file)
            )
            ext_features = get_spatiotemporal_features(trajectories)
            create_dir(out_dir)
            save_path = os.path.join(out_dir, csv_file)
            ext_features.to_csv(save_path)
            logger.info("%s saved", save_path)


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
