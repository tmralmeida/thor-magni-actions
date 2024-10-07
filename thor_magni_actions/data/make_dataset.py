# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from thor_magni_actions.io.checkers import check_dir
from .config import (
    MIN_SPEED,
    MAX_SPEED,
    TRAJECTORY_LEN,
    SKIP_WINDOW,
    MIN_PEDESTRIANS,
    ACTIONS_INFO_PATH,
)
from .processors import MagniProcessor


@click.command()
@click.argument("dataset_name", type=click.STRING)
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def main(dataset_name, input_directory, output_directory):
    """Runs data processing scripts to turn interim data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final dataset")
    check_dir(input_directory, create=False)
    check_dir(output_directory, create=True)
    if dataset_name == "thor_magni":
        processor = MagniProcessor
    pp = processor(
        actions_path=ACTIONS_INFO_PATH,
        min_speed=MIN_SPEED,
        max_speed=MAX_SPEED,
        traj_len=TRAJECTORY_LEN,
        skip_window=SKIP_WINDOW,
        min_pedestrians=MIN_PEDESTRIANS,
    )
    pp.run(input_directory, output_directory)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
