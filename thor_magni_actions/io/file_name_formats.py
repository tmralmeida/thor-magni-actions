from collections import namedtuple


def get_file_name_format(dataset_name: str, file_name: str):
    Date = namedtuple("Date", ["day", "month", "year"])
    if dataset_name == "thor_magni":
        splits = file_name.split("_")
        date_splits = splits[1]
        run_splits = splits[3]
        run = run_splits.split(".")[0]
        date = Date(date_splits[:2], date_splits[2:4], date_splits[4:])
        scenario_id = splits[2].split("SC")[-1]
        run_id = run.split("R")[-1]
    else:
        raise NotImplementedError(dataset_name)

    return dict(date=date, scenario_id=scenario_id, run_id=run_id)
