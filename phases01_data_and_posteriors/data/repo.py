"""
File for interacting with openml repository. Provides functions for filtering
metadata and downloading datasets.
"""

import openml
import pandas as pd

from openml.exceptions import OpenMLServerError, PyOpenMLError
from requests.exceptions import ChunkedEncodingError
from arff import BadNominalValue, BadDataFormat

from .io import get_downloaded_dataset_ids, write_download_error, \
                is_task_file, read_task_dataset_ids, write_task_dataset_ids
from .repo_local import SUPPORTED_TASKS

DOWNLOADING_ERRORS = (OpenMLServerError, PyOpenMLError, ChunkedEncodingError,
                      BadNominalValue, BadDataFormat, EOFError) 


def get_dataset_ids():
    """Get the ids of all openml datasets that don't have missing values"""
    dataset_metadata = openml.datasets.list_datasets()
    metadata_df = pd.DataFrame.from_dict(dataset_metadata, orient='index')
    filtered_df = metadata_df[metadata_df.NumberOfInstancesWithMissingValues == 0]
    return filtered_df.did.values


def get_dataset_ids_by_task(task):
    """Get the ids of all openml datasets that have the specified task type"""
    if task not in SUPPORTED_TASKS:
        raise ValueError('Unsupported task: {}\nSupported tasks: {}'
                         .format(task, SUPPORTED_TASKS))
    tasks = openml.tasks.list_tasks()
    tasks_df = pd.DataFrame.from_dict(tasks, orient='index')
    dataset_ids_by_task = tasks_df[tasks_df.task_type == task].did.values
    return dataset_ids_by_task


def get_downloaded_dataset_ids_by_task(task):
    """Get the ids of all downloaded datasets that have the specified task type"""
    downloaded_ids = set(get_downloaded_dataset_ids())
    task_ids = get_dataset_ids_by_task(task)
    downloaded_task_ids = list(downloaded_ids.intersection(task_ids))
    downloaded_task_ids.sort()
    return downloaded_task_ids


def download_dataset(dataset_id):
    """
    Download openml dataset corresponding to dataset_id and return the
    corresponding dictionary of its content
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    return get_dataset_dict(dataset)


def download_dataset_and_log(dataset_id, verbose=False):
    """
    Download openml dataset corresponding to dataset_id and return the
    corresponding dictionary of its content. Log any download errors.
    """
    if verbose: print('Downloading dataset {} ...'.format(dataset_id), end=' ')
    try:        
        d = download_dataset(dataset_id)
        if verbose: print('Success!')
        return d
    # except DOWNLOADING_ERRORS as e:
    except Exception as e:
        write_download_error(e, dataset_id)
        if verbose: print('Failure!')


def get_dataset_dict(dataset):
    """Unpack the openml dataset object into a dictionary"""
    if dataset.default_target_attribute is None:
        print(
            f'Warning: dataset.default_target_attribute is None for {dataset.dataset_id}.\n'
            f'This means dataset has nothing to regress or classify on.')
    X, y, categorical, columns = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True)
    return {
        'X': X,
        'y': y,
        'categorical': categorical,
        'columns': columns
    }
