"""
File for downloading the OpenML datasets
"""

import sys
import os
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.config import CONFIG
from data.io import write_dataset


import pickle
import pandas as pd

import openml
from openml.exceptions import OpenMLServerError, PyOpenMLError
from requests.exceptions import ChunkedEncodingError
from arff import BadNominalValue

CHECKPOINT_ITERS = 25


def get_dataset_ids():
    """Get the ids of the dotasets to download"""
    dataset_metadata = openml.datasets.list_datasets()
    metadata_df = pd.DataFrame.from_dict(dataset_metadata, orient='index')
    filtered_df = metadata_df[metadata_df.NumberOfInstancesWithMissingValues == 0]
    return filtered_df.did.values
    
    
def download_datasets(dataset_ids, start_iteration=0, verbose=False):
    """Download the datasets that correspond to all of the give IDs"""
    num_datasets = len(dataset_ids)
    good_dataset_ids = []
    bad_dataset_ids = []
    exceptions = []
    
    # load previous saved values of above variables
    info_filename = get_info_filename()
    if os.path.isfile(info_filename):
        if verbose: print('Loading download info from file')
        with open(info_filename, 'rb') as f:
            info = pickle.load(f)
        start_iteration = info['iteration']
        good_dataset_ids = info['good_dataset_ids']
        bad_dataset_ids = info['bad_dataset_ids']
        exceptions = info['exceptions']

    # loop through dataset_ids and download corresponding datasets
    for i in range(start_iteration, num_datasets):
        dataset_id = dataset_ids[i]
        if verbose:
            print('{} of {}\tdataset ID: {} ...' \
                  .format(i + 1, num_datasets, dataset_id), end=' ')
        # OpenML likes to throw all kinds of errors when getting datasets
        try:
            dataset_id = int(dataset_id)
            dataset = openml.datasets.get_dataset(dataset_id)
            good_dataset_ids.append(dataset_id)
            write_dataset(dataset_id, dataset)
            if verbose: print('Success')
        # except (OpenMLServerError, PyOpenMLError, ChunkedEncodingError,
        #         BadNominalValue, EOFError) as e:
        except Exception as e:
            bad_dataset_ids.append(dataset_id)
            exceptions.append(e)
            if verbose: print('Failure', repr(e))
        # checkpoint info
        if (i + 1) % CHECKPOINT_ITERS == 0:
            if verbose:
                print('Reached iteration {}. Writing download info' \
                      .format(i + 1))
            write_download_info({
                'iteration': i + 1,
                'num_datasets': num_datasets,
                'good_dataset_ids': good_dataset_ids,
                'bad_dataset_ids': bad_dataset_ids,
                'exceptions': exceptions
            })

    
def write_download_info(info):
    """Write the information about the success/failure of downloading datasets"""
    filename = get_info_filename()
    with open(filename, 'wb') as f:
        pickle.dump(info, f) 


def get_info_filename():
    """Get location of where to write the download information"""
    return CONFIG['datasets_info']


if __name__ == '__main__':
    download_datasets(get_dataset_ids(), verbose=True)
