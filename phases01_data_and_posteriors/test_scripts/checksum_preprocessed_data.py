'''
Run this script to check that the preprocessed data to be used for fitting
models and generating samples in Phase 0 is correct. The MD5 checksum of
each data file is computed and compared to the checksums stored in
checksum_preprocessed_data.json.

This script can also be used to update (recompute) checksum_preprocessed_data.json
by setting `action` to 'update' rather than 'check' below.
'''
import json
import sys
import os

# Must be run from the project root so the data package will be added to
# the path.
sys.path.append(os.path.abspath('.'))

from data.io import get_downloaded_dataset_ids, get_dataset_filename
from data.config import Preprocess

from checksum import compute_checksum_from_dataset


# Change action below to recompute all checksums and overwrite the existing JSON data.
action = ['check', 'update'][0]

# Load existing checksum data
checksum_path = os.path.join(
    os.path.dirname(
        os.path.realpath(__file__)),
    'checksum_preprocessed_data.json')
try:
    with open(checksum_path, 'r') as fp:
        checksums = json.load(fp)
except FileNotFoundError:
    checksums = {}

# Compute checksums from data files and compare against or update existing checksums.
dataset_ids = get_downloaded_dataset_ids()
for dataset_id in dataset_ids:
    for preprocess in [Preprocess.ONEHOT, Preprocess.STANDARDIZED, Preprocess.ROBUST, Preprocess.WHITENED]:
        checksum_key = f'{preprocess.value}/{dataset_id}'
        if action == 'check':
            if checksum_key not in checksums:
                print(f'Preprocessed data {preprocess.value}/{dataset_id} was found on disk but was not expected.')
            else:
                with open(get_dataset_filename(dataset_id, preprocess=preprocess), 'rb') as fp:
                    checksum = compute_checksum_from_dataset(fp)
                if checksums[checksum_key] != checksum:
                    print(f'Mismatch in {preprocess.value}/{dataset_id}. Expected {checksums[checksum_key]} but got {checksum}.')
                del checksums[checksum_key]
        else:
            with open(get_dataset_filename(dataset_id, preprocess=preprocess), 'rb') as fp:
                checksum = compute_checksum_from_dataset(fp)
            checksums[checksum_key] = checksum

if (action == 'check') and (len(checksums) > 0):
    print('The following checksum keys were expected but not found on disk:')
    print('\n'.join(checksums.keys()))

if action == 'update':
    with open(checksum_path, 'w') as fp:
        json.dump(checksums, fp)
