import json
import sys
import os

# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.io import get_downloaded_dataset_ids, get_dataset_filename
from data.config import Preprocess

from checksum import compute_checksum


checksum_path = os.path.join(
    os.path.dirname(
        os.path.realpath(__file__)),
    'checksum_preprocessed_data.json')

with open(checksum_path, 'r') as fp:
    checksums = json.load(fp)

dataset_ids = get_downloaded_dataset_ids()
for dataset_id in dataset_ids:
    for preprocess in [Preprocess.ONEHOT, Preprocess.STANDARDIZED, Preprocess.ROBUST, Preprocess.WHITENED]:
        checksum_key = f'{preprocess.value}/{dataset_id}'
        if checksum_key not in checksums:
            print(f'Preprocessed data {preprocess.value}/{dataset_id} was found on disk but was not expected.')
        else:
            with open(get_dataset_filename(dataset_id, preprocess=preprocess), 'rb') as fp:
                checksum = compute_checksum(fp)
            if checksums[checksum_key] != checksum:
                print(f'Mismatch in {preprocess.value}/{dataset_id}. Expected {checksums[checksum_key]} but got {checksum}.')
            del checksums[checksum_key]
if len(checksums) > 0:
    print('The following checksum keys were expected but not found on disk:')
    print('\n'.join(checksums.keys()))
