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

checksums = {}

dataset_ids = get_downloaded_dataset_ids()
for dataset_id in dataset_ids:
    dataset_id = str(dataset_id)  # JSON stores integer keys as strings
    for preprocess in [Preprocess.ONEHOT, Preprocess.STANDARDIZED, Preprocess.ROBUST, Preprocess.WHITENED]:
        try:
            with open(get_dataset_filename(dataset_id, preprocess=preprocess), 'rb') as fp:
                checksum = compute_checksum(fp)
        except FileNotFoundError as e:
            # Skip the files we don't have since we'll compute those later
            print(f'FileNotFoundError. {e.strerror}: {e.filename}')
        else:
            key = f'{preprocess.value}/{dataset_id}'
            checksums[key] = checksum

with open(checksum_path, 'w') as fp:
    json.dump(checksums, fp)
