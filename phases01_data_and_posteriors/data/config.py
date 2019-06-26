"""
File for data config
"""

import os
import sys
import getpass
from enum import Enum


def path_from_unix_path(unix_path):
    return os.path.join(os.sep, *unix_path.split('/'))


DATA_BASE_PATH = ''  # Set this to the root of your data directory
if not DATA_BASE_PATH:
    sys.stderr.write('''
You need to specify a base path for the data directory in

    phases01_data_and_posteriors/data/config.py

The OpenML datasets, preprocessed datasets, and generated samples will all be
stored here. This requires _a lot_ of space, in the order of a few hundred GB.
''')
    raise ValueError('Invalid base path for data directory')


DATA_BASE_PATH = path_from_unix_path(DATA_BASE_PATH)
DATASETS_FOLDER = os.path.join(DATA_BASE_PATH, 'datasets')
ERRORS_FOLDER = os.path.join(DATASETS_FOLDER, 'errors')
TASKS_FOLDER = os.path.join(DATA_BASE_PATH, 'tasks')
SAMPLES_FOLDER = os.path.join(DATA_BASE_PATH, 'samples')
DIAGNOSTICS = os.path.join(SAMPLES_FOLDER, 'diagnostics.pkl')

# Folder name constants
class Preprocess(Enum):
    RAW = 'raw'
    ONEHOT = 'one-hot'
    STANDARDIZED = 'standardized'
    ROBUST = 'robust_standardized'
    WHITENED = 'whitened'


CONFIG = {
    preprocess.value + '_folder': os.path.join(DATASETS_FOLDER, preprocess.value)
    for preprocess in Preprocess
}
CONFIG['datasets_info'] = os.path.join(DATA_BASE_PATH, 'info.pkl')
CONFIG['errors_folder'] = ERRORS_FOLDER
CONFIG['tasks_folder'] = TASKS_FOLDER
CONFIG['samples_folder'] = SAMPLES_FOLDER
CONFIG['diagnostics'] = DIAGNOSTICS

# Create all directories that don't exist
for key, value in CONFIG.items():
    if key.endswith('_folder'):
        os.makedirs(value, exist_ok=True)
