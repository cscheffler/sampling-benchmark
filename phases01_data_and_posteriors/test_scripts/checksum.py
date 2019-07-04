import hashlib
import pickle
import numpy as np
from scipy.sparse import issparse


READ_BUFFER_SIZE = 2**20


def compute_checksum_from_file(fp):
    checksum = hashlib.md5()
    for chunk in iter(lambda: fp.read(READ_BUFFER_SIZE), b''):
        checksum.update(chunk)
    checksum = checksum.hexdigest()
    return checksum


def compute_checksum_from_dataset(fp):
    dataset = pickle.load(fp)
    checksum = hashlib.md5()
    if issparse(dataset['X']):
        checksum.update(dataset['X'].indices)
        checksum.update(dataset['X'].data)
    else:
        checksum.update(np.ascontiguousarray(dataset['X']))
    checksum.update(dataset['y'])
    if 'categorical' in dataset:
        checksum.update(np.array(dataset['categorical']))
    if 'columns' in dataset:
        checksum.update(np.array(dataset['columns']))
    checksum = checksum.hexdigest()
    return checksum
