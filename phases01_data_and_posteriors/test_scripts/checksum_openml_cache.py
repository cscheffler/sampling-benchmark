'''
Run this script to check that the data downloaded from openml.org is correct. The
MD5 checksum of each data file is computed and compared to the checksums stored in
checksum_openml_cache.json.

This script can also be used to update (recompute) checksum_openml_cache.json by
setting `action` to 'update' rather than 'check' below.
'''
import json
import glob
import os

from checksum import compute_checksum_from_file


# The OpenML cache is usualy located in your home directory at the path below.
openml_cache_path = os.path.expanduser('~/.openml/cache/org/openml/www/datasets/')

# Change action below to recompute all checksums and overwrite the existing JSON data.
action = ['check', 'update'][0]

# Load existing checksum data
checksum_path = os.path.join(
    os.path.dirname(
        os.path.realpath(__file__)),
    'checksum_openml_cache.json')
try:
    with open(checksum_path, 'r') as fp:
        checksums = json.load(fp)
except FileNotFoundError:
    checksums = {}

# Compute checksums from data files and compare against or update existing checksums.
for path in glob.iglob(os.path.join(openml_cache_path, '*', '*')):
    assert path.startswith(openml_cache_path)
    relative_path = path[len(openml_cache_path):]
    if action == 'check':
        if relative_path not in checksums:
            print(f'Path {relative_path} in cache but was not expected.')
        else:
            with open(path, 'rb') as fp:
                checksum = compute_checksum_from_file(fp)
            if checksums[relative_path] != checksum:
                print(f'Mismatch in {relative_path}. Expected {checksums[relative_path]} but got {checksum}.')
            del checksums[relative_path]
    else:
        with open(path, 'rb') as fp:
            checksum = compute_checksum_from_file(fp)
        checksums[relative_path] = checksum

if (action == 'check') and (len(checksums) > 0):
    print('The following paths were expected but not found in cache:')
    print('\n'.join(checksums.keys()))

if action == 'update':
    with open(checksum_path, 'w') as fp:
        json.dump(checksums, fp)
