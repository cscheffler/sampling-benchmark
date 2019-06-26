import json
import glob
import os

from checksum import compute_checksum


openml_cache_path = os.path.expanduser('~/.openml/cache/org/openml/www/datasets/')

checksum_path = os.path.join(
    os.path.dirname(
        os.path.realpath(__file__)),
    'checksum_openml_cache.json')

with open(checksum_path, 'r') as fp:
    checksums = json.load(fp)

for path in glob.iglob(os.path.join(openml_cache_path, '*/*')):
    assert path.startswith(openml_cache_path)
    relative_path = path[len(openml_cache_path):]
    if relative_path not in checksums:
        print(f'Path {relative_path} in cache but was not expected.')
    else:
        with open(path, 'rb') as fp:
            checksum = compute_checksum(fp)
        if checksums[relative_path] != checksum:
            print(f'Mismatch in {relative_path}. Expected {checksums[relative_path]} but got {checksum}.')
        del checksums[relative_path]
if len(checksums) > 0:
    print('The following paths were expected but not found in cache:')
    print('\n'.join(checksums.keys()))
