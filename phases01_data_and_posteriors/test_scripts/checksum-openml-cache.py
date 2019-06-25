import hashlib
import glob
import os
import json

openml_cache_path = '/home/carl/.openml/cache/org/openml/www/datasets/'

with open('checksum-openml-cache.json', 'r') as fp:
    checksums = json.load(fp)

read_buffer_size = 2**20
for path in glob.iglob(os.path.join(openml_cache_path, '*/*')):
    assert path.startswith(openml_cache_path)
    rel_path = path[len(openml_cache_path):]
    checksum = hashlib.md5()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(read_buffer_size), b''):
            checksum.update(chunk)
    checksum = checksum.hexdigest()
    if rel_path not in checksums:
        print('Path', rel_path, 'in cache but was not expected.')
    elif checksums[rel_path] != checksum:
        print(f'Mismatch in {rel_path}. Expected {checksums[rel_path]} but got {checksum}.')
        del checksums[rel_path]
    else:
        del checksums[rel_path]

if len(checksums) > 0:
    print('The following paths were expected but not found in cache:', ', '.join(checksums.keys()))
