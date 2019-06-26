import hashlib


READ_BUFFER_SIZE = 2**20


def compute_checksum(fp):
    checksum = hashlib.md5()
    for chunk in iter(lambda: fp.read(READ_BUFFER_SIZE), b''):
        checksum.update(chunk)
    checksum = checksum.hexdigest()
    return checksum
