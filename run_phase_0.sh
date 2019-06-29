source venv/bin/activate

cd phases01_data_and_posteriors/

# Download data from OpenML
# The openml Python library creates a cache of about 82G in ~/.openml/
# The pickles stored by this script amount to about 32G for a total of
# about 114G.
python data_scripts/download_datasets.py

# Test that the OpenML cache of downloaded files matches what we expect.
python test_scripts/checksum_openml_cache.py

# Delete broken and empty data sets.
# This script will also try to redownload any broken or empty data sets, which
# is probably not a bad idea - just in case something downloaded incorrectly
# from OpenML.
python data_scripts/purge_bad_datasets.py

# Test that the cleaned downloaded data matches what we expect.
python test_scripts/checksum_raw_data.py

# Reads raw data and computes various derived versions (one hot, normalized, etc.).
# Uses a huge amount of disk space.
# I got out of memory errors for i in [432, 433] on a 16 GB MacBook Pro
python data_scripts/preprocess_datasets.py

# Test that the preprocesed data matches what we expect.
python test_scripts/checksum_preprocessed_data.py
