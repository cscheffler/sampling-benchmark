source venv/bin/activate

# Create data directory structure
cd /Users/carl/work/estimating-evidence/tmp/lisa/data/openml/datasets
mkdir raw
mkdir one-hot
mkdir standardized
mkdir robust_standardized
mkdir whitened
mkdir errors

cd phases01_data_and_posteriors/

# Download data from OpenML
# The openml Python library creates a cache of about 82G in ~/.openml/
# The pickles stored by this script amount to about 32G for a total of
# about 114G.
python data_scripts/download_datasets.py

# Delete broken and empty data sets.
# This script will also try to redownload any broken or empty data sets, which
# is probably not a bad idea - just in case something downloaded incorrectly
# from OpenML.
python data_scripts/purge_bad_datasets.py

# Reads raw data and computes various derived versions (one hot, normalized, etc.).
# Uses a huge amount of disk space.
# I got out of memory errors for i in [432, 433] on a 16 GB MacBook Pro
python data_scripts/preprocess_datasets.py

