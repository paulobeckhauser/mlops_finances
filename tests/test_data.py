import os
from torch.utils.data import Dataset

def test_raw_data():
    """Test that the raw data directory contains exactly two non-empty files"""
    raw_data_dir = "data/raw"

    # Get all files in the directory
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv") and os.path.isfile(os.path.join(raw_data_dir, f))]

    # Check that there are exactly two files
    assert len(csv_files) == 2, f"Expected 2 .csv files, but found {len(csv_files)} in {raw_data_dir}"

    # Check that none of the .csv files are empty
    for file in csv_files:
        file_path = os.path.join(raw_data_dir, file)
        assert os.path.getsize(file_path) > 0, f"The file {file} is empty"
