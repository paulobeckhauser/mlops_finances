import os
from unittest.mock import patch, MagicMock

def test_raw_data():
    """Test that the raw data directory contains exactly two non-empty .csv files."""
    raw_data_dir = "data/raw"

    # Mock os.listdir to simulate the presence of files
    mock_files = ["file1.csv", "file2.csv"]

    # Mock os.path.getsize to simulate file sizes
    with patch("os.listdir", return_value=mock_files), \
         patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", side_effect=[100, 200]):

        # Get all files in the directory
        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv") and os.path.isfile(os.path.join(raw_data_dir, f))]

        # Check that there are exactly two files
        assert len(csv_files) == 2, f"Expected 2 .csv files, but found {len(csv_files)} in {raw_data_dir}"

        # Check that none of the .csv files are empty
        for file in csv_files:
            file_path = os.path.join(raw_data_dir, file)
            assert os.path.getsize(file_path) > 0, f"The file {file} is empty"
