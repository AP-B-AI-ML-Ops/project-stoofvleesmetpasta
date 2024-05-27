import os
from unittest.mock import MagicMock, patch

import pytest
from load.collect import collect_flow
from prefect.testing.utilities import prefect_test_harness


# Ensure the ./data directory is cleaned up before and after tests
@pytest.fixture(autouse=True)
def cleanup_data_directory():
    if os.path.exists("./data"):
        for root, dirs, files in os.walk("./data", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("./data")
    yield
    if os.path.exists("./data"):
        for root, dirs, files in os.walk("./data", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("./data")


def test_collect_flow():
    with prefect_test_harness():
        with patch("load.collect.KaggleApi") as MockKaggleApi:
            # Mock the KaggleApi instance and its methods
            mock_api = MockKaggleApi.return_value
            mock_api.dataset_download_files = MagicMock()

            # Run the flow
            collect_flow(update=True)

            # Assertions to ensure the API methods were called
            mock_api.authenticate.assert_called_once()
            mock_api.dataset_download_files.assert_called_once_with(
                "uciml/student-alcohol-consumption", path="./data", unzip=True
            )


if __name__ == "__main__":
    pytest.main()
