import os
import re
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _extract_file_id(url: str) -> str:
        """
        Pull the Google Drive file ID out of a share link.
        Supports:
          https://drive.google.com/file/d/<ID>/view?usp=sharing
          https://drive.google.com/uc?id=<ID>
        """
        if "/folders/" in url:
            raise ValueError(
                "source_URL is a Google Drive FOLDER link. "
                "Upload the dataset as a single .zip file and use its file link "
                "(https://drive.google.com/file/d/<ID>/view?usp=sharing)."
            )

        match = (re.search(r"/d/([a-zA-Z0-9_-]+)", url)
                 or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url))
        if not match:
            raise ValueError(f"Could not find a file ID in URL: {url}")
        return match.group(1)

    def download_file(self) -> str:
        """
        Fetch the zip file from Google Drive
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_path = str(self.config.local_data_file)
            os.makedirs(self.config.root_dir, exist_ok=True)

            # A manually copied dataset is ready to use; do not redownload it.
            if self.config.dataset_dir.is_dir():
                logger.info(f"Using local dataset at {self.config.dataset_dir}")
                return str(self.config.dataset_dir)

            # Remove a stale/partial file from a previous failed run
            if os.path.exists(zip_download_path) and not zipfile.is_zipfile(zip_download_path):
                logger.info(f"Removing invalid existing file: {zip_download_path}")
                os.remove(zip_download_path)

            # Skip download if a valid zip is already there
            if os.path.exists(zip_download_path):
                logger.info(f"File already exists ({get_size(self.config.local_data_file)}), skipping download")
                return zip_download_path

            file_id = self._extract_file_id(dataset_url)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_path}")

            result = gdown.download(id=file_id, output=zip_download_path, quiet=False)
            if result is None or not os.path.exists(zip_download_path):
                raise RuntimeError("gdown did not download the file. Check sharing permissions.")

            if not zipfile.is_zipfile(zip_download_path):
                raise RuntimeError(
                    "Downloaded file is not a valid zip. Make sure the Drive link points to a .zip file."
                )

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_path}")
            return zip_download_path

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the unzip directory.
        Returns None
        """
        unzip_path = str(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)
        if self.config.dataset_dir.is_dir():
            logger.info(f"Dataset is already extracted at {self.config.dataset_dir}; skipping extraction.")
            return
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file into {unzip_path}")
