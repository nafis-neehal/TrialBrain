# base_data_loader.py

import requests
import yaml
import pandas as pd
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv


class BaseDataLoader(ABC):
    def __init__(self, config_path: str = 'config.yml'):
        """
        Initialize the base data loader with configuration.

        Args:
            config_path (str): Path to the configuration file
        """
        load_dotenv()
        self.configs = self._load_config(config_path)
        self._setup_logging()
        # Initialize task_name to None - should be set by child classes
        self.task_name = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        log_path = f"{self.configs['paths']['base_path']}/{self.configs['paths']['log_path']['data_loader']}"

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        logging.basicConfig(
            filename=f"{log_path}/data_loader.log",
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S',
            encoding='utf-8',
            level=logging.INFO,
            force=True
        )

    def _concat_fields(self, fields: List[str]) -> str:
        """Concatenate API fields with pipe separator."""
        return '|'.join(fields)

    def fetch_data(self, **kwargs) -> Dict:
        """
        Fetch data from the Clinical Trials API.

        Args:
            **kwargs: Additional parameters for the API call

        Returns:
            Dict: JSON response from the API
        """
        base_url = self.configs['data_loader']['base_url']
        query_params = self._build_query_params(**kwargs)

        response = requests.get(base_url, params=query_params)

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error fetching data, status code: {response.status_code}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def fetch_by_id(self, nct_id: str) -> Dict:
        """
        Fetch trial data by NCT ID.

        Args:
            nct_id (str): The NCT ID of the trial

        Returns:
            Dict: JSON response from the API
        """
        base_url = f"{self.configs['data_loader']['base_url']}/{nct_id}"
        query_params = {
            'fields': self._concat_fields(self.get_required_fields())
        }

        response = requests.get(base_url, params=query_params)

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error fetching data for NCT ID {nct_id}, status code: {response.status_code}"
            logging.error(error_msg)
            raise Exception(error_msg)

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for the specific task."""
        pass

    @abstractmethod
    def _build_query_params(self, **kwargs) -> Dict:
        """Build query parameters for the API call."""
        pass

    @abstractmethod
    def parse_data_to_df(self, data: Dict) -> pd.DataFrame:
        """Parse API response into a DataFrame."""
        pass

    def run_data_download(self) -> None:
        """Run the complete data download process."""
        data = self.fetch_data()
        data_df = self.parse_data_to_df(data)

        iter_count = 1
        while 'nextPageToken' in data:
            data = self.fetch_data(nextPageToken=data['nextPageToken'])
            data_df = pd.concat(
                [data_df, self.parse_data_to_df(data)], ignore_index=True)
            print(f'Iteration: {iter_count}')
            iter_count += 1

        print(f"Total number of trials fetched: {data_df.shape[0]}")
        logging.info(f"Total number of trials fetched: {data_df.shape[0]}")

        self._save_data(data_df)

    def _save_data(self, data_df: pd.DataFrame) -> None:
        """Save the DataFrame to a CSV file."""
        if not self.task_name:
            raise ValueError("task_name must be set in the child class")

        # Get the task type from configs
        task_type = self.configs['tasks'][self.task_name]

        # Build the save directory path
        save_dir = os.path.join(
            self.configs['paths']['base_path'],
            self.configs['paths']['data_path'][task_type]
        )

        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the file
        file_name = self._get_output_filename()
        file_path = os.path.join(save_dir, file_name)

        data_df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")

    @abstractmethod
    def _get_output_filename(self) -> str:
        """Return the output filename for the specific task."""
        pass

    def _split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        if not self.task_name:
            raise ValueError("task_name must be set in the child class")

        # Get the task type from configs
        task_type = self.configs['tasks'][self.task_name]

        # Build the paths
        test_ids_path = os.path.join(
            self.configs['paths']['base_path'],
            self.configs['paths']['data_path']['raw'],
            'CTRepo_IDs.csv'
        )

        data_path = os.path.join(
            self.configs['paths']['base_path'],
            self.configs['paths']['data_path'][task_type],
            self._get_output_filename()
        )

        # Load the data
        test_ids = pd.read_csv(test_ids_path)
        all_trials = pd.read_csv(data_path)

        # Split the data
        train_data = all_trials[~all_trials['NCTId'].isin(test_ids['NCTId'])]
        test_data = all_trials[all_trials['NCTId'].isin(test_ids['NCTId'])]

        return train_data, test_data

    def push_to_hf_hub(self, dataset_name: str, private: bool = True) -> None:
        """
        Push the dataset to Hugging Face Hub.

        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            private (bool): Whether to make the dataset private
        """
        login(token=os.getenv('HF_TOKEN'))

        train_data, test_data = self._split_train_test()
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_data),
            'test': Dataset.from_pandas(test_data)
        })

        dataset_dict.push_to_hub(dataset_name, private=private)
