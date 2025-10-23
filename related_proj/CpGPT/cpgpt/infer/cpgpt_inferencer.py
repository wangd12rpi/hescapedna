from pathlib import Path

import boto3
import botocore
import hydra
import torch
from omegaconf import OmegaConf

from cpgpt.log.utils import get_class_logger
from cpgpt.model.cpgpt_module import CpGPTLitModule


class CpGPTInferencer:
    """A class for performing inference with CpGPT models.

    This class provides functionality to load CpGPT models, process input data,
    and perform inference on methylation data. It handles device management,
    model loading, and data processing for efficient inference.

    Attributes:
        logger: A logger instance for the class.
        device: The device (CPU or CUDA) to be used for computations.
        dependencies_dir: Directory for model dependencies.
        data_dir: Directory for datasets.
        available_models: List of available model names.
        available_datasets: List of available GSE datasets.

    """

    def __init__(self, dependencies_dir: str = "dependencies", data_dir: str = "data") -> None:
        """Initialize the CpGPTInferencer.

        Sets up logging and determines the appropriate device (CPU/CUDA) for computations.

        Args:
            dependencies_dir: Path to the dependencies directory.
            data_dir: Path to the data directory.

        """
        self.logger = get_class_logger(self.__class__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dependencies_dir = dependencies_dir
        self.data_dir = data_dir
        self.available_models = []
        self.available_datasets = []

        self.logger.info(f"Using device: {self.device}.")
        self.logger.info(f"Using dependencies directory: {self.dependencies_dir}")
        self.logger.info(f"Using data directory: {self.data_dir}")
        if self.device == "cpu":
            self.logger.warning("Using CPU for inference. This may be slow.")

        # Initialize S3 client
        try:
            self.s3_client = boto3.client("s3")
            self.s3_resource = boto3.resource("s3")
            self.bucket_name = "cpgpt-lucascamillo-public"

            # Get available models
            self._get_available_models()
            if self.available_models:
                examples = (
                    self.available_models[:3]
                    if len(self.available_models) > 3
                    else self.available_models
                )
                self.logger.info(
                    f"There are {len(self.available_models)} CpGPT models available "
                    f"such as {', '.join(examples)}, etc."
                )

            # Get available datasets
            self._get_available_datasets()
            if self.available_datasets:
                examples = (
                    self.available_datasets[:3]
                    if len(self.available_datasets) > 3
                    else self.available_datasets
                )
                self.logger.info(
                    f"There are {len(self.available_datasets)} GSE datasets available "
                    f"such as {', '.join(examples)}, etc."
                )

        except Exception as e:
            self.logger.warning(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
            self.s3_resource = None

    def _get_available_models(self) -> None:
        """Query S3 to get a list of all available models."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="dependencies/model/weights/",
                RequestPayer="requester",
            )

            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".ckpt"):
                    model_name = key.split("/")[-1].replace(".ckpt", "")
                    self.available_models.append(model_name)
        except Exception as e:
            self.logger.warning(f"Failed to get available models: {e}")

    def _get_available_datasets(self) -> None:
        """Query S3 to get a list of all available GSE datasets."""
        try:
            # Initialize pagination parameters
            continuation_token = None
            max_datasets = 10000  # Set a higher limit to avoid truncation
            dataset_count = 0

            while True:
                # Prepare request parameters
                params = {
                    "Bucket": self.bucket_name,
                    "Prefix": "data/cpgcorpus/raw/",
                    "Delimiter": "/",
                    "RequestPayer": "requester",
                    "MaxKeys": 1000,  # Maximum allowed by S3 API
                }

                # Add continuation token if we're paginating
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                # Make the request
                response = self.s3_client.list_objects_v2(**params)

                # Process the results
                for prefix in response.get("CommonPrefixes", []):
                    prefix_name = prefix.get("Prefix", "")
                    if prefix_name:
                        # Extract GSE ID from the prefix
                        gse_id = prefix_name.strip("/").split("/")[-1]
                        if gse_id.startswith("GSE"):
                            self.available_datasets.append(gse_id)
                            dataset_count += 1

                            # Check if we've reached our limit
                            if dataset_count >= max_datasets:
                                self.logger.warning(
                                    f"Reached maximum dataset count ({max_datasets}). "
                                    "There may be more datasets available."
                                )
                                return

                # Check if there are more results to fetch
                if not response.get("IsTruncated"):
                    break

                # Get the continuation token for the next request
                continuation_token = response.get("NextContinuationToken")

        except Exception as e:
            self.logger.warning(f"Failed to get available datasets: {e}")

    def download_model(
        self,
        model_name: str = "small",
        overwrite: bool = False,
        dependencies_dir: str | None = None,
    ) -> None:
        """Download a CpGPT model from the S3 bucket.

        Args:
            model_name: Name of the model to download.
            overwrite: Whether to overwrite existing files.
            dependencies_dir: Custom dependencies directory. If None, uses the instance's
                dependencies_dir.

        Returns:
            bool: True if download was successful, False otherwise.

        Raises:
            FileNotFoundError: If the model checkpoint is not available.
            ConnectionError: If boto3 is not properly configured.

        """
        if self.s3_client is None:
            self.logger.error(
                "S3 client is not initialized. Make sure boto3 is installed and AWS "
                "credentials are configured."
            )
            msg = "S3 client is not initialized"
            raise ConnectionError(msg)

        # Use custom directory if provided, otherwise use instance's directory
        target_dir = dependencies_dir if dependencies_dir is not None else self.dependencies_dir
        if dependencies_dir is not None:
            self.logger.warning(
                f"Using custom dependencies directory: {dependencies_dir} (overrides default)"
            )

        # Create directories if they don't exist
        weights_dir = Path(f"{target_dir}/model/weights")
        config_dir = Path(f"{target_dir}/model/config")
        vocab_dir = Path(f"{target_dir}/model/vocab")

        weights_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        vocab_dir.mkdir(parents=True, exist_ok=True)

        # Define paths
        s3_model_key = f"dependencies/model/weights/{model_name}.ckpt"
        s3_config_key = f"dependencies/model/config/{model_name}.yaml"
        s3_vocab_key = f"dependencies/model/vocab/{model_name}.json"

        local_model_path = f"{target_dir}/model/weights/{model_name}.ckpt"
        local_config_path = f"{target_dir}/model/config/{model_name}.yaml"
        local_vocab_path = f"{target_dir}/model/vocab/{model_name}.json"

        # Check if the model exists in S3
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name, Key=s3_model_key, RequestPayer="requester"
            )
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Use already fetched available models if possible
                if not self.available_models:
                    self._get_available_models()

                error_msg = (
                    f"Model '{model_name}' does not have a checkpoint available. "
                    f"Available options: {', '.join(self.available_models)}."
                )
                self.logger.exception(error_msg)
                raise FileNotFoundError(error_msg)
            self.logger.exception(f"Error checking model existence: {e}")
            raise

        # Download model checkpoint if it doesn't exist or overwrite is True
        if not Path(local_model_path).exists() or overwrite:
            self.logger.info(f"Downloading model checkpoint to {local_model_path}.")
            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_model_key,
                    local_model_path,
                    ExtraArgs={"RequestPayer": "requester"},
                )
            except Exception as e:
                self.logger.exception(f"Failed to download model checkpoint: {e}")
                raise
        else:
            self.logger.info(
                f"Model checkpoint already exists at {local_model_path} (skipping download)."
            )

        # Download model config if it doesn't exist or overwrite is True
        if not Path(local_config_path).exists() or overwrite:
            self.logger.info(f"Downloading model config to {local_config_path}")
            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_config_key,
                    local_config_path,
                    ExtraArgs={"RequestPayer": "requester"},
                )
            except Exception as e:
                self.logger.exception(f"Failed to download model config: {e}")
                raise
        else:
            self.logger.info(
                f"Model config already exists at {local_config_path} (skipping download)."
            )

        # Check if vocab file exists and download it if needed
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name, Key=s3_vocab_key, RequestPayer="requester"
            )
            if not Path(local_vocab_path).exists() or overwrite:
                self.logger.info(f"Downloading model vocabulary to {local_vocab_path}")
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_vocab_key,
                    local_vocab_path,
                    ExtraArgs={"RequestPayer": "requester"},
                )
            else:
                self.logger.info(
                    f"Model vocabulary already exists at {local_vocab_path} (skipping download)."
                )
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.logger.warning(f"No vocabulary file found for model '{model_name}'.")
            else:
                self.logger.exception(f"Error checking vocabulary file: {e}")

        self.logger.info(f"Successfully downloaded model '{model_name}'.")

    def download_dependencies(
        self, species: str = "human", overwrite: bool = False, dependencies_dir: str | None = None
    ) -> None:
        """Download dependencies for a specific species.

        Args:
            species: Species to download dependencies for. Options: "human" or "mammalian".
            overwrite: Whether to overwrite existing files.
            dependencies_dir: Custom dependencies directory. If None, uses the instance's
                dependencies_dir.

        Returns:
            bool: True if download was successful, False otherwise.

        Raises:
            ValueError: If the species is not supported.
            ConnectionError: If boto3 is not properly configured.

        """
        if self.s3_client is None:
            self.logger.error(
                "S3 client is not initialized. Make sure boto3 is installed and "
                "AWS credentials are configured."
            )
            msg = "S3 client is not initialized"
            raise ConnectionError(msg)

        # Use custom directory if provided, otherwise use instance's directory
        target_dir_base = (
            dependencies_dir if dependencies_dir is not None else self.dependencies_dir
        )
        if dependencies_dir is not None:
            self.logger.warning(
                f"Using custom dependencies directory: {dependencies_dir} (overrides default)"
            )

        if species not in ["human", "mammalian"]:
            error_msg = (
                f"Species '{species}' is not supported. Available options: 'human', 'mammalian'"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Create directory if it doesn't exist
        target_dir = Path(f"{target_dir_base}/{species}")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download the dependencies
        s3_prefix = f"dependencies/{species}/"
        local_path = f"{target_dir_base}/{species}"

        # Get list of all files that should be downloaded from S3
        expected_files = []
        missing_files = []

        try:
            # List all objects in S3 to get expected files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=s3_prefix, RequestPayer="requester"
            )

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix) :]
                    if not relative_path:  # Skip directory markers
                        continue
                    expected_files.append(relative_path)

                    # Check if local file exists
                    local_file_path = Path(f"{local_path}/{relative_path}")
                    if not local_file_path.exists():
                        missing_files.append(relative_path)
        except Exception as e:
            self.logger.exception(f"Failed to list S3 objects: {e}")
            raise

        # Check if we should skip download or proceed
        if not overwrite and expected_files:
            if not missing_files:
                self.logger.info(
                    f"All {len(expected_files)} dependency files for {species} already exist "
                    f"at {target_dir}. Skipping download."
                )
                return
            else:
                self.logger.warning(
                    f"Dependencies directory exists but {len(missing_files)} out of "
                    f"{len(expected_files)} files are missing. Proceeding with download."
                )

        self.logger.info(f"Downloading {species} dependencies to {local_path}.")

        try:
            # Download all files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=s3_prefix, RequestPayer="requester"
            )

            files_downloaded = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    # Get the relative path
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix) :]
                    if not relative_path:  # Skip directory markers
                        continue

                    # Create local directories if needed
                    local_file_path = Path(f"{local_path}/{relative_path}")
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download file if it doesn't exist or overwrite is True
                    if not local_file_path.exists() or overwrite:
                        self.s3_client.download_file(
                            self.bucket_name,
                            key,
                            str(local_file_path),
                            ExtraArgs={"RequestPayer": "requester"},
                        )
                        files_downloaded += 1

            self.logger.info(f"Downloaded {files_downloaded} files for {species} dependencies.")
        except Exception as e:
            self.logger.exception(f"Failed to download dependencies: {e}")
            raise

        self.logger.info(f"Successfully downloaded {species} dependencies.")
        return

    def download_cpgcorpus_dataset(
        self, gse_id: str, overwrite: bool = False, data_dir: str | None = None
    ) -> None:
        """Download a dataset from CpGCorpus.

        Args:
            gse_id: GSE ID of the dataset to download.
            overwrite: Whether to overwrite existing files.
            data_dir: Custom data directory. If None, uses the instance's data_dir.

        Returns:
            bool: True if download was successful, False otherwise.

        Raises:
            FileNotFoundError: If the GSE ID is not available.
            ConnectionError: If boto3 is not properly configured.

        """
        if self.s3_client is None:
            self.logger.error(
                "S3 client is not initialized. Make sure boto3 is installed and "
                "AWS credentials are configured."
            )
            msg = "S3 client is not initialized"
            raise ConnectionError(msg)

        # Use custom directory if provided, otherwise use instance's directory
        target_dir_base = data_dir if data_dir is not None else self.data_dir
        if data_dir is not None:
            self.logger.warning(f"Using custom data directory: {data_dir} (overrides default)")

        # Check if GSE ID exists in S3
        s3_prefix = f"data/cpgcorpus/raw/{gse_id}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_prefix, MaxKeys=1, RequestPayer="requester"
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                # Use already fetched available datasets if possible
                if not self.available_datasets:
                    self._get_available_datasets()

                error_msg = (
                    f"GSE ID '{gse_id}' is not available in CpGCorpus. Available options "
                    f"include: {', '.join(self.available_datasets[:5])}..."
                )
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        except Exception as e:
            self.logger.exception(f"Failed to check GSE ID: {e}")
            raise

        # Create directory if it doesn't exist
        target_dir = Path(f"{target_dir_base}/cpgcorpus/raw/{gse_id}")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download the dataset
        local_path = f"{target_dir_base}/cpgcorpus/raw/{gse_id}"

        # Get list of all files that should be downloaded from S3
        expected_files = []
        missing_files = []

        try:
            # List all objects in S3 to get expected files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=s3_prefix, RequestPayer="requester"
            )

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix) :]
                    if not relative_path:  # Skip directory markers
                        continue
                    expected_files.append(relative_path)

                    # Check if local file exists
                    local_file_path = Path(f"{local_path}/{relative_path}")
                    if not local_file_path.exists():
                        missing_files.append(relative_path)
        except Exception as e:
            self.logger.exception(f"Failed to list S3 objects for dataset {gse_id}: {e}")
            raise

        # Check if we should skip download or proceed
        if not overwrite and expected_files:
            if not missing_files:
                self.logger.info(
                    f"All {len(expected_files)} files for dataset {gse_id} already exist "
                    f"at {target_dir}. Skipping download."
                )
                return
            else:
                self.logger.warning(
                    f"Dataset directory exists but {len(missing_files)} out of "
                    f"{len(expected_files)} files are missing. Proceeding with download."
                )

        self.logger.info(f"Downloading dataset {gse_id} to {local_path}.")

        try:
            # Download all files
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=s3_prefix, RequestPayer="requester"
            )

            files_downloaded = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    # Get the relative path
                    key = obj["Key"]
                    relative_path = key[len(s3_prefix) :]
                    if not relative_path:  # Skip directory markers
                        continue

                    # Create local directories if needed
                    local_file_path = Path(f"{local_path}/{relative_path}")
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download file if it doesn't exist or overwrite is True
                    if not local_file_path.exists() or overwrite:
                        self.s3_client.download_file(
                            self.bucket_name,
                            key,
                            str(local_file_path),
                            ExtraArgs={"RequestPayer": "requester"},
                        )
                        files_downloaded += 1

            self.logger.info(f"Downloaded {files_downloaded} files for dataset {gse_id}.")
        except Exception as e:
            self.logger.exception(f"Failed to download dataset: {e}")
            raise

        self.logger.info(f"Successfully downloaded dataset {gse_id}.")
        return

    def load_cpgpt_config(
        self,
        config_path: str,
    ) -> OmegaConf:
        """Load a yaml file containing the configuration for a CpGPT model.

        Args:
            config_path (str): Path to the yaml configuration file.

        Returns:
            OmegaConf: An omega dictionary with the model configuration.

        """
        config = OmegaConf.load(config_path)
        self.logger.info("Loaded CpGPT model config.")

        return config

    def load_cpgpt_model(
        self,
        config: OmegaConf,
        model_ckpt_path: str | None = None,
        strict_load: bool = True,
    ) -> CpGPTLitModule:
        """Load a CpGPT model from a checkpoint file and return the model.

        If no checkpoint path is provided, the model will be returned
        with randomly initialized weights.

        Args:
            config (OmegaConf): Hydra config containing the model definition.
            model_ckpt_path (str, optional): Path to the checkpoint file. If not provided,
                random initialization is used.
            strict_load (bool, optional): If True, requires exact key matching
                when loading the checkpoint.

        Returns:
            CpGPTLitModule: The instantiated (and optionally checkpoint-loaded) model.

        """
        # Instantiate the model
        model: CpGPTLitModule = hydra.utils.instantiate(config.model)
        self.logger.info("Instantiated CpGPT model from config.")

        # Load to device
        model.to(self.device)
        self.logger.info(f"Using device: {self.device}.")

        # Load checkpoint if a valid path is provided
        if model_ckpt_path is not None:
            ckpt_path_obj = Path(model_ckpt_path)
            if not ckpt_path_obj.exists():
                msg = f"Checkpoint file not found: {model_ckpt_path}"
                self.logger.error(msg)
                raise FileNotFoundError(msg)

            self.logger.info(f"Loading checkpoint from: {model_ckpt_path}")
            checkpoint = torch.load(ckpt_path_obj, map_location=self.device, weights_only=False)
            state_dict = checkpoint["state_dict"]
            cleaned_state_dict = {
                k.replace("net._orig_mod.", "net."): v for k, v in state_dict.items()
            }
            model.load_state_dict(cleaned_state_dict, strict=strict_load)
            self.logger.info("Checkpoint loaded into the model.")
        else:
            self.logger.info("No checkpoint path provided; using random initialization.")

        return model
