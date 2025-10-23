import gc
from pathlib import Path

import numpy as np
import pandas as pd
import sqlitedict
from tqdm.rich import tqdm

from cpgpt.log.utils import get_class_logger

from .dna_llm_embedder import DNALLMEmbedder
from .illumina_methylation_prober import IlluminaMethylationProber


class CpGPTDataSaver:
    """A class for processing and saving methylation data.

    This class handles the loading, processing, and storage of methylation data from
    multiple input files. It uses IlluminaMethylationProber and DNALLMEmbedder for
    processing probe IDs and genomic locations. Please ensure a column "species" is
    present in the input data files with the scientific name of the species.

    Attributes:
        data_paths (List[str]): Paths to the input data files.
        processed_dir (str): Directory to store processed data.
        dataset_metrics (Dict[str, Dict]): Metrics for each processed dataset.
        all_genomic_locations (Set[str]): Set of all genomic locations processed.

    """

    def __init__(
        self,
        data_paths: str | list[str] | Path | list[Path],
        processed_dir: str | Path = "tutorial_processed",
        metadata_cols: str | list[str] | None = None,
        data_cols: str | list[str] | None = None,
    ) -> None:
        """Initialize the CpGPTDataSaver.

        Args:
            data_paths (Union[str | List[str] | Path | List[Path]]): Path(s) to the input
                data file(s).
            processed_dir (Union[str | Path]): Directory to store processed data.
            metadata_cols (List[str]): Name of the columns that contain metadata to be
                saved. Defaults to None.
            data_cols (Union[str, List[str]]): Name of the data columns to keep. If None,
                all non-metadata columns are kept. Defaults to None.

        Raises:
            ValueError: If input parameters are invalid.
            FileNotFoundError: If a specified file does not exist.

        """
        # Convert data_paths to list of strings
        data_paths = [data_paths] if isinstance(data_paths, str | Path) else data_paths
        data_paths = [str(path) for path in data_paths]

        for path in data_paths:
            if not isinstance(path, str):
                msg = f"data_path must be a string or Path, got {type(path)}."
                raise TypeError(msg)
            if not path.endswith(".arrow") and not path.endswith(".feather"):
                msg = f"File {path} is not a .arrow or .feather file."
                raise ValueError(msg)
            if not Path(path).exists():
                msg = f"File {path} does not exist."
                raise FileNotFoundError(msg)

        # Convert processed_dir to string
        processed_dir = str(processed_dir)
        if not isinstance(processed_dir, str):
            msg = f"processed_dir must be a string or Path, got {type(processed_dir)}."
            raise TypeError(msg)

        metadata_cols = [metadata_cols] if isinstance(metadata_cols, str) else metadata_cols
        if not isinstance(metadata_cols, (type(None) | str | list)):
            msg = (
                "metadata_cols must be a string, a list of strings, or None, "
                f"got {type(metadata_cols)}."
            )
            raise TypeError(msg)

        data_cols = [data_cols] if isinstance(data_cols, str) else data_cols
        if not isinstance(data_cols, (type(None) | str | list)):
            msg = f"data_cols must be a string, a list of strings, or None, got {type(data_cols)}."
            raise TypeError(msg)

        self.data_paths = data_paths
        self.processed_dir = processed_dir
        self.metadata_cols = metadata_cols
        self.data_cols = data_cols
        self.dataset_metrics: dict[str, dict] = {}
        self.all_genomic_locations: dict[str, set[str]] = {}

        self.logger = get_class_logger(self.__class__)

        self._create_directories()
        self._load_dataset_metrics()
        self._load_genomic_locations()

    def _create_directories(self) -> None:
        """Create necessary directories for storing data.

        Creates:
            - processed_dir: Directory for storing processed dataset files
        """
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Dataset folders will be stored under {self.processed_dir}.")

    def _load_dataset_metrics(self) -> None:
        """Load existing dataset metrics if available."""
        metrics_file = Path(self.processed_dir) / "dataset_metrics.db"
        if metrics_file.exists():
            with sqlitedict.SqliteDict(metrics_file, autocommit=True) as db:
                self.dataset_metrics = dict(db)
            self.logger.info("Loaded existing dataset metrics.")
        else:
            self.logger.info("No existing dataset metrics found. Please process files.")

    def _load_genomic_locations(self) -> None:
        """Load existing genomic locations if available."""
        genomic_locations_file = Path(self.processed_dir) / "genomic_locations.db"
        if genomic_locations_file.exists():
            with sqlitedict.SqliteDict(genomic_locations_file, autocommit=True) as db:
                self.all_genomic_locations = {
                    species: set(locations) for species, locations in db.items()
                }
            self.logger.info("Loaded existing genomic locations.")
        else:
            self.logger.info("No existing genomic locations found. Please process files.")

    def process_files(
        self,
        prober: IlluminaMethylationProber,
        embedder: DNALLMEmbedder,
        check_methylation_pattern: bool = False,
    ) -> None:
        """Process all data files.

        Processes each data file by converting probe IDs to genomic locations,
        validating methylation patterns, and saving processed data.

        Args:
            prober (IlluminaMethylationProber): Prober for methylation data
            embedder (DNALLMEmbedder): Embedder for DNA sequences
            check_methylation_pattern (bool, optional): Whether to validate methylation patterns.
                Defaults to False

        """
        self.logger.info("Starting file processing.")

        # Filter out already processed files
        data_paths_to_process = [f for f in self.data_paths if f not in self.dataset_metrics]
        if len(data_paths_to_process) < len(self.data_paths):
            self.logger.info(
                f"{len(self.data_paths) - len(data_paths_to_process)} files "
                "already processed. Skipping those.",
            )

        if len(data_paths_to_process) > 0:
            progress_bar = tqdm(data_paths_to_process, desc="Processing files", leave=False)
            for data_path in progress_bar:
                filename = Path(data_path).name
                progress_bar.set_description(f"Processing: {filename}")
                self._process_single_file(data_path, prober, embedder, check_methylation_pattern)

            self.logger.info("File processing completed.")

    def _process_single_file(
        self,
        data_path: str,
        prober: IlluminaMethylationProber,
        embedder: DNALLMEmbedder,
        check_methylation_pattern: bool = False,
    ) -> None:
        """Process a single data file.

        Reads data file, processes features, validates methylation patterns if requested,
        and saves processed data.

        Args:
            data_path (str): Path to the data file
            prober (IlluminaMethylationProber): Prober for methylation data
            embedder (DNALLMEmbedder): Embedder for DNA sequences
            check_methylation_pattern (bool, optional): Whether to validate methylation patterns.
                Defaults to False

        """
        self.logger.debug(f"Processing file: {data_path}.")

        dataset_name = str(Path(data_path).with_suffix("")).replace("/", "_").replace("\\", "_")
        dataset_dir = Path(self.processed_dir) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Read feather or arrow file
        input_df = pd.read_feather(data_path)

        # Process features and get genomic locations
        final_df, metadata_df, genomic_locations, species, metrics = self._process_features(
            input_df,
            prober,
            embedder,
        )

        # If final_df is empty, skip this file
        if final_df.empty or len(final_df.columns) == 0:
            self.logger.warning(f"DataFrame is empty for {dataset_name}. Skipping.")
            return

        # Perform methylation-specific validations
        if check_methylation_pattern and not self._validate_methylation_data(
            final_df.values,
            dataset_name,
        ):
            return

        # Process genomic locations
        chrom_index_pos_list = [
            self._parse_genomic_location(loc, species, embedder) for loc in genomic_locations
        ]

        # If the chrom_index_pos_list is empty, raise a warning and return without saving
        if not chrom_index_pos_list:
            self.logger.warning(f"No genomic locations found for {dataset_name}. Skipping.")
            return

        # Save processed data
        self._save_processed_data(
            dataset_dir,
            final_df,
            metadata_df,
            chrom_index_pos_list,
            species,
        )

        # Update dataset metrics
        self._update_dataset_metrics(data_path, final_df, metadata_df, species, metrics)

        self.logger.debug(f"Processed {dataset_name} successfully.")

    def _process_metadata(
        self,
        input_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Process and extract metadata from the input DataFrame.

        Args:
            input_df (pd.DataFrame): Input DataFrame containing methylation data and metadata

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Tuple containing:
                - DataFrame with metadata columns removed
                - DataFrame containing only metadata columns, or None if no metadata

        Raises:
            ValueError: If metadata columns are not numeric

        """
        if self.metadata_cols and all(col in input_df.columns for col in self.metadata_cols):
            metadata_df = input_df[self.metadata_cols]
            input_df = input_df.drop(columns=self.metadata_cols)
            try:
                metadata_df = metadata_df.astype(float)
            except ValueError:
                self.logger.exception("Metadata dtype is not numeric. Failed: {e}")
            self.logger.debug(f"Extracted metadata columns: {self.metadata_cols}")
        elif self.metadata_cols and any(col not in input_df.columns for col in self.metadata_cols):
            metadata_df = None
            self.logger.warning(
                "One or many metadata columns not found in the DataFrame. "
                "Setting metadata_df to None.",
            )
        else:
            metadata_df = None
            self.logger.debug("No metadata column declared.")

        return input_df, metadata_df

    def _process_species(
        self,
        betas_df: pd.DataFrame,
        embedder: DNALLMEmbedder,
    ) -> tuple[pd.DataFrame, str | None]:
        """Process and extract species information.

        Args:
            betas_df (pd.DataFrame): Input DataFrame with possible species column
            embedder (DNALLMEmbedder): Embedder for DNA sequences

        Returns:
            Tuple[pd.DataFrame, Optional[str]]: Tuple containing:
                - DataFrame with species column removed
                - Species name string, or None if multiple species found

        Note:
            Defaults to "homo_sapiens" if no species column found

        """
        if "species" in betas_df.columns:
            unique_species = betas_df["species"].unique()
            if len(unique_species) > 1:
                self.logger.warning(
                    f"Found multiple species in dataset: {unique_species}. "
                    "Each dataset must contain only one species. Skipping this dataset.",
                )
                betas_df = betas_df.drop(columns=["species"])
                return betas_df, None

            species = unique_species[0]
            betas_df = betas_df.drop(columns=["species"])
            self.logger.debug(f"Detected species: {species}.")
            if species not in embedder.ensembl_metadata_dict:
                self.logger.error(
                    f"Detected species {species} is not present in the embedder's "
                    "Ensembl metadata.",
                )
        else:
            species = "homo_sapiens"
            self.logger.warning(f"No species column found. Defaulting to {species}.")

        return betas_df, species

    def _process_probes(
        self,
        betas_df: pd.DataFrame,
        species: str,
        prober: IlluminaMethylationProber,
    ) -> pd.DataFrame:
        """Convert probe IDs to genomic locations.

        Args:
            betas_df (pd.DataFrame): Input DataFrame with probe IDs as columns
            species (str): Species name
            prober (IlluminaMethylationProber): Prober for methylation data

        Returns:
            pd.DataFrame: DataFrame with probe IDs converted to genomic locations

        Note:
            Preserves original column order and non-probe columns

        """
        probe_set = set(prober.illumina_metadata_dict[species].keys())

        # Create a mapping of probe features to genomic locations
        probe_to_genomic = {
            col: prober.illumina_metadata_dict[species][col]
            for col in betas_df.columns
            if col in probe_set
        }

        if not probe_to_genomic:
            self.logger.debug("No probe features found. Returning original DataFrame.")
            return betas_df

        self.logger.debug(f"Found {len(probe_to_genomic)} probe features to convert.")

        # Create a new column names list, replacing probe IDs with genomic locations
        new_columns = [probe_to_genomic.get(col, col) for col in betas_df.columns]
        betas_df.columns = new_columns

        self.logger.debug(
            f"Processed {len(probe_to_genomic)} probe features to genomic locations.",
        )
        self.logger.debug(f"DataFrame shape unchanged: {betas_df.shape}")

        return betas_df

    def _process_genomic_locations(
        self,
        probes_df: pd.DataFrame,
        species: str,
        embedder: DNALLMEmbedder,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate and process genomic location features.

        Args:
            probes_df (pd.DataFrame): Input DataFrame with genomic locations
            species (str): Species name
            embedder (DNALLMEmbedder): Embedder for DNA sequences

        Returns:
            Tuple[pd.DataFrame, List[str]]: Tuple containing:
                - DataFrame with only valid genomic locations
                - List of valid genomic location strings

        Note:
            Filters out invalid genomic locations and logs warnings

        """
        # Check if DataFrame is empty
        if probes_df.empty or len(probes_df.columns) == 0:
            self.logger.warning(
                "Empty DataFrame received. Returning empty DataFrame and empty list.",
            )
            return probes_df, []

        vocab_set = set(embedder.ensembl_metadata_dict[species]["vocab"].keys())

        # Function to validate genomic location
        def is_valid_location(feature: str) -> bool:
            return feature.split(":")[0] in vocab_set

        # Make the index the first column if it's not a genomic location
        if not is_valid_location(probes_df.columns[0]):
            self.logger.debug(f"Setting index to first column: {probes_df.columns[0]}.")
            probes_df = probes_df.set_index(probes_df.columns[0])

        # Identify valid genomic features
        valid_features = [col for col in probes_df.columns if is_valid_location(col)]
        invalid_features = [col for col in probes_df.columns if not is_valid_location(col)]

        if invalid_features:
            self.logger.warning(
                f"Found {len(invalid_features)} invalid genomic locations "
                f"(e.g. {invalid_features[0]}, {invalid_features[-1]}). "
                "These will be excluded.",
            )
            self.logger.debug(f"First few invalid features: {invalid_features[:10]}")
            self.logger.debug(f"Last few invalid features: {invalid_features[-10:]}")

        # Filter DataFrame to keep only valid features
        probes_df = probes_df.loc[:, probes_df.columns.isin(valid_features)]

        return probes_df, valid_features

    def _process_features(
        self,
        input_df: pd.DataFrame,
        prober: IlluminaMethylationProber,
        embedder: DNALLMEmbedder,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], str, dict]:
        """Process all features in the input DataFrame.

        Handles metadata extraction, species identification, probe conversion,
        and genomic location validation.

        Args:
            input_df (pd.DataFrame): Input DataFrame
            prober (IlluminaMethylationProber): Prober for methylation data
            embedder (DNALLMEmbedder): Embedder for DNA sequences

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str], str, Dict]: Tuple containing:
                - Processed DataFrame with methylation data
                - Metadata DataFrame if available, else None
                - List of valid genomic locations
                - Species name
                - Dictionary of processing metrics

        """
        # Filter columns if data_cols is provided
        if self.data_cols is not None:
            # Keep metadata and species columns if they exist
            required_cols = (self.metadata_cols or []) + (
                ["species"] if "species" in input_df.columns else []
            )
            cols_to_keep = list(set(self.data_cols + required_cols))
            [col for col in self.data_cols if col not in input_df.columns]
            keep_mask = input_df.columns.isin(cols_to_keep)
            input_df = input_df.iloc[:, keep_mask]

        # Process metadata
        betas_df, metadata_df = self._process_metadata(input_df)

        # Process species information
        betas_df, species = self._process_species(betas_df, embedder)

        # Process probes
        df_probes = self._process_probes(betas_df, species, prober)

        # Process genomic locations
        df_genomic_locations, genomic_locations = self._process_genomic_locations(
            df_probes,
            species,
            embedder,
        )

        # Combine metrics
        metrics = {
            "original_number_of_features": len(input_df.columns),
            "filtered_number_of_features": len(df_genomic_locations.columns),
            "genomic_locations": genomic_locations,
        }

        return df_genomic_locations, metadata_df, genomic_locations, species, metrics

    def _update_dataset_metrics(
        self,
        data_path: str,
        df: pd.DataFrame,
        metadata_df: pd.DataFrame | None,
        species: str,
        metrics: dict,
    ) -> None:
        """Update dataset metrics and genomic locations.

        Args:
            data_path (str): Path to the data file
            df (pd.DataFrame): Processed DataFrame
            metadata_df (Optional[pd.DataFrame]): Metadata DataFrame if available
            species (str): Species name
            metrics (Dict): Processing metrics dictionary

        Note:
            Updates both dataset metrics and global genomic locations sets

        """

        def calc_stats(data: np.ndarray, prefix: str) -> dict[str, float]:
            return {
                f"{prefix}_mean": float(np.nanmean(data)),
                f"{prefix}_max": float(np.nanmax(data)),
                f"{prefix}_median": float(np.nanmedian(data)),
                f"{prefix}_min": float(np.nanmin(data)),
                f"{prefix}_std": float(np.nanstd(data)),
            }

        X_stats = calc_stats(df.values, "X")
        metadata_stats = (
            calc_stats(metadata_df.values, "metadata") if metadata_df is not None else {}
        )

        # Update the global set of genomic locations
        if species not in self.all_genomic_locations:
            self.all_genomic_locations[species] = set()
        new_locations = set(metrics["genomic_locations"]) - self.all_genomic_locations[species]
        self.all_genomic_locations[species].update(new_locations)

        # Store metrics for the current dataset
        self.dataset_metrics[data_path] = {
            "species": species,
            "num_samples": len(df),
            **X_stats,
            **metadata_stats,
            "original_number_of_features": metrics["original_number_of_features"],
            "filtered_number_of_features": metrics["filtered_number_of_features"],
        }

        self._save_dataset_metrics()
        self._save_genomic_locations()

    def _parse_genomic_location(
        self,
        location: str,
        species: str,
        embedder: DNALLMEmbedder,
    ) -> list[int]:
        """Parse genomic location string to chromosome index and position.

        Args:
            location (str): Genomic location string (format: "chr:position")
            species (str): Species name
            embedder (DNALLMEmbedder): Embedder for DNA sequences

        Returns:
            List[int]: List containing [chromosome_index, position]

        """
        chrom, pos = location.replace("chr", "").split(":")
        vocab = embedder.ensembl_metadata_dict[species]["vocab"]
        chrom_index = vocab[chrom]
        pos = int(pos)
        return [chrom_index, pos]

    def _save_processed_data(
        self,
        dataset_dir: Path,
        df: pd.DataFrame,
        metadata_df: pd.DataFrame | None,
        chrom_index_pos_list: list[list[int]],
        species: str,
    ) -> None:
        """Save processed data to memory-mapped files.

        Args:
            dataset_dir (Path): Directory to save the data
            df (pd.DataFrame): Processed DataFrame
            metadata_df (Optional[pd.DataFrame]): Metadata DataFrame if available
            chrom_index_pos_list (List[List[int]]): List of [chrom_index, position] pairs
            species (str): Species name

        Note:
            Saves data as memory-mapped arrays for efficient access

        """
        X = np.memmap(
            dataset_dir / "X.mmap",
            dtype="float32",
            mode="w+",
            shape=df.to_numpy().shape,
        )
        var = np.memmap(
            dataset_dir / "var.mmap",
            dtype="int32",
            mode="w+",
            shape=(len(chrom_index_pos_list), 2),
        )
        if metadata_df is not None:
            obsm = np.memmap(
                dataset_dir / "obsm.mmap",
                dtype="float32",
                mode="w+",
                shape=metadata_df.to_numpy().shape,
            )

        X[:] = df.to_numpy()
        var[:] = np.array(chrom_index_pos_list, dtype=np.int32)
        if metadata_df is not None:
            obsm[:] = metadata_df.to_numpy()

        X.flush()
        var.flush()
        if metadata_df is not None:
            obsm.flush()

        np.save(dataset_dir / "var_names.npy", ["chrom", "position"])
        np.save(dataset_dir / "obs_names.npy", df.index.values)
        np.save(dataset_dir / "species.npy", species)
        np.save(dataset_dir / "X_shape.npy", df.to_numpy().shape)

        if metadata_df is not None:
            np.save(dataset_dir / "obsm_names.npy", metadata_df.columns.values)

        del X, var
        if metadata_df is not None:
            del obsm
        gc.collect()

    def _save_dataset_metrics(self) -> None:
        """Save dataset metrics to SQLite database."""
        metrics_file = Path(self.processed_dir) / "dataset_metrics.db"
        with sqlitedict.SqliteDict(metrics_file, autocommit=True) as db:
            db.update(self.dataset_metrics)
        self.logger.debug(f"Saved dataset metrics under {metrics_file}.")

    def _save_genomic_locations(self) -> None:
        """Save genomic locations to SQLite database."""
        genomic_locations_file = Path(self.processed_dir) / "genomic_locations.db"
        with sqlitedict.SqliteDict(genomic_locations_file, autocommit=True) as db:
            for species, locations in self.all_genomic_locations.items():
                db[species] = list(locations)
        self.logger.debug(f"Saved genomic locations under {genomic_locations_file}.")

    def _validate_methylation_data(
        self,
        values: np.ndarray,
        dataset_name: str,
        min_extreme_ratio: float = 0.0005,
        unmethylated_threshold: float = 0.25,
        methylated_threshold: float = 0.7,
        min_density: float = 0.05,
        min_peak_to_middle_ratio: float = 2.0,
        extreme_low_threshold: float = 0.05,
        extreme_high_threshold: float = 0.95,
    ) -> bool:
        """Validate methylation data for typical patterns and ranges.

        Args:
            values (np.ndarray): Array of methylation values
            dataset_name (str): Name of dataset for logging
            min_extreme_ratio (float, optional): Minimum ratio of extreme values.
                Defaults to 0.0005
            unmethylated_threshold (float, optional): Maximum unmethylated value. Defaults to 0.25
            methylated_threshold (float, optional): Minimum methylated value. Defaults to 0.7
            min_density (float, optional): Minimum density in peaks. Defaults to 0.05
            min_peak_to_middle_ratio (float, optional): Minimum peak/valley ratio. Defaults to 2.0
            extreme_low_threshold (float, optional): Low extreme cutoff. Defaults to 0.05
            extreme_high_threshold (float, optional): High extreme cutoff. Defaults to 0.95

        Returns:
            bool: True if data passes validation, False otherwise

        Note:
            Checks for:
            - Values in [0,1] range
            - Sufficient extreme values
            - Bimodal distribution
            - Clear valley between peaks

        """
        valid_values = values[~np.isnan(values)].flatten()

        # Check for values outside [0,1] range
        if (valid_values > 1).any() or (valid_values < 0).any():
            self.logger.warning(
                "Found values greater than 1 or less than 0 in the processed data. "
                f"Skipping {dataset_name}.",
            )
            return False

        # Check for presence of extreme values
        low_values = (valid_values <= extreme_low_threshold).sum()
        high_values = (valid_values >= extreme_high_threshold).sum()
        total_values = len(valid_values)

        if (low_values / total_values < min_extreme_ratio) or (
            high_values / total_values < min_extreme_ratio
        ):
            self.logger.warning(
                "Data lacks sufficient extreme values (near 0 or 1). "
                f"Low values: {low_values/total_values:.4%}, "
                f"High values: {high_values/total_values:.4%}. "
                f"Skipping {dataset_name}.",
            )
            return False

        # Check for methylation-specific bimodality
        hist, bin_edges = np.histogram(valid_values, bins=50, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram to get probability density
        hist = hist / np.sum(hist)

        # Calculate density in unmethylated and methylated regions
        unmethylated_mask = bin_centers <= unmethylated_threshold
        methylated_mask = bin_centers >= methylated_threshold

        unmethylated_density = np.sum(hist[unmethylated_mask])
        methylated_density = np.sum(hist[methylated_mask])

        # Check if both regions have sufficient density
        has_bimodal_density = (
            unmethylated_density >= min_density and methylated_density >= min_density
        )

        # Calculate ratio between peaks and middle region
        middle_mask = (bin_centers > unmethylated_threshold) & (bin_centers < methylated_threshold)
        middle_density = np.sum(hist[middle_mask])
        peak_to_middle_ratio = (unmethylated_density + methylated_density) / (
            middle_density + 1e-10
        )
        has_valley = peak_to_middle_ratio > min_peak_to_middle_ratio

        if not (has_bimodal_density and has_valley):
            self.logger.warning(
                f"Beta value distribution does not show typical methylation pattern "
                f"(peaks near 0 and 1). Skipping {dataset_name}.",
            )
            return False

        return True
