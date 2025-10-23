import gc
import gzip
import json
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import requests
import sqlitedict
import torch
from loguru import logger
from pyfaidx import Fasta
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.models.bert.configuration_bert import BertConfig

from cpgpt.log.utils import DownloadProgressBar, get_class_logger


class DNALLMEmbedder:
    """A class for generating and managing DNA embeddings using language models.

    This class provides functionality to load DNA language models, generate embeddings
    for genomic locations, and manage the storage and retrieval of these embeddings.
    It supports multiple species and different DNA language models.

    Attributes:
        dependencies_dir (str): Directory for storing dependencies and embeddings.
        genome_dir (str): Directory for storing genome files.
        dna_embeddings_dir (str): Directory for storing DNA embeddings.
        ensembl_metadata_dict (Dict): Dictionary containing Ensembl metadata for various species.
        llm_embedding_size_dict (Dict): Dict of valid DNA language models and their
            embedding sizes.

    """

    def __init__(self, dependencies_dir: str) -> None:
        """Initialize the DNALLMEmbedder.

        Args:
            dependencies_dir (str): Directory for storing dependencies and embeddings.

        Raises:
            ValueError: If dependencies_dir is not a string.

        """
        if not isinstance(dependencies_dir, str):
            msg = f"dependencies_dir must be a string, got {type(dependencies_dir)}"
            raise TypeError(msg)

        self.dependencies_dir = dependencies_dir
        self.genome_dir = Path(self.dependencies_dir) / "genomes"
        self.dna_embeddings_dir = Path(self.dependencies_dir) / "dna_embeddings"
        self.ensembl_metadata_dict: dict = {}
        self.llm_embedding_size_dict = {
            "nucleotide-transformer-v2-500m-multi-species": 1024,
            "DNABERT-2-117M": 768,
            "hyenadna-large-1m-seqlen-hf": 256,
        }

        self.logger = get_class_logger(self.__class__)

        self._create_directories()
        self._load_ensembl_metadata()

    def _create_directories(self) -> None:
        """Create necessary directories for storing data.

        Creates:
            - dependencies_dir: Main directory for dependencies
            - genome_dir: Directory for genome files
            - dna_embeddings_dir: Directory for DNA embeddings
        """
        Path(self.dependencies_dir).mkdir(parents=True, exist_ok=True)
        Path(self.genome_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dna_embeddings_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Genome files will be stored under {self.genome_dir}.")
        self.logger.info(
            f"DNA embeddings will be stored under {self.dna_embeddings_dir} and subdirectories.",
        )

    def _load_ensembl_metadata(self) -> None:
        """Load or download the Ensembl metadata file and parse it into a dictionary.

        Downloads metadata from Ensembl if not already present locally.
        Parses the metadata into a dictionary containing genome assembly and sequence information.

        Raises:
            Exception: If metadata cannot be downloaded or parsed

        """
        ensembl_metadata_dict_file = Path(self.dependencies_dir) / "ensembl_metadata.db"
        if ensembl_metadata_dict_file.exists():
            with sqlitedict.SqliteDict(ensembl_metadata_dict_file, autocommit=True) as db:
                self.ensembl_metadata_dict = dict(db)
            self.logger.info("Ensembl metadata dictionary loaded successfully")
            return

        self.logger.info("Loading Ensembl metadata. Might take a while")
        ensembl_metadata_file = self.genome_dir / "species_metadata_EnsemblVertebrates.json"

        if not ensembl_metadata_file.exists():
            self._download_ensembl_metadata(ensembl_metadata_file)

        self._parse_ensembl_metadata(ensembl_metadata_file)

    def _download_ensembl_metadata(self, ensembl_metadata_file: Path) -> None:
        """Download the Ensembl metadata file.

        Args:
            ensembl_metadata_file (Path): Path to save the downloaded metadata file

        Raises:
            Exception: If download fails or response status is not 200

        """
        url = "https://ftp.ensembl.org/pub/release-113/species_metadata_EnsemblVertebrates.json"
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192
            progress_bar = DownloadProgressBar(
                self.logger,
                self.__class__.__name__,
                total=total_size,
            )
            ensembl_metadata_file_tmp = ensembl_metadata_file.with_suffix(".tmp")
            with open(ensembl_metadata_file_tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    size = f.write(chunk)
                    progress_bar.update(size)
                progress_bar.close()
            shutil.move(ensembl_metadata_file_tmp, ensembl_metadata_file)
            self.logger.info("Ensembl metadata downloaded successfully.")
        else:
            self.logger.error(
                f"Failed to download Ensembl metadata. Status code: {response.status_code}",
            )

    def _parse_ensembl_metadata(self, ensembl_metadata_file: str) -> None:
        """Parse the Ensembl metadata file and create a dictionary.

        Args:
            ensembl_metadata_file (str): Path to the Ensembl metadata file

        Creates dictionary entries containing:
            - scientific_name: Scientific name of organism
            - assembly: Default genome assembly
            - vocab: Dictionary mapping sequence names to indices
            - reverse_vocab: Dictionary mapping indices to sequence names
            - embeddings: Empty dictionary for each supported LLM

        """
        with open(ensembl_metadata_file) as f:
            try:
                ensembl_metadata = json.load(f)
            except json.JSONDecodeError:
                self.logger.exception(f"Failed to parse Ensembl metadata: {f}")
                raise

        self.logger.info("Parsing genome assemblies and sequence names.")
        for organism in ensembl_metadata:
            name = organism["organism"]["name"]
            self.ensembl_metadata_dict[name] = {
                "scientific_name": organism["organism"]["scientific_name"],
                "assembly": organism["assembly"]["assembly_default"],
                "vocab": {
                    seq["name"]: i for i, seq in enumerate(organism["assembly"]["sequences"])
                },
                "reverse_vocab": {
                    i: seq["name"] for i, seq in enumerate(organism["assembly"]["sequences"])
                },
                **{llm: {} for llm in self.llm_embedding_size_dict},
            }

        self._save_ensembl_metadata()

        self.logger.info("Ensembl metadata dictionary created successfully.")

    def _load_dna_model(
        self,
        dna_llm: str = "nucleotide-transformer-v2-500m-multi-species",
    ) -> tuple[nn.Module, PreTrainedTokenizer]:
        """Load the specified DNA language model and tokenizer.

        Args:
            dna_llm (str): Name of the DNA language model to load. Defaults to
                "nucleotide-transformer-v2-500m-multi-species".

        Returns:
            Tuple[nn.Module, PreTrainedTokenizer]: Tuple containing:
                - model: The loaded DNA language model
                - tokenizer: The associated tokenizer

        Raises:
            ValueError: If an unsupported DNA language model is specified

        """
        self.logger.info(f"Loading DNA LLM: {dna_llm}.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Using CPU for inference.")

        if dna_llm == "nucleotide-transformer-v2-500m-multi-species":
            tokenizer = AutoTokenizer.from_pretrained(
                "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                trust_remote_code=True,
            )
            model = AutoModelForMaskedLM.from_pretrained(
                "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                trust_remote_code=True,
            )
        elif dna_llm == "DNABERT-2-117M":
            tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
            )
            config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
            model = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True,
                config=config,
            )
        elif dna_llm == "hyenadna-large-1m-seqlen-hf":
            tokenizer = AutoTokenizer.from_pretrained(
                "LongSafari/hyenadna-large-1m-seqlen-hf",
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                "LongSafari/hyenadna-large-1m-seqlen-hf",
                trust_remote_code=True,
            )
        else:
            msg = f"Unsupported DNA language model: {dna_llm}"
            raise ValueError(msg)

        model = model.to(device)
        model.eval()
        self.logger.info("DNA LLM loaded successfully.")
        return model, tokenizer

    def parse_dna_embeddings(
        self,
        genomic_locations: list[str],
        species: str,
        dna_context_len: int = 2001,
        dna_llm: str = "nucleotide-transformer-v2-500m-multi-species",
        batch_size: int = 1,
        num_workers: int = 8,
        genome_file: str | None = None,
    ) -> None:
        """Generate or retrieve DNA embeddings for given genomic locations.

        Args:
            genomic_locations (List[str]): List of genomic locations
                (e.g., ["chr1:1000", "chr2:2000"])
            species (str): Ensembl species name (e.g., "homo_sapiens")
            dna_context_len (int, optional): Context length for DNA sequences. Defaults to 2001
            dna_llm (str, optional): Name of the DNA language model. Defaults to
                "nucleotide-transformer-v2-500m-multi-species"
            batch_size (int, optional): Batch size for DNA LLM processing. Defaults to 1
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 8
            genome_file (Optional[str], optional): Path to the genome fasta file. Defaults to None

        Raises:
            ValueError: If input parameters are invalid
            Exception: If there's an error during embedding generation or saving

        """
        # Validate input parameters
        self._validate_parse_dna_embeddings_input(dna_context_len, dna_llm)

        self.logger.info(f"Processing DNA embeddings for species: {species}.")

        # Load genome
        if genome_file is None:
            genome_file = self._load_genome(species)
        genome = Fasta(genome_file)

        # Set up embedding files
        embeddings_dir = Path(self.dna_embeddings_dir) / species / dna_llm
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = embeddings_dir / f"{dna_context_len}bp_dna_embeddings.mmap.tmp"

        # Initialize or load embeddings
        embeddings, initial_size = self._initialize_embeddings(
            species,
            dna_llm,
            dna_context_len,
            embeddings_file,
        )

        # Load DNA model
        model, tokenizer = self._load_dna_model(dna_llm)

        # Process genomic locations
        self._process_genomic_locations(
            genomic_locations,
            species,
            dna_llm,
            dna_context_len,
            genome,
            embeddings,
            model,
            tokenizer,
            batch_size,
            num_workers,
        )

        # Final save and cleanup
        self._finalize_embeddings(species, dna_llm, dna_context_len, embeddings, embeddings_file)

        self.logger.info("DNA embeddings processed successfully.")

    def _validate_parse_dna_embeddings_input(self, dna_context_len: int, dna_llm: str) -> None:
        """Validate input parameters for parse_dna_embeddings method.

        Args:
            dna_context_len (int): Context length for DNA sequences
            dna_llm (str): Name of the DNA language model

        Raises:
            ValueError: If dna_context_len is invalid or dna_llm is not supported

        """
        if not isinstance(dna_context_len, int) or dna_context_len < 100:
            msg = f"dna_context_len must be an integer >= 100, got {dna_context_len}"
            raise ValueError(msg)

        if dna_llm not in self.llm_embedding_size_dict:
            msg = f"dna_llm must be one of {self.llm_embedding_size_dict.keys()}, got {dna_llm}"
            raise ValueError(
                msg,
            )

    def _initialize_embeddings(
        self,
        species: str,
        dna_llm: str,
        dna_context_len: int,
        embeddings_file: str,
    ) -> tuple[np.memmap, int]:
        """Initialize or load existing embeddings.

        Args:
            species (str): Species name
            dna_llm (str): Name of DNA language model
            dna_context_len (int): Context length for DNA sequences
            embeddings_file (str): Path to embeddings file

        Returns:
            Tuple[np.memmap, int]: Tuple containing:
                - embeddings: Memory-mapped array of embeddings
                - initial_size: Initial size of embeddings array

        Raises:
            Exception: If embeddings file cannot be created or accessed

        """
        if dna_context_len not in self.ensembl_metadata_dict[species][dna_llm]:
            self.ensembl_metadata_dict[species][dna_llm][dna_context_len] = {}

        initial_size = max(
            10000,
            len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len]),
        )
        embedding_size = self.llm_embedding_size_dict[dna_llm]

        # Check if the permanent embeddings file (without .tmp) exists
        permanent_file = Path(embeddings_file).with_suffix("").with_suffix(".mmap")
        if permanent_file.exists():
            self.logger.info(f"Found existing embeddings file: {permanent_file}")
            # Load existing embeddings
            existing_embeddings = np.memmap(
                permanent_file,
                dtype="float32",
                mode="r",
                shape=(
                    len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len]),
                    embedding_size,
                ),
            )

            # Create new temporary file with sufficient size
            embeddings = np.memmap(
                embeddings_file,
                dtype="float32",
                mode="w+",
                shape=(max(initial_size, existing_embeddings.shape[0]), embedding_size),
            )

            # Copy existing embeddings to the temporary file
            embeddings[: existing_embeddings.shape[0]] = existing_embeddings[:]
            embeddings.flush()

            return embeddings, initial_size

        # If the temporary file exists, open in r+ mode
        if Path(embeddings_file).exists():
            embeddings = np.memmap(
                embeddings_file,
                dtype="float32",
                mode="r+",
                shape=(
                    max(
                        initial_size,
                        len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len]),
                    ),
                    embedding_size,
                ),
            )
        else:
            # Create new temporary file if neither exists
            embeddings = np.memmap(
                embeddings_file,
                dtype="float32",
                mode="w+",
                shape=(initial_size, embedding_size),
            )

        return embeddings, initial_size

    def _process_genomic_locations(
        self,
        genomic_locations: list[str],
        species: str,
        dna_llm: str,
        dna_context_len: int,
        genome: Fasta,
        embeddings: np.memmap,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Process genomic locations and generate embeddings.

        Args:
            genomic_locations (List[str]): List of genomic locations
            species (str): Species name
            dna_llm (str): Name of DNA language model
            dna_context_len (int): Context length for DNA sequences
            genome (Fasta): Genome reference object
            embeddings (np.memmap): Memory-mapped array of embeddings
            model (nn.Module): DNA language model
            tokenizer (PreTrainedTokenizer): Associated tokenizer
            batch_size (int): Batch size for processing
            num_workers (int): Number of dataloader workers

        Raises:
            ValueError: If input validation fails
            RuntimeError: If embeddings file is corrupted
            Exception: If embedding generation fails

        """
        last_save_time = time.time()
        save_time_interval = 300  # Save progress every 5 minutes

        # Verify embeddings file integrity before starting
        try:
            embeddings[0]  # Verify we can read from mmap
        except Exception as e:
            self.logger.exception("Failed to access embeddings file")
            msg = "Embeddings file is corrupted or inaccessible"
            raise RuntimeError(msg) from e

        # Track processed locations for verification
        processed_locations = {}

        try:
            # Filter out already processed embeddings
            processed_genomic_locations = set(
                self.ensembl_metadata_dict[species][dna_llm][dna_context_len].keys(),
            )
            genomic_locations_to_process = [
                f for f in genomic_locations if f not in processed_genomic_locations
            ]
            if processed_genomic_locations:
                self.logger.info(
                    f"{len(genomic_locations) - len(genomic_locations_to_process)} "
                    "genomic locations already processed. Skipping those.",
                )

            # Return nothing if everything has already been processed
            if len(genomic_locations_to_process) == 0:
                return

            # Create dataset
            dataset = GenomicSequenceDataset(genomic_locations_to_process, genome, dna_context_len)

            # Create dataloader with partial collate function
            collate_fn = partial(collate_sequences, tokenizer=tokenizer)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=num_workers > 0,
            )

            with tqdm(
                desc="Generating embeddings",
                total=len(genomic_locations_to_process),
            ) as pbar:
                for i, batch in enumerate(dataloader):
                    # Verify input data
                    if not isinstance(batch["input_ids"], torch.Tensor):
                        msg = f"Expected tensor input, got {type(batch['input_ids'])}"
                        self._raise_type_error(msg)
                    if len(batch["locations"]) != batch["input_ids"].shape[0]:
                        msg = "Batch size mismatch between locations and inputs"
                        self._raise_validation_error(msg)

                    # Generate embeddings with error checking
                    try:
                        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
                        attention_mask = batch["attention_mask"].float().to(model.device)

                        with torch.inference_mode():
                            if dna_llm == "DNABERT-2-117M":
                                outputs = model(input_ids)
                                # Apply mask before mean
                                masked_outputs = outputs[0] * attention_mask.unsqueeze(-1)
                                batch_embeddings = (
                                    (
                                        masked_outputs.sum(dim=1)
                                        / attention_mask.sum(dim=1, keepdim=True)
                                    )
                                    .cpu()
                                    .numpy()
                                )
                            else:
                                outputs = model(input_ids, output_hidden_states=True)
                                # Apply mask before mean
                                masked_hidden = outputs.hidden_states[
                                    -1
                                ] * attention_mask.unsqueeze(-1)
                                batch_embeddings = (
                                    (
                                        masked_hidden.sum(dim=1)
                                        / attention_mask.sum(dim=1, keepdim=True)
                                    )
                                    .cpu()
                                    .numpy()
                                )

                    except Exception:
                        self.logger.exception(
                            f"Failed to generate embeddings for batch {i}",
                        )
                        raise

                    # Verify embedding dimensions
                    expected_dim = self.llm_embedding_size_dict[dna_llm]
                    if batch_embeddings.shape[1] != expected_dim:
                        msg = (
                            f"Generated embeddings have wrong dimension: "
                            f"{batch_embeddings.shape[1]} != {expected_dim}"
                        )
                        self._raise_validation_error(msg)

                    # Save embeddings with verification
                    try:
                        current_size = len(
                            self.ensembl_metadata_dict[species][dna_llm][dna_context_len],
                        )
                        batch_size = len(batch["locations"])
                        new_indices = np.arange(current_size, current_size + batch_size)

                        # Resize embeddings if needed
                        if new_indices[-1] >= embeddings.shape[0]:
                            new_size = max(embeddings.shape[0] * 2, new_indices[-1] + 1)
                            self.logger.debug(f"Resizing embeddings array to {new_size}")
                            embeddings.flush()

                            # Verify old data before resize
                            old_data = embeddings[:current_size].copy()

                            embeddings = np.memmap(
                                embeddings.filename,
                                dtype="float32",
                                mode="r+",
                                shape=(new_size, batch_embeddings.shape[1]),
                            )

                            # Verify data preserved after resize
                            np.testing.assert_array_equal(
                                embeddings[:current_size],
                                old_data,
                                err_msg="Data corruption detected during resize",
                            )

                        # Save embeddings
                        embeddings[new_indices] = batch_embeddings
                        embeddings.flush()  # Force write to disk

                        # Verify saved embeddings
                        np.testing.assert_array_equal(
                            embeddings[new_indices],
                            batch_embeddings,
                            err_msg="Saved embeddings don't match generated embeddings",
                        )

                        # Update metadata with verification
                        for loc, idx in zip(batch["locations"], new_indices, strict=False):
                            self.ensembl_metadata_dict[species][dna_llm][dna_context_len][loc] = (
                                idx
                            )
                            processed_locations[loc] = idx

                    except Exception:
                        self.logger.exception(f"Failed to save embeddings for batch {i}")
                        raise

                    # Periodic verification and saving
                    if time.time() - last_save_time > save_time_interval:
                        self._verify_saved_data(
                            processed_locations,
                            embeddings,
                            species,
                            dna_llm,
                            dna_context_len,
                        )
                        embeddings.flush()
                        self._save_ensembl_metadata()
                        last_save_time = time.time()

                    # Update progress bar with batch size
                    pbar.update(len(batch["locations"]))

            # Final verification
            self._verify_saved_data(
                processed_locations,
                embeddings,
                species,
                dna_llm,
                dna_context_len,
            )

        except Exception:
            self.logger.exception("Fatal error during embedding generation")
            embeddings.flush()
            self._save_ensembl_metadata()  # Save what we have
            raise

    def _verify_saved_data(
        self,
        processed_locations: dict[str, int],
        embeddings: np.memmap,
        species: str,
        dna_llm: str,
        dna_context_len: int,
    ) -> None:
        """Verify integrity of saved embeddings and metadata.

        Args:
            processed_locations (Dict[str, int]): Dictionary mapping locations to indices
            embeddings (np.memmap): Memory-mapped array of embeddings
            species (str): Species name
            dna_llm (str): Name of DNA language model
            dna_context_len (int): Context length for DNA sequences

        Raises:
            ValueError: If data verification fails or corruption is detected

        """
        self.logger.debug("Verifying data integrity...")

        try:
            # Verify each processed location
            for loc, idx in processed_locations.items():
                # Check metadata consistency
                stored_idx = self.ensembl_metadata_dict[species][dna_llm][dna_context_len].get(loc)
                if stored_idx != idx:
                    msg = f"Index mismatch for location {loc}: {stored_idx} != {idx}"
                    self._raise_validation_error(msg)

                # Verify embedding is accessible and valid
                try:
                    embedding = embeddings[idx]
                    if np.any(np.isnan(embedding)):
                        msg = f"NaN values detected in embedding for location {loc}"
                        self._raise_validation_error(msg)
                    if np.any(np.isinf(embedding)):
                        msg = f"Infinite values detected in embedding for location {loc}"
                        self._raise_validation_error(msg)
                except Exception as e:
                    msg = (
                        f"Cannot access or validate embedding for location {loc} "
                        f"at index {idx}: {e!s}"
                    )
                    raise TypeError(msg) from e

            self.logger.debug("Data integrity verification completed successfully")

        except Exception:
            self.logger.exception("Data integrity verification failed")
            raise

    def _finalize_embeddings(
        self,
        species: str,
        dna_llm: str,
        dna_context_len: int,
        embeddings: np.memmap,
        embeddings_file: str,
    ) -> None:
        """Finalize the embeddings processing.

        Resizes the memory-mapped array if necessary and saves metadata.

        Args:
            species (str): Species name
            dna_llm (str): Name of DNA language model
            dna_context_len (int): Context length for DNA sequences
            embeddings (np.memmap): Memory-mapped array of embeddings
            embeddings_file (str): Path to embeddings file

        """
        embeddings.flush()
        if (
            len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len])
            < embeddings.shape[0]
        ):
            embeddings_file = Path(embeddings_file)
            new_embeddings = np.memmap(
                embeddings_file,
                dtype="float32",
                mode="r+",
                shape=(
                    len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len]),
                    embeddings.shape[1],
                ),
            )
            new_embeddings[:] = embeddings[
                : len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len])
            ]
            new_embeddings.flush()
            embeddings_file.rename(embeddings_file.with_suffix("").with_suffix(".mmap"))

        self._save_ensembl_metadata()

    def _save_ensembl_metadata(self) -> None:
        """Save the Ensembl metadata dictionary to a SQLite database.

        Saves the metadata dictionary containing species information, genome assemblies,
        and embedding indices to a persistent SQLite database.
        """
        ensembl_metadata_dict_file = Path(self.dependencies_dir) / "ensembl_metadata.db"
        ensembl_metadata_dict_file_tmp = ensembl_metadata_dict_file.with_suffix(".db.tmp")

        with sqlitedict.SqliteDict(ensembl_metadata_dict_file_tmp, autocommit=True) as db:
            db.update(self.ensembl_metadata_dict)

        ensembl_metadata_dict_file_tmp.rename(ensembl_metadata_dict_file)
        self.logger.debug(f"Saved Ensembl metadata dictionary under {ensembl_metadata_dict_file}.")

    def _load_genome(self, species: str) -> str:
        """Load or download the genome file for the specified species.

        Args:
            species (str): Scientific name of the species

        Returns:
            str: Path to the loaded genome file

        Raises:
            ValueError: If the species is not found in the Ensembl metadata

        """
        if species not in self.ensembl_metadata_dict:
            msg = f"Ensembl species {species} not found in Ensembl metadata."
            raise ValueError(msg)

        self.logger.info(
            f"Loading assembly {self.ensembl_metadata_dict[species]['assembly']} "
            f"for {species}. Might take a while.",
        )

        genome_file = (
            self.genome_dir
            / f"{species}.{self.ensembl_metadata_dict[species]['assembly']}.dna.toplevel.fa"
        )
        genome_file_gz = genome_file.with_suffix(".fa.gz")

        if genome_file.exists():
            self.logger.info("Uncompressed genome file already exists.")
            return str(genome_file)

        if genome_file_gz.exists():
            self.logger.info("Compressed genome file exists. Decompressing.")
            self._decompress_file(str(genome_file_gz), str(genome_file))
            return str(genome_file)

        url = f"https://ftp.ensembl.org/pub/release-113/fasta/{species}/dna/{species.capitalize()}.{self.ensembl_metadata_dict[species]['assembly']}.dna.toplevel.fa.gz"
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = DownloadProgressBar(
                self.logger,
                self.__class__.__name__,
                total=total_size,
            )
            with open(genome_file_gz, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    progress_bar.update(size)
                progress_bar.close()

            self.logger.info("Genome downloaded successfully. Decompressing.")
            self._decompress_file(str(genome_file_gz), str(genome_file))
            Path.unlink(genome_file_gz)  # Remove the compressed file after decompression
            self.logger.info("Genome file decompressed and ready for use.")
            return str(genome_file)
        self.logger.error("Failed to download genome file.")
        return None

    @staticmethod
    def _decompress_file(input_file: str, output_file: str) -> None:
        """Decompress a gzipped file.

        Args:
            input_file (str): Path to the input gzipped file.
            output_file (str): Path to save the decompressed file.

        """
        output_file_tmp = output_file + ".tmp"
        with gzip.open(input_file, "rb") as f_in, open(output_file_tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        shutil.move(output_file_tmp, output_file)

    def get_embedding(
        self,
        location: str,
        species: str,
        dna_llm: str,
        dna_context_len: int,
    ) -> np.ndarray | None:
        """Retrieve the embedding for a given genomic location.

        Args:
            location (str): Genomic location (e.g., "chr1:1000")
            species (str): Ensembl species name
            dna_llm (str): Name of the DNA language model
            dna_context_len (int): Context length for DNA sequences

        Returns:
            Optional[np.ndarray]: The embedding if found, None otherwise

        """
        if (
            species not in self.ensembl_metadata_dict
            or dna_llm not in self.ensembl_metadata_dict[species]
            or dna_context_len not in self.ensembl_metadata_dict[species][dna_llm]
            or location not in self.ensembl_metadata_dict[species][dna_llm][dna_context_len]
        ):
            return None

        embeddings_file = (
            Path(self.dna_embeddings_dir)
            / species
            / dna_llm
            / f"{dna_context_len}bp_dna_embeddings.mmap"
        )
        embedding_size = self.llm_embedding_size_dict[dna_llm]
        embeddings = np.memmap(
            embeddings_file,
            dtype="float32",
            mode="r",
            shape=(
                len(self.ensembl_metadata_dict[species][dna_llm][dna_context_len]),
                embedding_size,
            ),
        )

        index = self.ensembl_metadata_dict[species][dna_llm][dna_context_len][location]
        return embeddings[index]

    def cleanup(self) -> None:
        """Release memory and clean up resources.

        Performs garbage collection and clears CUDA cache if available.
        """
        self.logger.info("Cleaning up resources.")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Cleanup completed.")

    def _raise_validation_error(self, message: str) -> None:
        """Raise a ValueError with the given message.

        Args:
            message (str): Error message to include with the ValueError

        Raises:
            ValueError: Always raises with the provided message

        """
        raise ValueError(message)

    def _raise_type_error(self, message: str) -> None:
        """Raise a TypeError with the given message.

        Args:
            message (str): Error message to include with the TypeError

        Raises:
            TypeError: Always raises with the provided message

        """
        raise TypeError(message)


class GenomicSequenceDataset(Dataset):
    """Dataset for retrieving genomic sequences.

    Args:
        genomic_locations (List[str]): List of genomic locations in format "chr:position"
        genome (Fasta): Initialized Fasta object representing the genome
        context_len (int, optional): Context length for sequences. Defaults to 2001.

    Attributes:
        locations (List[str]): List of genomic locations
        genome (Fasta): Genome reference object
        context_len (int): Length of sequence context window

    """

    def __init__(
        self,
        genomic_locations: list[str],
        genome: Fasta,
        context_len: int = 2001,
    ) -> None:
        """Initialize the dataset.

        Args:
            genomic_locations (List[str]): List of genomic locations
            genome (Fasta): Initialized Fasta object representing the genome
            context_len (int): Context length for sequences

        """
        self.locations = genomic_locations
        self.genome = genome
        self.context_len = context_len

    def __len__(self) -> int:
        """Return the number of genomic locations in the dataset.

        Returns:
            int: Number of locations in the dataset

        """
        return len(self.locations)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        """Get a sequence for a given index.

        Args:
            idx (int): Index into the dataset

        Returns:
            Dict[str, Union[str, torch.Tensor]]: Dictionary containing:
                - location (str): Genomic location
                - sequence (str): DNA sequence at that location

        Raises:
            Exception: If there is an error processing the sequence

        """
        try:
            location = self.locations[idx]
            sequence = self._get_sequence(location)
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Error processing sequence at index {idx}: {e!s}")
            return {
                "location": location,
                "sequence": "N" * self.context_len,
            }
        else:
            return {"location": location, "sequence": sequence}

    def _get_sequence(self, location: str) -> str:
        """Retrieve genomic sequence for a location.

        Args:
            location (str): Genomic location in format "chr:position"

        Returns:
            str: DNA sequence at the specified location

        Raises:
            KeyError: If chromosome not found in genome file

        """
        chrom, pos = location.split(":")
        pos = int(pos)

        context_start = max(0, pos - self.context_len // 2)
        context_end = min(len(self.genome[f"{chrom}"]) - 1, pos + self.context_len // 2)

        try:
            return str(self.genome[f"{chrom}"][context_start:context_end])
        except KeyError as e:
            msg = f"Chromosome {chrom} not found in genome file"
            raise KeyError(msg) from e


def collate_sequences(
    batch: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, list[str] | torch.Tensor]:
    """Collate function for sequence batches.

    Args:
        batch (List[Dict[str, str]]): List of dictionaries containing:
            - location (str): Genomic location
            - sequence (str): DNA sequence
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer

    Returns:
        Dict[str, Union[List[str], torch.Tensor]]: Dictionary containing:
            - locations (List[str]): List of genomic locations
            - input_ids (torch.Tensor): Tokenized sequences
            - attention_mask (torch.Tensor): Attention mask for sequences

    """
    # Extract locations and sequences
    locations = [item["location"] for item in batch]
    sequences = [item["sequence"] for item in batch]

    # Standard tokenization for other models
    tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]  # Already provided by the tokenizer

    return {
        "locations": locations,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
