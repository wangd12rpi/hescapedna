import gc
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import sqlitedict
from tqdm.rich import tqdm

from cpgpt.log.utils import get_class_logger

from .dna_llm_embedder import DNALLMEmbedder


class IlluminaMethylationProber:
    """A class for converting Illumina methylation probe IDs to genomic locations.

    This class downloads, parses, and manages Illumina methylation manifest files,
    providing a way to map probe IDs to their corresponding genomic locations for
    various species.

    Attributes:
        dependencies_dir (str): Directory for storing dependencies and embeddings.
        manifests_dir (str): Directory for storing manifest files.
        illumina_metadata_dict (Dict): Dictionary to store parsed metadata.

    """

    def __init__(self, dependencies_dir: str, embedder: DNALLMEmbedder) -> None:
        """Initialize the IlluminaMethylationProber.

        Args:
            dependencies_dir (str): Directory for storing dependencies and embeddings.
            embedder (DNALLMEmbedder): Instance of DNALLMEmbedder for DNA embeddings.

        Raises:
            TypeError: If dependencies_dir is not a string.

        """
        if not isinstance(dependencies_dir, str):
            msg = f"dependencies_dir must be a string, got {type(dependencies_dir)}."
            raise TypeError(msg)

        if not isinstance(embedder, DNALLMEmbedder):
            msg = f"embedder must be a DNALLMEmbedder, got {type(embedder)}."
            raise TypeError(msg)

        self.dependencies_dir = dependencies_dir
        self.manifests_dir = Path(self.dependencies_dir) / "manifests"
        self.embedder = embedder
        self.illumina_metadata_dict = {"processed_files": []}

        self.logger = get_class_logger(self.__class__)

        self._create_directories()
        self._load_illumina_metadata()

    def _create_directories(self) -> None:
        """Create necessary directories for storing data.

        Creates:
            - dependencies_dir: Main directory for dependencies
            - manifests_dir: Directory for manifest files
        """
        Path(self.dependencies_dir).mkdir(parents=True, exist_ok=True)
        Path(self.manifests_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Illumina methylation manifest files will be stored under {self.manifests_dir}.",
        )

    def _load_illumina_metadata(self) -> None:
        """Load the Illumina methylation manifest files.

        Loads existing metadata from SQLite database if available, including
        the list of processed files.
        """
        illumina_metadata_dict_file = Path(self.dependencies_dir) / "illumina_metadata.db"
        if not illumina_metadata_dict_file.exists():
            self.logger.warning("Illumina metadata not found. Downloading now.")
            self.download_illumina_metadata()
            self.logger.info("Illumina metadata downloaded successfully.")
            self.logger.info("Parsing Illumina metadata now.")
            self.parse_illumina_metadata()
            self.logger.info("Illumina metadata parsed successfully.")

        with sqlitedict.SqliteDict(illumina_metadata_dict_file, autocommit=True) as db:
            self.illumina_metadata_dict = dict(db)
        self.logger.info("Illumina metadata dictionary loaded successfully.")

    def download_illumina_metadata(self, force: bool = False) -> None:
        """Download Illumina metadata from GitHub repository.

        Downloads and extracts manifest files by cloning a specific commit of the
        InfiniumAnnotationV1 repository.

        Args:
            force (bool, optional): Whether to force download even if files exist.
                Defaults to False.

        Raises:
            subprocess.CalledProcessError: If git operations fail
            Exception: If there's an error processing the repository

        """
        if self.illumina_metadata_dict["processed_files"] and not force:
            self.logger.info(
                "It seems the Illumina metadata is already downloaded. Use force=True "
                "to redownload.",
            )
            return

        self.logger.info("Downloading Illumina metadata. Might take a while.")
        url = "https://github.com/zhou-lab/InfiniumAnnotationV1.git"
        repo_name = "InfiniumAnnotationV1"
        specific_commit = "7f5f2d43fc3e53cc29df793ea3f3e847b38cfc5d"

        # Full path to where the repository will be cloned
        repo_path = self.manifests_dir / repo_name

        try:
            # Clone the repository directly into self.manifests_dir
            subprocess.run(
                ["/usr/bin/git", "clone", "--quiet", "--depth", "1", url, repo_name],
                cwd=self.manifests_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Checkout the specific commit
            subprocess.run(
                ["/usr/bin/git", "checkout", "--quiet", specific_commit],
                cwd=repo_path,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Move contents one level up to self.manifests_dir
            for item in os.listdir(repo_path):
                s = repo_path / item
                d = self.manifests_dir / item
                if s.is_dir():
                    if d.exists():
                        shutil.rmtree(d)  # Remove existing directory if it exists
                    shutil.move(s, d)
                else:
                    shutil.move(s, d)

            # Remove the now-empty repo directory
            shutil.rmtree(repo_path)

            self.logger.info(
                "Illumina metadata downloaded, extracted, and decompressed successfully.",
            )
        except subprocess.CalledProcessError:
            self.logger.exception("Failed to clone repository or checkout commit")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            raise
        except Exception:
            self.logger.exception("An error occurred while processing the repository")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            raise

    def _save_illumina_metadata(self) -> None:
        """Save the Illumina metadata dictionary to SQLite database.

        Saves the metadata dictionary containing probe mappings and processed files
        to a persistent SQLite database.
        """
        illumina_metadata_dict_file = Path(self.dependencies_dir) / "illumina_metadata.db"
        with sqlitedict.SqliteDict(illumina_metadata_dict_file, autocommit=True) as db:
            db.update(self.illumina_metadata_dict)
        self.logger.debug(
            f"Saved illumina metadata dictionary under {illumina_metadata_dict_file}.",
        )

    def parse_illumina_metadata(self, human: bool = True, mammalian: bool = False) -> None:
        """Parse Illumina metadata files and populate metadata dictionary.

        Processes manifest files for human and/or mammalian arrays, mapping probe IDs
        to genomic locations. Saves progress periodically and can resume if interrupted.

        Args:
            human (bool, optional): Whether to parse human arrays. Defaults to True.
            mammalian (bool, optional): Whether to parse mammalian arrays. Defaults to False.

        Raises:
            TypeError: If neither human nor mammalian is True

        """
        if not human and not mammalian:
            msg = "At least one of human or mammalian must be True"
            raise TypeError(msg)

        self.logger.info("Parsing Illumina methylation manifest files.")

        all_files = self._collect_files_to_process(human=human, mammalian=mammalian)

        # Filter out already processed files
        processed_files = set(self.illumina_metadata_dict.get("processed_files", []))
        files_to_process = [f for f in all_files if f[0] not in processed_files]
        if processed_files:
            self.logger.info(f"{len(processed_files)} files already processed. Skipping those.")

        # Process all files with a single progress bar
        if len(files_to_process) > 0:
            for i, (file_path, species, remove_chr) in enumerate(
                tqdm(files_to_process, desc="Processing array files"),
            ):
                sucessfully_processed = self._process_file(file_path, species, remove_chr)
                if sucessfully_processed:
                    self.illumina_metadata_dict.setdefault("processed_files", []).append(file_path)

                # Clear memory
                gc.collect()

                # Save progress every 10 files or at the end
                if (i + 1) % 10 == 0 or i == len(files_to_process) - 1:
                    self._save_illumina_metadata()

            self._save_illumina_metadata()

        self.logger.info("Illumina metadata parsing completed.")

    def _collect_files_to_process(
        self,
        human: bool = True,
        mammalian: bool = False,
    ) -> list[tuple[str, str, bool]]:
        """Collect all manifest files that need to be processed.

        Args:
            human (bool, optional): Whether to include human arrays. Defaults to True.
            mammalian (bool, optional): Whether to include mammalian arrays. Defaults to False.

        Returns:
            List[Tuple[str, str, bool]]: List of tuples containing:
                - file_path (str): Path to manifest file
                - species (str): Species name
                - remove_chr (bool): Whether to remove 'chr' prefix

        """
        all_files = []

        # Process Homo sapiens arrays
        if human:
            human_files = [
                "Anno/EPICv2/EPICv2.hg38.manifest.tsv.gz",
                "Anno/EPIC+/EPIC+.hg38.manifest.tsv.gz",
                "Anno/EPIC/EPIC.hg38.manifest.tsv.gz",
                "Anno/HM27/HM27.hg38.manifest.tsv.gz",
                "Anno/HM450/HM450.hg38.manifest.tsv.gz",
                "Anno/MSA/MSA.hg38.manifest.tsv.gz",
            ]

            for file_path in human_files:
                full_path = self.manifests_dir / file_path
                if full_path.exists():
                    all_files.append((str(full_path), "homo_sapiens", True))
                else:
                    self.logger.warning(f"Homo sapiens array file not found: {full_path}.")

        # Process mammalian arrays
        if mammalian:
            folder_path = self.manifests_dir / "Mammal40"
            for filename in os.listdir(folder_path):
                if filename.endswith(".tsv.gz"):
                    file_path = folder_path / filename
                    species = filename.replace(".tsv.gz", "")

                    # Skip species that are not in Ensembl
                    if species not in self.embedder.ensembl_metadata_dict:
                        self.logger.debug(
                            f"Skipping {filename} as the species is not in the Ensembl metadata.",
                        )
                        continue

                    all_files.append((str(file_path), species, False))

        return all_files

    def _process_file(self, file_path: str, species: str, remove_chr: bool = False) -> bool:
        """Process a single manifest file and update metadata dictionary.

        Args:
            file_path (str): Path to the manifest file
            species (str): Ensembl species name for the manifest file
            remove_chr (bool, optional): Whether to remove 'chr' prefix. Defaults to False.

        Returns:
            bool: True if processing successful, False otherwise

        Raises:
            Exception: If there's an error processing the file

        """
        try:
            # Read the gzipped TSV file directly without decompressing and handle HM27 edge case
            if "HM27.hg38.manifest.tsv.gz" in file_path:
                manifest_df = pd.read_csv(
                    file_path,
                    sep="\t",
                    compression="gzip",
                    usecols=["probeID", "CpG_chrm", "CpG_beg"],
                    dtype=str,
                )
                manifest_df["Probe_ID"] = manifest_df["probeID"]
            else:
                manifest_df = pd.read_csv(
                    file_path,
                    sep="\t",
                    compression="gzip",
                    usecols=["Probe_ID", "CpG_chrm", "CpG_beg"],
                    dtype=str,
                )

            # Filter out rows with NA in 'CpG_chrm'
            manifest_df = manifest_df.dropna(subset=["CpG_chrm"])

            # Remove 'chr' prefix if needed
            if remove_chr:
                manifest_df["CpG_chrm"] = manifest_df["CpG_chrm"].str.replace("chr", "")

            # Remove probes that only measure SNPs or non-variable sites
            manifest_df = manifest_df[~manifest_df["Probe_ID"].str.startswith(("rs", "nv"))]

            # Collapase probe IDs that map to the same locus in EPICv2
            manifest_df["Probe_ID"] = manifest_df["Probe_ID"].str.split("_").str[0]

            # Replace mitochondrial chromosome 'M' with 'MT'
            manifest_df["CpG_chrm"] = manifest_df["CpG_chrm"].replace("M", "MT")

            # Create genomic location string
            manifest_df["genomic_location"] = (
                manifest_df["CpG_chrm"] + ":" + manifest_df["CpG_beg"].astype(str)
            )

            # Create or update the species subdictionary
            if species not in self.illumina_metadata_dict:
                self.illumina_metadata_dict[species] = {}

            # Populate the species subdictionary
            for _, row in manifest_df.iterrows():
                probe_id = row["Probe_ID"]
                genomic_location = row["genomic_location"]

                if (
                    genomic_location.split(":")[0]
                    not in self.embedder.ensembl_metadata_dict[species]["vocab"]
                ):
                    self.logger.debug(
                        f"Chromosome {genomic_location.split(':')[0]} not found in embedder "
                        f"for species {species}. Skipping.",
                    )
                    continue
                if probe_id not in self.illumina_metadata_dict[species]:
                    self.illumina_metadata_dict[species][probe_id] = genomic_location
        except Exception:
            self.logger.exception(f"Error processing file {file_path}")
            return False
        else:
            return True

    def locate_probes(self, probe_ids: list[str], species: str) -> list[str]:
        """Convert probe IDs to genomic locations.

        Args:
            probe_ids (List[str]): List of probe IDs
            species (str): Species name

        Returns:
            List[str]: List of corresponding genomic locations

        Note:
            Logs error if probe ID not found for given species

        """
        genomic_locations = []
        for probe_id in probe_ids:
            if (
                species in self.illumina_metadata_dict
                and probe_id in self.illumina_metadata_dict[species]
            ):
                genomic_locations.append(self.illumina_metadata_dict[species][probe_id])
            else:
                self.logger.error(
                    f"Genomic location not found for probe ID {probe_id} and species {species}.",
                )
        return genomic_locations

    def probe_location(self, genomic_locations: list[str], species: str) -> list[str]:
        """Convert genomic locations to probe IDs.

        Args:
            genomic_locations (List[str]): List of genomic locations
            species (str): Species name

        Returns:
            List[str]: List of corresponding probe IDs

        Note:
            Logs error if probe not found for given location and species

        """
        probe_ids = []
        if species not in self.illumina_metadata_dict:
            self.logger.error(f"Species {species} not found in illumina_metadata_dict.")
            return probe_ids

        # Create a reverse lookup dictionary
        reverse_lookup = {v: k for k, v in self.illumina_metadata_dict[species].items()}

        for location in genomic_locations:
            if location in reverse_lookup:
                probe_ids.append(reverse_lookup[location])
            else:
                self.logger.error(
                    f"Probe ID not found for genomic location {location} and species {species}.",
                )

        return probe_ids
