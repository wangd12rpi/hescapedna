# Random seed for reproducibility
RANDOM_SEED = 42

# Directory paths
DEPENDENCIES_DIR = "../cpgpt_files/dependencies"
DATA_DIR = "../wsi_dnameth"
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer


MODEL_NAME = "large"

if __name__ == '__main__':
    inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=DATA_DIR)
    inferencer.download_dependencies(species="human")
    inferencer.download_model(MODEL_NAME)


