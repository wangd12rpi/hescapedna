import torch
from omegaconf import DictConfig, OmegaConf
from hescape.data_modules.image_gexp_dataset import ImageGexpDataModule
from hescape.constants import DatasetEnum
from hescape.models.clip import CLIPModel
from torch.distributed import get_rank  # For rank fallback

# Full config_dict (from your original YAML, complete datamodule + model + paths)
config_dict = {
    "devices_per_job": 4,
    "model": {
        "litmodule": {
            "_target_": "hescape.modules.pretrain_module.PretrainModule",
            "_partial_": True,
            "input_genes": 343,
            "embed_dim": 128,
            "img_enc_name": "uni",
            "gene_enc_name": "scFoundation",
            "loss": "CLIP",
            "img_finetune": True,
            "gene_finetune": False,
            "img_proj": "moe",  # Test moe
            "gene_proj": "linear",
            "n_tissue": None,
            "n_region": None,
            "image_size": 224,
            "temperature": 0.07,
            "lr": 1.0e-05,
            "weight_decay": 0.01
        },
        "optimizer": {
            "lr": 1.0e-05,
            "weight_decay": 0.01
        },
    },
    "paths": {
        "anatomy": {
            "dataset_name": "human-lung-healthy-panel",
            "train_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/train.csv",
            "val_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/val.csv",
            "test_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/test.csv",
            "output": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/../results/human_lung_healthy_panel/local",
            "pretrain_weights": {
                "drvi_model_dir": "drvi_human_lung_healthy_panel",
                "data_gene_reference_path": "../human_lung_healthy_panel/nicheformer_reference.h5ad"
            }
        },
        "dataset_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/human_lung_panel",
        "data_files_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data",
        "base_output_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/../results",
        "pretrain_weights": {
            "img_enc_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/pretrain_weights/image",
            "gene_enc_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/pretrain_weights/gene",
            "base_data_gene_reference_path": "/p/project1/hai_spatial_clip/hescape/data"
        }
    },
    "datamodule": {
        "_target_": "hescape.data_modules.image_gexp_dataset.ImageGexpDataModule",
        "dataset_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/human_lung_panel",
        "dataset_name": "human-lung-healthy-panel",
        "data_files_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data",
        "img_model_name": "uni",
        "gene_model_name": "scFoundation",
        "num_workers": 0,  # Single GPU, no workers
        "pin_memory": True,
        "persistent_workers": False,  # Single, no persistent
        "batch_size": 4,  # Small for test
        "source_key": "tissue",
        "source_value": ["lung"],
        "split_key": "name",
        "split_train_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/train.csv",
        "split_val_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/val.csv",
        "split_test_csv": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/test.csv",
        "input_genes": 343,
        "seed": 24442,
        "data_gene_reference_path": "/lus/lfs1aip1/home/u5t/chuhansong.u5t/UCFhescape/data/human-lung-healthy-panel/nicheformer_reference.h5ad"
    },
    "name": "hescape_default_training"
}

cfg = OmegaConf.create(config_dict)

# Extract datamodule config
dm_config = cfg.datamodule

# Instantiate and setup datamodule (single GPU, small batch)
dm = ImageGexpDataModule(
    dataset_path=dm_config.dataset_path,
    dataset_name=dm_config.dataset_name,
    data_files_path=dm_config.data_files_path,
    img_model_name=dm_config.img_model_name,
    gene_model_name=dm_config.gene_model_name,
    num_workers=dm_config.num_workers,  # 0 for single
    pin_memory=dm_config.pin_memory,
    persistent_workers=dm_config.persistent_workers,  # False for single
    batch_size=dm_config.batch_size,  # 4 small
    source_key=dm_config.source_key,
    source_value=dm_config.source_value,
    split_key=dm_config.split_key,
    split_train_csv=dm_config.split_train_csv,
    split_val_csv=dm_config.split_val_csv,
    split_test_csv=dm_config.split_test_csv,
    input_genes=dm_config.input_genes,
    seed=dm_config.seed,
    data_gene_reference_path=dm_config.data_gene_reference_path  # This is the key arg
)

dm.prepare_data()
dm.setup('fit')  # Setup for training

# Get one batch from train loader
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
print(f"Batch keys: {batch.keys()}")
print(f"Image input shape: {batch[DatasetEnum.IMG].shape}")
print(f"GEXP input shape: {batch[DatasetEnum.GEXP].shape}")

# Extract model params from config (litmodule section)
model_params = cfg.model.litmodule
model_params.world_size = 1  # Single process test
model_params.rank = 0  # For single process test
model_params.img_enc_path = cfg.paths.pretrain_weights.img_enc_path
model_params.gene_enc_path = cfg.paths.pretrain_weights.gene_enc_path
model_params.drvi_model_dir = cfg.paths.anatomy.pretrain_weights.drvi_model_dir
model_params.cfg = cfg  # Pass full config if needed

# Instantiate CLIPModel with modified img_proj="moe"
model = CLIPModel(
    input_genes=model_params.input_genes,
    embed_dim=model_params.embed_dim,
    img_enc_name=model_params.img_enc_name,
    gene_enc_name=model_params.gene_enc_name,
    loss=model_params.loss,
    img_finetune=model_params.img_finetune,
    gene_finetune=model_params.gene_finetune,
    img_proj=model_params.img_proj,  # Now "moe"
    gene_proj=model_params.gene_proj,
    n_tissue=model_params.n_tissue,
    n_region=model_params.n_region,
    image_size=model_params.image_size,
    temperature=model_params.temperature,
    world_size=model_params.world_size,
    rank=model_params.rank,
    cfg=model_params.cfg,
    img_enc_path=model_params.img_enc_path,
    gene_enc_path=model_params.gene_enc_path,
    drvi_model_dir=model_params.drvi_model_dir,
)

# Move to device (use first GPU for test)
device = torch.device('cuda:0')
model.to(device)
model.eval()

# Forward pass on the real batch (move batch to device)
batch = {k: v.to(device) for k, v in batch.items()}
with torch.no_grad():
    img_embed, gexp_embed, logit_scale = model(batch, norm=True)

print(f"Image Embedding Dimension: {img_embed.shape}")
print(f"Gene Expression Embedding Dimension: {gexp_embed.shape}")
print(f"Logit Scale: {logit_scale.item():.2f}")

# REPRODUCE THE BUG: Manually call compute_loss to trigger the matmul error
print("\n=== Reproducing the Bug in compute_loss ===")
try:
    # This will fail with the matmul error if dims mismatch
    loss_value = model.compute_loss(img_embed, gexp_embed)
    print(f"Loss computed successfully: {loss_value.item():.4f}")
except RuntimeError as e:
    print(f"Reproduced Error: {str(e)}")
    print(f"Img embed shape in error context: {img_embed.shape}")
    print(f"Gexp embed shape in error context: {gexp_embed.shape}")

# ADDITIONAL DEBUG: Simulate the exact matmul from ClipLoss
print("\n=== Simulating ClipLoss Matmul ===")
try:
    # From open_clip/loss.py: logits_per_image = logit_scale * img_features @ gexp_features.T
    scaled_img = logit_scale * img_embed
    logits = scaled_img @ gexp_embed.T
    print(f"Matmul successful: logits shape {logits.shape}")
except RuntimeError as e:
    print(f"Matmul Error: {str(e)}")
    # If it matches the bug shapes, print more details
    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
        print(f" - Img for matmul: {img_embed.shape} (rows x cols)")
        print(f" - Gexp.T for matmul: {gexp_embed.T.shape} (cols x rows)")
        print(f"Inner dims mismatch: {img_embed.shape[-1]} != {gexp_embed.shape[-1]}")