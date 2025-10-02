from datasets import load_dataset
import pandas as pd


from huggingface_hub import login

# 用你在 Hugging Face 账号里生成的 token 登录




# 会自动下载全部 802 个 parquet 分片到本地缓存
dataset = load_dataset("Peng-AI/hescape-pyarrow", name="human-lung-healthy-panel", split="train")


dataset.save_to_disk("C:/Users/ioone/PycharmProjects/moemodified/hescape/datasets/human_breast_panel")
