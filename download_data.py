"""Used once to create the CSV files in data/ from Hugging Face. Normally not needed: data.py reads the CSV files.

Run: python download_data.py
"""
import json
import os

from datasets import load_dataset

import config
from data import GOEMOTIONS_LABELS


def path(filename):
    return os.path.join(config.DATA_DIR, filename)


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # dair-ai/emotion, saved unchanged (columns: text, label)
    emotion = load_dataset("dair-ai/emotion")
    for split in ["train", "validation", "test"]:
        emotion[split].to_pandas().to_csv(path(f"emotion_{split}.csv"), index=False)
        print("Išsaugota:", path(f"emotion_{split}.csv"))

    # GoEmotions, saved unchanged (columns: text, labels, id); filtering happens in data.load_goemotions()
    go = load_dataset("google-research-datasets/go_emotions", "simplified")
    assert go["train"].features["labels"].feature.names == GOEMOTIONS_LABELS
    for split in ["train", "validation", "test"]:
        df = go[split].to_pandas()
        df["labels"] = df["labels"].map(lambda labels: json.dumps(labels.tolist()))   # [14] or [3, 17]
        df.to_csv(path(f"goemotions_{split}.csv"), index=False)
        print("Išsaugota:", path(f"goemotions_{split}.csv"))
