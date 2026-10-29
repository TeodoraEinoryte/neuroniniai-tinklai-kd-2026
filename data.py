import os

from datasets import load_dataset
import pandas as pd

# Emotion names by label index (from the dataset card)
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_data():
    """Download dair-ai/emotion and return train, validation and test splits as pandas DataFrames."""
    ds = load_dataset("dair-ai/emotion")  # downloads on first run, then uses the local cache
    train = ds["train"].to_pandas()       # columns: text, label
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()
    return train, val, test


if __name__ == "__main__":
    train, val, test = load_data()

    print("Split sizes:", len(train), len(val), len(test))
    print(train.head())

    # Number of examples per emotion in each split
    splits = {"train": train, "validation": val, "test": test}
    counts = pd.DataFrame({name: df["label"].value_counts() for name, df in splits.items()})
    counts.index = [EMOTIONS[i] for i in counts.index]
    counts = counts.sort_values("train", ascending=False)
    counts["total"] = counts.sum(axis=1)
    counts["share_%"] = (counts["total"] / counts["total"].sum() * 100).round(1)
    counts.loc["total"] = counts.sum()
    print(counts)

    # Save the table for the report
    os.makedirs("results/tables", exist_ok=True)
    counts.to_csv("results/tables/class_distribution.csv", index_label="emotion")
    print("Saved results/tables/class_distribution.csv")
