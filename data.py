import os
from collections import Counter

from datasets import load_dataset
import pandas as pd

# Emotion names by label index (from the dataset card)
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Vocabulary size: train has ~15k unique words; the 10k most frequent cover ~98% of all
# word occurrences, and over half of the words appear only once (see vocab_coverage.csv).
NUM_WORDS = 10000

# Sequence length: 98.7% of train texts have <= 50 words (longest is 66), so only a few
# texts lose their last words; longer sequences would slow down RNN training (see length_coverage.csv).
MAXLEN = 50


def load_data():
    """Download dair-ai/emotion and return train, validation and test splits as pandas DataFrames."""
    ds = load_dataset("dair-ai/emotion")  # downloads on first run, then uses the local cache
    train = ds["train"].to_pandas()       # columns: text, label
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()
    return train, val, test


def vocab_coverage(texts, sizes):
    """For each vocabulary size N, the share of word occurrences covered by the N most frequent words."""
    counts = Counter(word for text in texts for word in text.split())
    freqs = [n for _, n in counts.most_common()]   # word frequencies, most frequent first
    total = sum(freqs)
    rows = [{"vocab_size": n, "coverage_%": round(sum(freqs[:n]) / total * 100, 1)} for n in sizes]
    print("Unique words:", len(counts), "| appearing once:", sum(1 for n in freqs if n == 1))
    return pd.DataFrame(rows)


def length_coverage(texts, lengths):
    """For each maxlen L, the share of texts that fit fully (have <= L words)."""
    n_words = texts.str.split().str.len()
    rows = [{"maxlen": l, "texts_fit_%": round((n_words <= l).mean() * 100, 1)} for l in lengths]
    print(f"Words per text: mean {n_words.mean():.1f}, median {n_words.median():.0f}, max {n_words.max()}")
    return pd.DataFrame(rows)


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

    # Statistics behind NUM_WORDS and MAXLEN (computed on train only)
    vocab = vocab_coverage(train["text"], [1000, 2000, 5000, 8000, 10000, 15000])
    print(vocab)
    vocab.to_csv("results/tables/vocab_coverage.csv", index=False)

    lengths = length_coverage(train["text"], [30, 40, 50, 60, 66])
    print(lengths)
    lengths.to_csv("results/tables/length_coverage.csv", index=False)
    print("Saved results/tables/vocab_coverage.csv, results/tables/length_coverage.csv")
