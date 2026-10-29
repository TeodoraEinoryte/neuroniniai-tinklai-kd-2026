import os
from collections import Counter

import pandas as pd

import config
from data import load

TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")


def class_distribution(d):
    """Number of examples per emotion in each split, plus total and share."""
    splits = {"mokymo": d["y_train"], "validavimo": d["y_val"], "testavimo": d["y_test"]}
    table = pd.DataFrame({name: pd.Series(y).value_counts() for name, y in splits.items()})
    table.index = [config.CLASSES_LT[i] for i in table.index]
    table = table.sort_values("mokymo", ascending=False)
    table["iš viso"] = table.sum(axis=1)
    table["dalis, %"] = (table["iš viso"] / table["iš viso"].sum() * 100).round(1)
    table.loc["iš viso"] = table.sum()
    return table


def vocab_coverage(texts, sizes):
    """For each vocabulary size N, the share of word occurrences covered by the N most frequent words."""
    counts = Counter(word for text in texts for word in text.split())
    freqs = [n for _, n in counts.most_common()]   # word frequencies, most frequent first
    total = sum(freqs)
    print("Skirtingų žodžių:", len(counts), "| pasitaiko tik kartą:", sum(1 for n in freqs if n == 1))
    return pd.DataFrame([{"žodyno dydis": n, "aprėptis, %": round(sum(freqs[:n]) / total * 100, 1)} for n in sizes])


def length_coverage(texts, lengths):
    """For each maxlen L, the share of texts that fit fully (have <= L words)."""
    n_words = pd.Series(texts).str.split().str.len()
    print(f"Žodžių tekste: vidurkis {n_words.mean():.1f}, mediana {n_words.median():.0f}, daugiausia {n_words.max()}")
    return pd.DataFrame([{"sekos ilgis": l, "telpa visas tekstas, %": round((n_words <= l).mean() * 100, 1)} for l in lengths])


def save(table, filename, **kwargs):
    table.to_csv(os.path.join(TABLES_DIR, filename), **kwargs)
    print("Išsaugota:", os.path.join(TABLES_DIR, filename))


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    d = load()
    print("Imčių dydžiai:", len(d["y_train"]), len(d["y_val"]), len(d["y_test"]))

    classes = class_distribution(d)
    print(classes)
    save(classes, "class_distribution.csv", index_label="emocija")

    # Statistics behind VOCAB_SIZE and SEQUENCE_LENGTH in config.py (computed on train only)
    vocab = vocab_coverage(d["x_train"], [1000, 2000, 5000, 8000, 10000, 15000])
    print(vocab)
    save(vocab, "vocab_coverage.csv", index=False)

    lengths = length_coverage(d["x_train"], [30, 40, 50, 60, 66])
    print(lengths)
    save(lengths, "length_coverage.csv", index=False)
