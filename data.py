import json
import os
import re

import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config

# GoEmotions label names by index (from the dataset card); the CSV stores label indices
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]


def clean_text(text):
    """Preprocess like dair-ai/emotion: lowercase, no punctuation ("Don't!" -> "dont").

    Letters of any alphabet are kept, so Lithuanian sentences stay readable ("Šiandien" -> "šiandien").
    """
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^\w ]|_", " ", text)        # every symbol that is not a letter or digit becomes a space
    return re.sub(r" +", " ", text).strip()


def load_emotion(task="6", fraction=1.0, seed=1):
    """Load dair-ai/emotion as numpy arrays of texts (x) and labels (y).

    task: "6" = six emotions, "2" = positive/negative (S5).
    fraction: stratified share of train to keep (S4); validation and test are never reduced.
    """
    d = {}
    for name, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        df = pd.read_csv(_path(f"emotion_{split}.csv"))
        if name == "train" and fraction < 1.0:
            # the same share from every class, so class proportions stay the same
            df = df.groupby("label").sample(frac=fraction, random_state=seed)
        d[f"x_{name}"] = df["text"].map(clean_text).to_numpy()   # already clean in the source; kept for consistency
        d[f"y_{name}"] = _labels(df["label"].to_numpy(), task)
    return d


def load_goemotions_raw():
    """All GoEmotions texts (all splits) with their label names, before our filter.

    Returns a DataFrame with columns text (original) and names (list of label names).
    """
    splits = [pd.read_csv(_path(f"goemotions_{split}.csv")) for split in ["train", "validation", "test"]]
    df = pd.concat(splits, ignore_index=True)
    df["names"] = df["labels"].map(lambda labels: [GOEMOTIONS_LABELS[i] for i in json.loads(labels)])   # "[14]" -> ["fear"]
    return df[["text", "names"]]


def goemotions_keep(names):
    """Our filter: True for texts with exactly one label that is one of our 6 emotions."""
    return names.map(lambda n: len(n) == 1 and n[0] in config.CLASSES_EN)


def load_goemotions(task="6"):
    """GoEmotions texts with our 6 emotions (generalisation test only, never trained on)."""
    df = load_goemotions_raw()
    df = df[goemotions_keep(df["names"])]
    x = df["text"].map(clean_text).to_numpy()
    y = df["names"].map(lambda n: config.CLASSES_EN.index(n[0])).to_numpy()   # ["fear"] -> 4
    return x, _labels(y, task)


def load_lt(task="6"):
    """Our own Lithuanian sentences and their English translations (generalisation test only).

    lt_sentences.csv columns: text_lt, text_en, label (emotion name in English, e.g. "joy").
    Returns (x_lt, x_en, y); both texts are cleaned the same way as the training data.
    """
    df = pd.read_csv(_path("lt_sentences.csv"))
    y = df["label"].map(config.CLASSES_EN.index).to_numpy()   # "joy" -> 1
    x_lt = df["text_lt"].map(clean_text).to_numpy()
    x_en = df["text_en"].map(clean_text).to_numpy()
    return x_lt, x_en, _labels(y, task)


def vectorize(d, mode="sequence", padding="post"):
    """Turn the texts in d into model input. The vocabulary is built from train only.

    mode "sequence": (n, SEQUENCE_LENGTH) word indices, for Embedding/CNN/RNN/Transformer models.
    mode "multi_hot": (n, VOCAB_SIZE) 0/1 vectors, for logistic regression and MLP.
    padding "post" (zeros at the end) or "pre" (zeros at the start, S2).
    Returns a copy of d with the x_* replaced and the fitted "vectorizer" added.
    """
    # Index 0 = padding, 1 = [UNK] (word not among the VOCAB_SIZE most frequent)
    vectorizer = TextVectorization(max_tokens=config.VOCAB_SIZE, ragged=True)
    vectorizer.adapt(d["x_train"])

    result = dict(d, vectorizer=vectorizer)
    for key in ["x_train", "x_val", "x_test"]:
        result[key] = encode(vectorizer, d[key], mode, padding)
    return result


def encode(vectorizer, texts, mode="sequence", padding="post"):
    """Encode any texts (e.g. GoEmotions, LT/EN sentences) with an already fitted vectorizer."""
    sequences = vectorizer(texts).to_list()       # "i feel happy" -> [2, 3, 57]
    if mode == "sequence":
        # truncating is always "post" (keep the first words), so S2 changes only the padding side
        return pad_sequences(sequences, maxlen=config.SEQUENCE_LENGTH, padding=padding, truncating="post")
    if mode == "multi_hot":
        x = np.zeros((len(sequences), config.VOCAB_SIZE), dtype=np.uint8)   # uint8: 16000 x 10000 = 152 MB
        for i, sequence in enumerate(sequences):
            x[i, sequence] = 1                    # 1 for every word that occurs in the text
        x[:, 0] = 0                               # column 0 is padding, never a word
        return x
    raise ValueError(f"Unknown mode: {mode}")


def _labels(y, task):
    if task == "2":
        return np.isin(y, config.POSITIVE_CLASSES).astype(int)
    return y


def _path(filename):
    return os.path.join(config.DATA_DIR, filename)


if __name__ == "__main__":
    d = vectorize(load_emotion())
    print("Masyvų dydžiai:", {k: v.shape for k, v in d.items() if k.startswith(("x_", "y_"))})
    print("Žodyno dydis:", d["vectorizer"].vocabulary_size())
