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

# The emotion word (keyword rule in data_analysis.py, keyword test S13): the first word after a feel-word,
# skipping intensifiers and small words, so "i feel so very lonely" -> "lonely". If that word is a pronoun,
# conjunction, preposition or auxiliary ("i feel like i ...", "i feel as if ..."), the text has no emotion word.
# The lists were checked on the validation set: keyword rule 71.0 % (first version) -> 73.0 %.
FEEL_WORDS = {"feel", "feels", "feeling", "felt"}
SKIP_WORDS = {"so", "very", "really", "a", "an", "bit", "little", "more", "quite", "kind", "sort", "of", "like", "that",
              "too", "pretty", "extremely", "incredibly", "rather", "somewhat", "even", "just", "most", "less", "the",
              "all", "completely", "totally", "slightly", "particularly", "absolutely", "especially", "utterly",
              "strongly", "deeply", "terribly", "almost", "tad", "strangely", "overly", "being", "kinda", "truly",
              "super", "fairly", "genuinely", "seriously", "increasingly", "awfully", "such", "lot", "much"}
NOT_EMOTION_WORDS = {"i", "im", "ive", "me", "my", "myself", "you", "your", "he", "hes", "him", "his", "she", "her",
                     "it", "its", "we", "us", "our", "they", "them", "their", "this", "these", "those", "there",
                     "theres", "someone", "one", "as", "and", "but", "or", "if", "when", "because", "though", "than",
                     "how", "what", "who", "about", "to", "in", "for", "at", "on", "out", "from", "with", "some", "any",
                     "every", "only", "no", "not", "is", "was", "were", "be", "have", "had", "has", "do", "did",
                     "would", "could", "can", "will"}
UNKNOWN_WORD = "xxunknownxx"    # not in the vocabulary, so it is encoded as [UNK] (keyword test, S13)


def clean_text(text):
    """Preprocess like dair-ai/emotion: lowercase, no punctuation ("Don't!" -> "dont").

    Letters of any alphabet are kept, so Lithuanian sentences stay readable ("Šiandien" -> "šiandien").
    """
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^\w ]|_", " ", text)        # every symbol that is not a letter or digit becomes a space
    return re.sub(r" +", " ", text).strip()


def emotion_word(text):
    """Position of the emotion word in a cleaned text, or None if the text has none.

    "i feel so humiliated today" -> 3 ("humiliated"); the word itself is text.split()[position].
    "i feel like i lost something" -> None ("i" is not an emotion word); "nothing here" -> None.
    """
    words = text.split()
    for i, word in enumerate(words[:-1]):
        if word in FEEL_WORDS:
            j = i + 1
            while j < len(words) - 1 and words[j] in SKIP_WORDS:   # "so", "very", "a little" ... are skipped
                j += 1
            return None if words[j] in NOT_EMOTION_WORDS else j
    return None


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


def load_lt_test(task="6"):
    """Our 2,000 test texts machine-translated into Lithuanian (download_lt_test.py); for testing only.

    lt_test.csv columns: test_index, text_lt, text_en, label (0-5, as in emotion_test.csv).
    Returns (x_lt, x_en, y, test_index); test_index is the row of the same text in emotion_test.csv,
    so Lithuanian and English predictions can be compared text by text.
    """
    df = pd.read_csv(_path("lt_test.csv"))
    x_lt = df["text_lt"].map(clean_text).to_numpy()      # "Aš jaučiuosi pasipiktinęs!" -> "aš jaučiuosi pasipiktinęs"
    x_en = df["text_en"].map(clean_text).to_numpy()
    return x_lt, x_en, _labels(df["label"].to_numpy(), task), df["test_index"].to_numpy()


def keyword_variants(texts):
    """Two copies of cleaned texts for the keyword test (S13), built around the emotion word (emotion_word):

    "without": the emotion word replaced by UNKNOWN_WORD, which the vectorizer encodes as [UNK]
               "i feel so humiliated today" -> "i feel so xxunknownxx today"
    "only":    the emotion word alone; "humiliated"
    Texts without an emotion word stay unchanged in "without" and are empty in "only".
    Also returns "keyword" (the word, or "") and "has_keyword" (bool), all as np.arrays in the order of texts.
    """
    without, only, keyword = [], [], []
    for text in texts:
        words = text.split()
        i = emotion_word(text)
        if i is None:
            without.append(text)
            only.append("")
            keyword.append("")
        else:
            without.append(" ".join(words[:i] + [UNKNOWN_WORD] + words[i + 1:]))
            only.append(words[i])
            keyword.append(words[i])
    keyword = np.array(keyword)
    return {"without": np.array(without), "only": np.array(only), "keyword": keyword, "has_keyword": keyword != ""}


def test_in_train(d):
    """True for each test text whose (cleaned) text is also in train: the duplicates check of the main table."""
    train = set(d["x_train"])
    return np.array([text in train for text in d["x_test"]])


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
    """Encode any texts (e.g. GoEmotions, the LT test set, keyword-test copies) with an already fitted vectorizer."""
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

    # Keyword test (S13) on hand-made examples
    kw = keyword_variants(np.array(["i feel so humiliated today", "i didnt feel humiliated", "i feel very very sad",
                                    "feeling slightly more optimistic", "a feeling of being invaded",
                                    "i feel like i lost it", "nothing to see here", "i feel"]))
    print("Be raktinio žodžio:", kw["without"].tolist())
    print("Tik raktinis žodis:", kw["only"].tolist())
    assert list(kw["without"]) == ["i feel so xxunknownxx today", "i didnt feel xxunknownxx",
                                   "i feel very very xxunknownxx", "feeling slightly more xxunknownxx",
                                   "a feeling of being xxunknownxx", "i feel like i lost it", "nothing to see here",
                                   "i feel"]
    assert list(kw["only"]) == ["humiliated", "humiliated", "sad", "optimistic", "invaded", "", "", ""]
    assert list(kw["has_keyword"]) == [True, True, True, True, True, False, False, False]
    assert UNKNOWN_WORD not in d["vectorizer"].get_vocabulary()      # so it really becomes [UNK]
    assert encode(d["vectorizer"], kw["without"][:1])[0][3] == 1     # 4th word -> index 1 = [UNK]

    # Duplicates check on a hand-made example: only "b" is in both train and test
    assert list(test_in_train({"x_train": np.array(["a", "b"]), "x_test": np.array(["b", "c"])})) == [True, False]
    print("Testavimo tekstų, kurie yra ir mokymo imtyje:", test_in_train(load_emotion()).sum())
    print("Patikrinimai sutampa.")
