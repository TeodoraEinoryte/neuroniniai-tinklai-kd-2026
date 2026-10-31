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


def load_emotion(task="6", fraction=1.0, seed=1, lang="en"):
    """Load dair-ai/emotion as numpy arrays of texts (x) and labels (y).

    task: "6" = six emotions, "2" = positive/negative (S5).
    fraction: stratified share of train to keep (S4); validation and test are never reduced.
    lang: "en", or "lt" = the same texts machine-translated into Lithuanian (S15; lt_*.csv, same row order).
    """
    d = {}
    for name, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        if lang == "lt":
            df = pd.read_csv(_path(f"lt_{split}.csv")).rename(columns={"text_lt": "text"})
        else:
            df = pd.read_csv(_path(f"emotion_{split}.csv"))
        if name == "train" and fraction < 1.0:
            # the same share from every class, so class proportions stay the same
            df = df.groupby("label").sample(frac=fraction, random_state=seed)
        d[f"x_{name}"] = df["text"].map(clean_text).to_numpy()   # English is already clean; kept for consistency
        d[f"y_{name}"] = _labels(df["label"].to_numpy(), task)
    return d


def load_carer_extra(part, task="6", lang="en"):
    """Extra CARER texts, never trained on (download_carer_extra.py, PLAN.md section 3):
    part "select" = selection set (20,000; all choices in T0), "xtest" = extended test (50,000).
    Returns (x, y); lang "lt" gives the Lithuanian translation of the same texts."""
    df = pd.read_csv(_path(f"carer_{part}.csv"))
    x = df["text_lt" if lang == "lt" else "text_en"].map(clean_text).to_numpy()
    return x, _labels(df["label"].to_numpy(), task)


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


def load_goemotions_splits(task="6"):
    """GoEmotions with our 6 emotions, by its own split (5,272 / 625 / 638 texts), for training on it (S17).
    Returns the same dict as load_emotion."""
    d = {}
    for name, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        df = pd.read_csv(_path(f"goemotions_{split}.csv"))
        df["names"] = df["labels"].map(lambda labels: [GOEMOTIONS_LABELS[i] for i in json.loads(labels)])
        df = df[goemotions_keep(df["names"])]
        d[f"x_{name}"] = df["text"].map(clean_text).to_numpy()
        d[f"y_{name}"] = _labels(df["names"].map(lambda n: config.CLASSES_EN.index(n[0])).to_numpy(), task)
    return d


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


def keyword_variants(texts, vocabulary=None, seed=0):
    """Copies of cleaned texts for the keyword test (S13), built around the emotion word (emotion_word):

    "without": every occurrence of the emotion word replaced by UNKNOWN_WORD, which the vectorizer encodes as [UNK]
               "i feel crappy i eat crappy" -> "i feel xxunknownxx i eat xxunknownxx"
    "only":    the emotion word alone; "crappy"
    "i_feel":  "i feel" + the emotion word, so the input is not unusually short; "i feel crappy"
    "control": one random other word of the text replaced by UNKNOWN_WORD instead, to separate
               "lost the emotion word" from "got an [UNK]"
    A text counts as a keyword text only if it has an emotion word and, when vocabulary (a set of words) is given,
    the word is in it; otherwise replacing it would change nothing. Other texts stay unchanged in "without" and
    "control" and are empty in "only" and "i_feel".
    Also returns "keyword" (the word, or "") and "has_keyword" (bool), all as np.arrays in the order of texts.
    """
    rng = np.random.default_rng(seed)
    without, only, i_feel, control, keyword = [], [], [], [], []
    for text in texts:
        words = text.split()
        i = emotion_word(text)
        word = words[i] if i is not None else ""
        if word == "" or (vocabulary is not None and word not in vocabulary):
            without.append(text)
            only.append("")
            i_feel.append("")
            control.append(text)
            keyword.append("")
            continue
        without.append(" ".join(UNKNOWN_WORD if w == word else w for w in words))
        only.append(word)
        i_feel.append(f"i feel {word}")
        others = [j for j, w in enumerate(words) if w != word and (vocabulary is None or w in vocabulary)]
        if others:
            j = others[rng.integers(len(others))]
            words = words[:j] + [UNKNOWN_WORD] + words[j + 1:]
        control.append(" ".join(words))
        keyword.append(word)
    keyword = np.array(keyword)
    return {"without": np.array(without), "only": np.array(only), "i_feel": np.array(i_feel),
            "control": np.array(control), "keyword": keyword, "has_keyword": keyword != ""}


def keyword_counts(train_texts, train_labels):
    """For every emotion word in train: how many train texts of each class it is the emotion word of."""
    counts = {}
    for text, label in zip(train_texts, train_labels):
        i = emotion_word(text)
        if i is not None:
            word = text.split()[i]
            counts.setdefault(word, np.zeros(6, dtype=int))[label] += 1
    return counts


def keyword_groups(texts, train_texts, train_labels, purity=0.95):
    """Group of each text by its emotion word (S13; PLAN.md section 3):
    "pure"      the word's train texts have its most common label at least `purity` of the time
    "ambiguous" the word is an emotion word in train, but less pure
    "none"      the text has no emotion word
    "unseen"    the word never is an emotion word in train
    """
    counts = keyword_counts(train_texts, train_labels)
    groups = []
    for text in texts:
        i = emotion_word(text)
        if i is None:
            groups.append("none")
            continue
        c = counts.get(text.split()[i])
        if c is None:
            groups.append("unseen")
        else:
            groups.append("pure" if c.max() / c.sum() >= purity else "ambiguous")
    return np.array(groups)


def negation_templates(train_texts, train_labels, per_class=10, purity=0.95):
    """Negation test sentences (S13, CheckList): the per_class most frequent pure emotion words of each class
    (the same purity as keyword_groups) in "i feel X", "i dont feel X" and "i never feel X".
    Returns a DataFrame with columns word, label, template, text; no gold label is needed for the test."""
    counts = keyword_counts(train_texts, train_labels)
    rows = []
    for label in range(6):
        pure = [(c.sum(), w) for w, c in counts.items() if c.argmax() == label and c.max() / c.sum() >= purity]
        for _, word in sorted(pure, reverse=True)[:per_class]:
            for template in ["i feel {}", "i dont feel {}", "i never feel {}"]:
                rows.append({"word": word, "label": label, "template": template, "text": template.format(word)})
    return pd.DataFrame(rows)


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
                                    "i feel like i lost it", "nothing to see here", "i feel",
                                    "i feel crappy i eat crappy"]))
    print("Be raktinio žodžio:", kw["without"].tolist())
    print("Tik raktinis žodis:", kw["only"].tolist())
    assert list(kw["without"]) == ["i feel so xxunknownxx today", "i didnt feel xxunknownxx",
                                   "i feel very very xxunknownxx", "feeling slightly more xxunknownxx",
                                   "a feeling of being xxunknownxx", "i feel like i lost it", "nothing to see here",
                                   "i feel", "i feel xxunknownxx i eat xxunknownxx"]   # every occurrence
    assert list(kw["only"]) == ["humiliated", "humiliated", "sad", "optimistic", "invaded", "", "", "", "crappy"]
    assert kw["i_feel"][0] == "i feel humiliated" and kw["i_feel"][5] == ""
    assert list(kw["has_keyword"]) == [True, True, True, True, True, False, False, False, True]
    # control: exactly one other word becomes the unknown word, the emotion word stays
    control = kw["control"][0].split()
    assert sum(a != b for a, b in zip(control, "i feel so humiliated today".split())) == 1
    assert UNKNOWN_WORD in control and "humiliated" in control
    # a word outside the vocabulary would be [UNK] in both copies, so the text does not count as a keyword text
    small = keyword_variants(np.array(["i feel so humiliated today"]), vocabulary={"i", "feel", "so", "today"})
    assert not small["has_keyword"][0] and small["without"][0] == "i feel so humiliated today"
    assert UNKNOWN_WORD not in d["vectorizer"].get_vocabulary()      # so it really becomes [UNK]
    assert encode(d["vectorizer"], kw["without"][:1])[0][3] == 1     # 4th word -> index 1 = [UNK]

    # Keyword groups and negation templates on a hand-made train set: "sad" always class 0, "odd" half 0 half 1
    train_x, train_y = np.array(["i feel sad", "i feel sad", "i feel odd", "i feel odd"]), np.array([0, 0, 0, 1])
    groups = keyword_groups(np.array(["i feel sad today", "i feel odd", "nothing here", "i feel happy"]),
                            train_x, train_y)
    assert list(groups) == ["pure", "ambiguous", "none", "unseen"]
    templates = negation_templates(train_x, train_y, per_class=1)
    assert list(templates["text"]) == ["i feel sad", "i dont feel sad", "i never feel sad"]   # "odd" is not pure
    e = load_emotion()
    groups = keyword_groups(e["x_test"], e["x_train"], e["y_train"])
    print("Raktinio žodžio grupės (testavimo imtis):",
          {g: f"{(groups == g).mean():.1%}" for g in ["pure", "ambiguous", "none", "unseen"]})

    # Duplicates check on a hand-made example: only "b" is in both train and test
    assert list(test_in_train({"x_train": np.array(["a", "b"]), "x_test": np.array(["b", "c"])})) == [True, False]
    print("Testavimo tekstų, kurie yra ir mokymo imtyje:", test_in_train(load_emotion()).sum())
    print("Patikrinimai sutampa.")
