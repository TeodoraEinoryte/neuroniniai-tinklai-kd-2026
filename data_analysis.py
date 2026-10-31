import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import evaluation
from data import (FEEL_WORDS, emotion_word, goemotions_keep, load_emotion, load_glove, load_goemotions,
                  load_goemotions_raw, load_lt_test, vectorize)

TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")
NEGATION = re.compile(r"\b(?:not|dont|didnt|never|cant|wont|isnt|wasnt)\b")


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


def dataset_overview(d, datasets):
    """Compare all datasets side by side: size, class shares, length, feel-words and unknown words.

    datasets: {column name: (texts, labels)}. Unknown words are counted with the dair-ai train vocabulary,
    the only one our models know.
    """
    vectorizer = vectorize(d)["vectorizer"]
    columns = {}
    for name, (texts, labels) in datasets.items():
        words = pd.Series(texts).str.split()
        ids = np.concatenate(vectorizer(texts).to_list() or [[]])   # all word ids of all texts
        column = {"tekstų": len(texts)}
        for i, emotion in enumerate(config.CLASSES_LT):
            column[f"{emotion}, %"] = round((labels == i).mean() * 100, 1)
        column["vid. žodžių tekste"] = round(words.str.len().mean(), 1)
        column["tekstų su feel/feeling/felt, %"] = round(
            pd.Series(texts).str.contains(r"\b(?:feel|feels|feeling|felt)\b").mean() * 100, 1)
        column["nežinomų žodžių, %"] = round((ids == 1).mean() * 100, 1)   # 1 = [UNK]
        columns[name] = column
    return pd.DataFrame(columns)


def goemotions_filter():
    """How many GoEmotions texts our filter (data.goemotions_keep) keeps and drops, and why."""
    names = load_goemotions_raw()["names"]
    keep = goemotions_keep(names)                 # the same filter the models get
    dropped = names[~keep]

    # Reasons for dropping are only described here; data.py needs just the kept texts
    steps = pd.DataFrame({"tekstų": {
        "iš viso": len(names),
        "išmesta: kelios emocijos": int((dropped.str.len() > 1).sum()),
        "išmesta: neutral": int(dropped.map(lambda n: n == ["neutral"]).sum()),
        "išmesta: kita emocija (22 kitos)": int(dropped.map(lambda n: len(n) == 1 and n != ["neutral"]).sum()),
        "liko: viena iš mūsų 6 emocijų": int(keep.sum()),
    }})

    # Per emotion: texts with this emotion among their labels vs texts kept after the filter
    kept = names[keep].str[0]
    per_class = pd.DataFrame({
        "pažymėta šia emocija": [int(names.map(lambda n: e in n).sum()) for e in config.CLASSES_EN],
        "liko po filtro": [int((kept == e).sum()) for e in config.CLASSES_EN],
    }, index=config.CLASSES_LT)
    per_class["liko, %"] = (per_class["liko po filtro"] / per_class["pažymėta šia emocija"] * 100).round(1)
    return steps, per_class


def keyword(text):
    """The emotion word of a text (data.emotion_word), or None."""
    i = emotion_word(text)
    return text.split()[i] if i is not None else None


def keyword_rule(x_train, y_train):
    """Keyword rule: each emotion word -> its most common label in train.
    Returns predict(texts); texts without a known emotion word get the majority class."""
    counts = defaultdict(Counter)
    for text, y in zip(x_train, y_train):
        k = keyword(text)
        if k:
            counts[k][y] += 1
    rule = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    majority = Counter(y_train).most_common(1)[0][0]
    return lambda texts: np.array([rule.get(keyword(t), majority) for t in texts])


def baselines(d):
    """Two baselines every model must beat: always the majority class, and the keyword rule."""
    majority = Counter(d["y_train"]).most_common(1)[0][0]
    rule = keyword_rule(d["x_train"], d["y_train"])
    rows = {}
    for name, predict in [(f"dauguma (visada „{config.CLASSES_LT[majority]}“)", lambda x: np.full(len(x), majority)),
                          ("raktinio žodžio taisyklė", rule)]:
        row = {}
        for split, label in [("val", "validavimo"), ("test", "testavimo")]:
            m = evaluation.metrics(d[f"y_{split}"], predict(d[f"x_{split}"]), len(config.CLASSES_EN))
            row[f"{label} bendras tikslumas, %"] = round(m["accuracy"] * 100, 1)
            row[f"{label} makro F-matas"] = round(m["macro_f1"], 3)
        rows[name] = row
    return pd.DataFrame(rows).T


def quality_checks(d):
    """Problems of the dataset that go into the report: keyword labels, conflicting labels, overlap of the splits."""
    x_train = pd.Series(d["x_train"])
    everything = pd.DataFrame({"text": np.concatenate([d["x_train"], d["x_val"], d["x_test"]]),
                               "label": np.concatenate([d["y_train"], d["y_val"], d["y_test"]])})
    train_texts = set(d["x_train"])

    # words seen in >= 3 train texts that always come with the same label (pure keyword signal)
    word_labels = defaultdict(Counter)
    for text, y in zip(d["x_train"], d["y_train"]):
        for word in set(text.split()):
            word_labels[word][y] += 1
    pure = [w for w, c in word_labels.items() if sum(c.values()) >= 3 and len(c) == 1]

    checks = {
        "mokymo tekstų su feel / feels / feeling / felt, %": x_train.str.split().map(lambda w: bool(FEEL_WORDS & set(w))).mean() * 100,
        "mokymo tekstų, kuriuose rastas emocijos žodis, %": x_train.map(lambda t: emotion_word(t) is not None).mean() * 100,
        "mokymo tekstų su neiginiu (not, dont, never ...), %": x_train.str.contains(NEGATION).mean() * 100,
        "tekstų, kurie kartojasi su skirtingomis žymomis (visi 20 000)": (everything.groupby("text")["label"].nunique() > 1).sum(),
        "testavimo tekstų, kurie yra ir mokymo imtyje": sum(t in train_texts for t in d["x_test"]),
        "validavimo tekstų, kurie yra ir mokymo imtyje": sum(t in train_texts for t in d["x_val"]),
        "žodžių (≥ 3 mokymo tekstuose), kurie visada su ta pačia emocija": len(pure),
    }
    return pd.DataFrame({"reikšmė": {k: round(float(v), 1) for k, v in checks.items()}})


def emotion_words(d, n=5):
    """The n most frequent emotion words of each class in train, and how often each comes with that class.

    Only words that come with the class in more than half of their texts, so words like "i" in
    "i feel like i ..." (no emotion word after feel) do not fill the table.
    """
    counts = defaultdict(Counter)
    for text, y in zip(d["x_train"], d["y_train"]):
        k = keyword(text)
        if k:
            counts[k][y] += 1
    rows = []
    for label, emotion in enumerate(config.CLASSES_LT):
        own = [w for w in counts if counts[w][label] > sum(counts[w].values()) / 2]
        words = sorted(own, key=lambda w: counts[w][label], reverse=True)[:n]
        for w in words:
            total = sum(counts[w].values())
            rows.append({"emocija": emotion, "žodis": w, "tekstų su šia emocija": counts[w][label],
                         "visada su šia emocija, %": round(counts[w][label] / total * 100, 1)})
    return pd.DataFrame(rows)


def glove_coverage(d, x_go):
    """S12: how many of our words have a GloVe vector, per text set and for the emotion words of the test texts.

    Our cleaning merges contractions ("didnt", "im"), which GloVe splits ("did", "n't"), so such words get none.
    Also the word occurrences outside the train vocabulary (VOCAB_SIZE words; [UNK] for our models) and how many of
    them GloVe has: those are the words the "glove-extended" arm can give a vector to.
    """
    glove = load_glove()
    vocabulary = set(vectorize(d)["vectorizer"].get_vocabulary())
    test_keywords = [k for k in map(keyword, d["x_test"]) if k]
    sets = {"mokymo imtis": d["x_train"], "testavimo imtis": d["x_test"], "GoEmotions": x_go,
            "testavimo tekstų emocijos žodžiai": test_keywords}
    rows = {}
    for name, texts in sets.items():
        counts = Counter(word for text in texts for word in text.split())
        total = sum(counts.values())
        outside = {w: n for w, n in counts.items() if w not in vocabulary}
        n_outside = sum(outside.values())
        rows[name] = {
            "skirtingų žodžių": len(counts),
            "skirtingų žodžių su GloVe vektoriumi, %": round(sum(w in glove for w in counts) / len(counts) * 100, 1),
            "žodžių pasikartojimų su GloVe vektoriumi, %": round(sum(n for w, n in counts.items() if w in glove)
                                                                 / total * 100, 1),
            "pasikartojimų už mokymo žodyno ribų, %": round(n_outside / total * 100, 1),
            "iš jų su GloVe vektoriumi, %": round(sum(n for w, n in outside.items() if w in glove)
                                                  / n_outside * 100, 1) if n_outside else 0.0,
        }
    missing = Counter(w for text in d["x_train"] for w in text.split() if w not in glove)
    print("Dažniausi mokymo imties žodžiai be GloVe vektoriaus:", missing.most_common(15))
    table = pd.DataFrame(rows).T
    table["skirtingų žodžių"] = table["skirtingų žodžių"].astype(int)
    return table


def plot_class_distribution(d, path):
    """Share of each emotion in train, validation and test (bars side by side)."""
    splits = [("mokymo", d["y_train"]), ("validavimo", d["y_val"]), ("testavimo", d["y_test"])]
    x = np.arange(len(config.CLASSES_LT))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, y) in enumerate(splits):
        shares = np.bincount(y, minlength=len(x)) / len(y) * 100
        ax.bar(x + (i - 1) * 0.27, shares, width=0.27, label=f"{name} imtis")
    ax.set_xticks(x, config.CLASSES_LT)
    ax.set_ylabel("Dalis, %")
    ax.set_title("Emocijų pasiskirstymas imtyse")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_lengths(texts, path):
    """Histogram of words per train text, with the sequence length and the IQR outlier bound (lecture 1)."""
    n_words = pd.Series(texts).str.split().str.len()
    q1, q3 = n_words.quantile(0.25), n_words.quantile(0.75)
    bound = q3 + 1.5 * (q3 - q1)                  # longer texts are outliers by the quartile rule
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(n_words, bins=range(1, n_words.max() + 2), color="tab:blue", alpha=0.8)
    ax.axvline(bound, color="tab:orange", linestyle="--", label=f"Q3 + 1,5·IQR = {bound:g}")
    ax.axvline(config.SEQUENCE_LENGTH, color="tab:red", label=f"sekos ilgis = {config.SEQUENCE_LENGTH}")
    ax.set_xlabel("Žodžių tekste")
    ax.set_ylabel("Tekstų skaičius")
    ax.set_title("Tekstų ilgis mokymo imtyje")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save(table, filename, **kwargs):
    table.to_csv(os.path.join(TABLES_DIR, filename), **kwargs)
    print("Išsaugota:", os.path.join(TABLES_DIR, filename))


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    d = load_emotion()
    if sys.argv[1:] == ["glove"]:
        # python data_analysis.py glove: only the GloVe coverage (S12; needs download_glove.py first)
        coverage = glove_coverage(d, load_goemotions()[0])
        print(coverage.to_string())
        save(coverage, "glove_coverage.csv", index_label="rinkinys", lineterminator="\n")
        sys.exit()
    print("Imčių dydžiai:", len(d["y_train"]), len(d["y_val"]), len(d["y_test"]))

    classes = class_distribution(d)
    print(classes)
    save(classes, "emotion_class_distribution.csv", index_label="emocija")

    # Statistics behind VOCAB_SIZE and SEQUENCE_LENGTH in config.py (computed on train only)
    vocab = vocab_coverage(d["x_train"], [1000, 2000, 5000, 8000, 10000, 15000])
    print(vocab)
    save(vocab, "vocab_coverage.csv", index=False)

    lengths = length_coverage(d["x_train"], [30, 40, 50, 60, 66])
    print(lengths)
    save(lengths, "length_coverage.csv", index=False)

    # All three datasets side by side; the LT test set is the dair-ai test set machine-translated into Lithuanian
    x_go, y_go = load_goemotions()
    x_lt, _, y_lt, _ = load_lt_test()
    overview = dataset_overview(d, {
        "dair-ai train": (d["x_train"], d["y_train"]),
        "dair-ai validation": (d["x_val"], d["y_val"]),
        "dair-ai test": (d["x_test"], d["y_test"]),
        "GoEmotions": (x_go, y_go),
        "LT testas (vertimas)": (x_lt, y_lt),
    })
    print(overview.to_string())
    save(overview, "dataset_overview.csv", index_label="rodiklis")

    steps, per_class = goemotions_filter()
    print(steps.to_string())
    print(per_class.to_string())
    save(steps, "goemotions_filter.csv", index_label="žingsnis")
    save(per_class, "goemotions_filter_per_class.csv", index_label="emocija")

    # Data quality and baselines (report 4.1)
    checks = quality_checks(d)
    print(checks.to_string())
    save(checks, "quality_checks.csv", index_label="rodiklis")

    base = baselines(d)
    print(base.to_string())
    save(base, "baselines.csv", index_label="bazinis lygis")

    words = emotion_words(d)
    print(words.to_string(index=False))
    save(words, "emotion_words.csv", index=False)

    plot_class_distribution(d, os.path.join(FIGURES_DIR, "class_distribution.png"))
    plot_lengths(d["x_train"], os.path.join(FIGURES_DIR, "length_distribution.png"))
    print("Išsaugoti grafikai:", FIGURES_DIR)
