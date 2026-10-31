"""Used once to create the extra data files from the Lithuanian CARER corpus (PLAN.md, section 3, step 1.9).
Normally not needed: data.py reads the CSVs.

Source: SkyWater21/lt_twitter_emotions on Hugging Face (MIT licence), the whole CARER corpus (416,809 texts)
with the English text, its Lithuanian machine translation and the same 6 labels (see download_lt_test.py).

Writes:
  data/lt_train.csv, data/lt_validation.csv   our train and validation texts in Lithuanian (S15), in the same row
                                              order as emotion_train.csv / emotion_validation.csv
  data/carer_select.csv                       selection set, 20,000 other CARER texts: all choices in T0
  data/carer_xtest.csv                        extended test, 50,000 other CARER texts: confidence intervals
Columns: text_lt, text_en, label. The two extra sets never overlap our 20,000 texts, carry one label each,
have no near-duplicate in train, and have the class shares of the official test set.

Run: python download_carer_extra.py
"""
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset

import config
from data import clean_text

SELECT_SIZE = 20000
XTEST_SIZE = 50000
JACCARD = 0.8          # a text this similar to a train text (shared words / all words) counts as a near-duplicate
SEED = 1


def path(filename):
    return os.path.join(config.DATA_DIR, filename)


def lithuanian_split(corpus, split):
    """Our train or validation texts with their Lithuanian translation, matched by English text and label."""
    ours = pd.read_csv(path(f"emotion_{split}.csv")).rename(columns={"text": "text_en"})
    merged = ours.merge(corpus[["text_en", "label", "text_lt"]], on=["text_en", "label"], how="left")
    assert merged["text_lt"].notna().all(), f"{split}: some texts have no translation"
    assert len(merged) == len(ours)                       # the corpus was deduplicated, so no row is doubled
    return merged[["text_lt", "text_en", "label"]]


def near_duplicates(texts, train_texts):
    """True for each text whose word set has Jaccard similarity >= JACCARD with some train text.

    Checking every pair (350,000 x 16,000) would be slow, so we use the prefix rule: sort each text's words
    from rarest to most common; two sets with Jaccard >= t must share a word among the first
    len(set) - ceil(t * len(set)) + 1 words of each. Only train texts found that way are compared exactly.
    """
    sets = [set(t.split()) for t in texts]
    train_sets = [set(t.split()) for t in train_texts]
    frequency = Counter(w for s in train_sets + sets for w in s)

    def prefix(s):
        words = sorted(s, key=lambda w: (frequency[w], w))
        return words[:len(s) - int(np.ceil(JACCARD * len(s))) + 1]

    index = defaultdict(list)                  # word -> train texts that have it in their prefix
    for j, s in enumerate(train_sets):
        for w in prefix(s):
            index[w].append(j)
    result = np.zeros(len(sets), dtype=bool)
    for i, s in enumerate(sets):
        candidates = {j for w in prefix(s) for j in index[w]}
        result[i] = any(len(s & train_sets[j]) / len(s | train_sets[j]) >= JACCARD for j in candidates)
    return result


def stratified(pool, n, shares, rng):
    """n texts from pool with the given class shares (dict label -> share)."""
    parts = []
    for label, share in shares.items():
        rows = pool[pool["label"] == label]
        parts.append(rows.iloc[rng.permutation(len(rows))[:round(n * share)]])
    return pd.concat(parts)


if __name__ == "__main__":
    ds = load_dataset("SkyWater21/lt_twitter_emotions")
    corpus = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
    corpus = corpus.rename(columns={"text": "text_en", "lt_text": "text_lt", "labels": "label"})
    corpus = corpus.drop_duplicates(subset=["text_en", "label"])

    # 1. Our train and validation texts in Lithuanian (the test texts are in lt_test.csv already)
    for split, name in [("train", "lt_train.csv"), ("validation", "lt_validation.csv")]:
        lithuanian_split(corpus, split).to_csv(path(name), index=False)
        print(f"{name}: {len(pd.read_csv(path(name)))} tekstų")

    # 2. The other CARER texts: not among our 20,000 (compared after cleaning), one label, no near-duplicate of train
    ours = pd.concat([pd.read_csv(path(f"emotion_{s}.csv")) for s in ["train", "validation", "test"]])
    corpus["clean"] = corpus["text_en"].map(clean_text)
    labels_per_text = corpus.groupby("clean")["label"].nunique()
    pool = corpus[~corpus["clean"].isin(set(ours["text"].map(clean_text)))]
    pool = pool[pool["clean"].map(labels_per_text) == 1].drop_duplicates("clean")
    print(f"Kitų CARER tekstų su viena žyme: {len(pool)} (tekstų su dviem skirtingomis žymėmis visame korpuse: "
          f"{(labels_per_text > 1).sum()})")
    train_clean = pd.read_csv(path("emotion_train.csv"))["text"].map(clean_text).to_numpy()
    duplicate = near_duplicates(pool["clean"].to_numpy(), train_clean)
    pool = pool[~duplicate]
    print(f"Pašalinta beveik pasikartojančių mokymo tekstus (Jaccard >= {JACCARD}): {duplicate.sum()}; liko {len(pool)}")

    # 3. Selection set and extended test: stratified to the official test's class shares, no text in both
    shares = pd.read_csv(path("emotion_test.csv"))["label"].value_counts(normalize=True).sort_index().to_dict()
    rng = np.random.default_rng(SEED)
    select = stratified(pool, SELECT_SIZE, shares, rng)
    xtest = stratified(pool.drop(select.index), XTEST_SIZE, shares, rng)
    for name, df in [("carer_select.csv", select), ("carer_xtest.csv", xtest)]:
        df = df.sample(frac=1, random_state=SEED)              # mix the classes
        df[["text_lt", "text_en", "label"]].to_csv(path(name), index=False)
        share_text = ", ".join(f"{config.CLASSES_LT[k]} {v:.1%}" for k, v in
                               df["label"].value_counts(normalize=True).sort_index().items())
        print(f"{name}: {len(df)} tekstų; {share_text}; vid. ilgis {df['clean'].str.split().map(len).mean():.1f} žodž.")
