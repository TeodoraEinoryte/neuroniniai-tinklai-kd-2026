"""Used once to create the extra data files from the Lithuanian CARER corpus.
Normally not needed: data.py reads the CSVs.

Source: SkyWater21/lt_twitter_emotions on Hugging Face (MIT licence), the whole CARER corpus (416,809 rows)
with the English text, its Lithuanian machine translation and the same 6 labels (see download_lt_test.py).

Writes:
  data/lt_train.csv, data/lt_validation.csv   our train and validation texts in Lithuanian (S15), in the same row
                                              order as emotion_train.csv / emotion_validation.csv
  data/carer_select.csv                       selection set, 20,000 other CARER rows: all choices in T0
  data/carer_xtest.csv                        extended test, 50,000 other CARER rows: confidence intervals
Columns: text_lt, text_en, label.

The two extra sets are drawn the way the official 20,000 seem to be: random ROWS of the corpus, so a text that
occurs several times (often with different labels) is as likely to be drawn as there. A first version drew unique
texts and dropped those with two labels; it had half as many ambiguous texts as the official test (7.0 % vs 13.9 %)
and every model scored ~7 points higher on it. Here: no text from our 20,000, the class shares
of the official test, and no text in both sets.

Run: python download_carer_extra.py
"""
import os

import numpy as np
import pandas as pd
from datasets import load_dataset

import config
from data import clean_text, keyword_groups, load_emotion

SELECT_SIZE = 20000
XTEST_SIZE = 50000
SEED = 1


def path(filename):
    return os.path.join(config.DATA_DIR, filename)


def lithuanian_split(corpus, split):
    """Our train or validation texts with their Lithuanian translation, matched by English text and label."""
    ours = pd.read_csv(path(f"emotion_{split}.csv")).rename(columns={"text": "text_en"})
    unique = corpus.drop_duplicates(subset=["text_en", "label"])          # one translation per text and label
    merged = ours.merge(unique[["text_en", "label", "text_lt"]], on=["text_en", "label"], how="left")
    assert merged["text_lt"].notna().all(), f"{split}: some texts have no translation"
    assert len(merged) == len(ours)
    return merged[["text_lt", "text_en", "label"]]


def draw_sets(pool, shares, rng):
    """Random rows of pool for the selection set and the extended test, with the given class shares;
    a text drawn into one set is never drawn into the other."""
    need = {name: {label: round(size * share) for label, share in shares.items()}
            for name, size in [("select", SELECT_SIZE), ("xtest", XTEST_SIZE)]}
    rows = {"select": [], "xtest": []}
    texts = {"select": set(), "xtest": set()}
    for i, row in enumerate(pool.iloc[rng.permutation(len(pool))].itertuples()):
        for name, other in [("select", "xtest"), ("xtest", "select")]:
            if need[name][row.label] > 0 and row.clean not in texts[other]:
                need[name][row.label] -= 1
                rows[name].append(row.Index)
                texts[name].add(row.clean)
                break
    assert all(n == 0 for part in need.values() for n in part.values()), need
    return pool.loc[rows["select"]], pool.loc[rows["xtest"]]


if __name__ == "__main__":
    ds = load_dataset("SkyWater21/lt_twitter_emotions")
    corpus = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
    corpus = corpus.rename(columns={"text": "text_en", "lt_text": "text_lt", "labels": "label"})

    # 1. Our train and validation texts in Lithuanian (the test texts are in lt_test.csv already)
    for split, name in [("train", "lt_train.csv"), ("validation", "lt_validation.csv")]:
        lithuanian_split(corpus, split).to_csv(path(name), index=False)
        print(f"{name}: {len(pd.read_csv(path(name)))} tekstų")

    # 2. The other CARER rows: every row whose text (after cleaning) is not among our 20,000
    ours = pd.concat([pd.read_csv(path(f"emotion_{s}.csv")) for s in ["train", "validation", "test"]])
    corpus["clean"] = corpus["text_en"].map(clean_text)
    pool = corpus[~corpus["clean"].isin(set(ours["text"].map(clean_text)))]
    print(f"Kitų CARER eilučių: {len(pool)} (skirtingų tekstų {pool['clean'].nunique()})")

    # 3. Selection set and extended test, with the official test's class shares
    shares = pd.read_csv(path("emotion_test.csv"))["label"].value_counts(normalize=True).sort_index().to_dict()
    select, xtest = draw_sets(pool, shares, np.random.default_rng(SEED))
    e = load_emotion()
    for name, df in [("carer_select.csv", select), ("carer_xtest.csv", xtest)]:
        df = df.sample(frac=1, random_state=SEED)              # mix the classes
        df[["text_lt", "text_en", "label"]].to_csv(path(name), index=False)
        groups = keyword_groups(df["clean"].to_numpy(), e["x_train"], e["y_train"])
        group_text = ", ".join(f"{g} {(groups == g).mean():.1%}" for g in ["pure", "ambiguous", "none", "unseen"])
        print(f"{name}: {len(df)} eilučių; emocijos žodžio grupės: {group_text}")
    groups = keyword_groups(e["x_test"], e["x_train"], e["y_train"])
    print("palyginimui, oficiali testavimo imtis:",
          ", ".join(f"{g} {(groups == g).mean():.1%}" for g in ["pure", "ambiguous", "none", "unseen"]))
