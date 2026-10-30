"""Used once to create data/lt_test.csv: our 2,000 test texts in Lithuanian. Normally not needed: data.py reads the CSV.

Source: SkyWater21/lt_twitter_emotions on Hugging Face (MIT licence), the whole CARER corpus
(dair-ai/emotion "unsplit", 416,809 texts) machine-translated into Lithuanian with Google Translate.
It keeps the English text and the same 6 labels, so our test texts can be found in it by English text and label.

Run: python download_lt_test.py
"""
import os

import pandas as pd
from datasets import load_dataset

import config


def path(filename):
    return os.path.join(config.DATA_DIR, filename)


if __name__ == "__main__":
    # all three splits of the translated corpus; columns lt_text, text (English), labels (0-5 as ours), labels_ekman
    ds = load_dataset("SkyWater21/lt_twitter_emotions")
    lt = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
    lt = lt.rename(columns={"text": "text_en", "lt_text": "text_lt", "labels": "label"})

    # the corpus has duplicates (the same English text twice), so keep one translation per text and label
    lt = lt.drop_duplicates(subset=["text_en", "label"])

    test = pd.read_csv(path("emotion_test.csv"))                  # columns text, label; the CSV row order is kept
    test = test.rename(columns={"text": "text_en"})
    test["test_index"] = range(len(test))                         # row of the same text in emotion_test.csv
    merged = test.merge(lt[["text_en", "label", "text_lt"]], on=["text_en", "label"], how="left")

    missing = merged["text_lt"].isna().sum()
    print(f"Testavimo tekstų: {len(test)}, rasta vertimų: {len(test) - missing}, nerasta: {missing}")
    merged = merged.dropna(subset=["text_lt"])

    merged[["test_index", "text_lt", "text_en", "label"]].to_csv(path("lt_test.csv"), index=False)
    print("Išsaugota:", path("lt_test.csv"))
