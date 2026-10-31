"""Emotion of one sentence with the app's models (PLAN.md section 7, step 4.4); app.py uses the same code.

Run: python predict.py "i feel so alone tonight"
     python predict.py --lang lt "šiandien jaučiuosi labai vieniša"
     python predict.py --save-models      once: copies the app's models into results/models/app/ (committed)
The app's models: English = the best S1 model on the selection set, BiLSTM + attention (it also shows which words
it looked at); Lithuanian = the best S15 model, Conv1D. Of each, the seed with the best selection-set macro-F1.
"""
import argparse
import json
import os
import shutil

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization

import config
import models_b                                   # registers AttentionPooling, so the attention model can be loaded
from data import clean_text, encode
from train import MODELS

APP_DIR = os.path.join(config.RESULTS_DIR, "models", "app")
APP_MODELS = {"en": ("S1", "bilstm_attention"), "lt": ("S15", "conv1d")}   # language: (study, model)
UNKNOWN_WARNING = 0.3     # warn when at least this share of the words is unknown to the model


def save_app_models():
    """Copy the best seed of each app model (by selection-set macro-F1) with its vocabulary into APP_DIR, and
    write app_models.json with where it came from and its scores."""
    os.makedirs(APP_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "results.csv"))
    info = {}
    for lang, (study, model) in APP_MODELS.items():
        best = df[(df["study"] == study) & (df["model"] == model)].sort_values("sel_macro_f1").iloc[-1]
        run_id = f"{study}_{model}_{best['variant']}_seed{best['seed']}"
        source = os.path.join(config.RESULTS_DIR, "models", run_id)
        shutil.copy(f"{source}.keras", os.path.join(APP_DIR, f"{lang}.keras"))
        shutil.copy(f"{source}_vocab.txt", os.path.join(APP_DIR, f"{lang}_vocab.txt"))
        info[lang] = {"run": run_id, "model": model, "sel_macro_f1": float(best["sel_macro_f1"]),
                      "test_macro_f1": float(best["test_macro_f1"]), "xtest_macro_f1": float(best["xtest_macro_f1"])}
        print(f"{lang}: {run_id} (parinkimo imtis {best['sel_macro_f1']:.4f}, testavimo {best['test_macro_f1']:.4f})")
    with open(os.path.join(APP_DIR, "app_models.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


class EmotionModel:
    """One of the app's models with its vocabulary: predict(text) gives the emotion probabilities, the attention
    weight of every word (attention model only) and the words the model does not know."""

    def __init__(self, lang):
        with open(os.path.join(APP_DIR, "app_models.json"), encoding="utf-8") as f:
            self.info = json.load(f)[lang]
        self.model = self.info["model"]
        self.name = config.MODEL_NAMES_LT[self.model]
        with open(os.path.join(APP_DIR, f"{lang}_vocab.txt"), encoding="utf-8") as f:
            vocabulary = f.read().split("\n")
        self.vocabulary = set(vocabulary)
        self.vectorizer = TextVectorization(vocabulary=vocabulary[2:], ragged=True)   # Keras adds "" and "[UNK]"
        self.net = keras.models.load_model(os.path.join(APP_DIR, f"{lang}.keras"))
        self.has_attention = self.model == "bilstm_attention"

    def predict(self, text):
        """Returns a dict: probabilities {emotion in Lithuanian: p}, words (cleaned), attention (one weight per
        word, or None), unknown (words not in the vocabulary, i.e. [UNK] for the model), unknown_share."""
        words = clean_text(text).split()[:config.SEQUENCE_LENGTH]     # the model reads at most 50 words
        cleaned = " ".join(words)
        x = encode(self.vectorizer, np.array([cleaned]), MODELS[self.model][1])
        probabilities = self.net.predict(x, verbose=0)[0]
        attention = models_b.attention_weights(self.net, x)[0][:len(words)] if self.has_attention and words else None
        unknown = [w for w in words if w not in self.vocabulary]
        return {"probabilities": dict(zip(config.CLASSES_LT, probabilities.tolist())), "words": words,
                "attention": attention, "unknown": unknown, "unknown_share": len(unknown) / len(words) if words else 0.0}


def print_result(result, name):
    print(f"Modelis: {name}")
    for emotion, p in sorted(result["probabilities"].items(), key=lambda item: -item[1]):
        print(f"  {emotion:11s} {p:6.1%} {'█' * round(p * 40)}")
    if result["attention"] is not None:
        print("Dėmesio svoriai:", "  ".join(f"{w} {a:.2f}" for w, a in zip(result["words"], result["attention"])))
    if result["unknown_share"] >= UNKNOWN_WARNING:
        print(f"Dėmesio: {result['unknown_share']:.0%} žodžių modeliui nežinomi ({', '.join(result['unknown'])}), "
              f"todėl rezultatas nepatikimas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion of a sentence.")
    parser.add_argument("text", nargs="?", help="the sentence")
    parser.add_argument("--lang", choices=sorted(APP_MODELS), default="en")
    parser.add_argument("--save-models", action="store_true", help="copy the app's models into results/models/app/")
    args = parser.parse_args()
    if args.save_models:
        save_app_models()
    if args.text:
        model = EmotionModel(args.lang)
        print_result(model.predict(args.text), model.name)
