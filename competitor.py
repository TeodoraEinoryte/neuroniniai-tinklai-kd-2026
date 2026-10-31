"""S14 and S14b: DistilBERT fine-tuned on our train split, the competitor of our models (PLAN.md, section 6).

Run: python competitor.py --seed 1                # S14: English DistilBERT (distil_bert_base_en_uncased)
     python competitor.py --seed 1 --study S14b   # S14b: multilingual DistilBERT, trained in English, tested in Lithuanian
     python competitor.py --seed 1 --dev          # quick check: 256 train texts, 1 epoch, results/dev/
Writes the same results.csv row, learning curve, confusion matrix and prediction file as train.py, so the analysis
reads DistilBERT like our models. DistilBERT has its own word-piece tokenizer, so our TextVectorization is not used.
The pretrained weights (~265 MB English, ~540 MB multilingual) are downloaded once into the kagglehub cache.
"""
import argparse
import os
import platform
import time
import unicodedata
from datetime import datetime

import kagglehub
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import config
import evaluation
from data import (UNKNOWN_WORD, keyword_variants, load_carer_extra, load_emotion, load_goemotions, load_lt_test,
                  negation_templates)
from train import _append_row, _path

PRESETS = {"S14": "distil_bert_base_en_uncased", "S14b": "distil_bert_base_multi"}   # KerasHub presets
PRESET_VERSION = 3      # Kaggle version of both presets (2026-10-31)
MODEL_NAMES = {"S14": "distilbert", "S14b": "distilbert_multi"}
SEQUENCE_LENGTH = 64    # word pieces with [CLS] and [SEP]; cuts 0.2 % of the train texts (multilingual model: 0.9 %)
BATCH_SIZE = 32         # batch, learning rate and epochs: the usual BERT fine-tuning settings
LEARNING_RATE = 2e-5
EPOCHS = 3
PREDICT_BATCH = 64
DEV_TEXTS = 256         # --dev: train texts, and texts per evaluation set


def load_preset(study):
    """Folder with the preset files. KerasHub's own download builds the Kaggle file paths with "\\" on Windows and
    gets 404, so the whole preset is downloaded with kagglehub and loaded from the local folder. The version is
    fixed (3), so a cached preset is used without asking Kaggle for its latest version (works offline)."""
    return kagglehub.model_download(f"keras/distil_bert/keras/{PRESETS[study]}/{PRESET_VERSION}")


def encode(tokenizer, texts, cache):
    """Cleaned texts -> DistilBERT input: [CLS] word pieces [SEP], cut and padded to SEQUENCE_LENGTH.

    Done word by word here instead of by KerasHub's preprocessor, for two reasons (both seen on examples):
    - UNKNOWN_WORD (keyword-test copies) must become DistilBERT's own [UNK] token. The tokenizer splits our
      placeholder into real word pieces ("xx", "##unk", ...), and without tensorflow-text (no Windows build)
      it does not keep "[UNK]" written in the text as one token either;
    - the uncased English model expects accents stripped, as in the original BERT; KerasHub's preset does not
      strip them, so every Lithuanian word with a diacritic ("pažeminta") became [UNK].
    Our texts have no punctuation, so splitting on spaces gives the same words the tokenizer would find.
    cache: dict word -> word-piece ids, shared between calls (the same words repeat across the sets).
    """
    new = sorted({w for text in texts for w in text.split()} - set(cache) - {UNKNOWN_WORD})
    for word, ids in zip(new, tokenizer([_normalise(tokenizer, w) for w in new])):
        cache[word] = np.asarray(ids).tolist()
    cache[UNKNOWN_WORD] = [tokenizer.token_to_id(tokenizer.oov_token)]
    token_ids = np.full((len(texts), SEQUENCE_LENGTH), tokenizer.pad_token_id, dtype="int32")
    for i, text in enumerate(texts):
        pieces = [p for w in text.split() for p in cache[w]]
        row = [tokenizer.cls_token_id] + pieces[:SEQUENCE_LENGTH - 2] + [tokenizer.sep_token_id]
        token_ids[i, :len(row)] = row
    return {"token_ids": token_ids, "padding_mask": token_ids != tokenizer.pad_token_id}


def _normalise(tokenizer, word):
    """Uncased model: strip accents like the original BERT ("pažeminta" -> "pazeminta"). The multilingual model is
    cased and keeps them (it has "##čiau"); our texts are lowercase anyway."""
    if not tokenizer.lowercase:
        return word
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def done(path, study, model, seed):
    """True if this run is already in results.csv (so a long run is not repeated by accident)."""
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    return ((df["study"] == study) & (df["model"] == model) & (df["seed"] == seed)).any()


def run(seed, study="S14", dev=False):
    model = MODEL_NAMES[study]
    run_id = f"{study}_{model}_base_seed{seed}"
    out = os.path.join(config.RESULTS_DIR, "dev") if dev else config.RESULTS_DIR
    if not dev and done(os.path.join(out, "results.csv"), study, model, seed):
        print(f"jau atlikta: {run_id}")
        return None
    n_classes = 6
    cut = (lambda a: a[:DEV_TEXTS]) if dev else (lambda a: a)
    print(f"\n=== {run_id} ===")

    keras.utils.set_random_seed(seed)                 # head weights, dropout and batch order; as in train.py
    tf.config.experimental.enable_op_determinism()
    path = load_preset(study)
    tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(path)
    cache = {}
    texts = load_emotion()                            # always English: S14b too is trained in English
    x = {k: encode(tokenizer, cut(texts[f"x_{k}"]), cache) for k in ["train", "val", "test"]}
    y = {k: cut(texts[f"y_{k}"]) for k in ["train", "val", "test"]}

    # Pretrained DistilBERT + a new classification head (Dense -> 6 classes, softmax); all weights are fine-tuned
    net = keras_hub.models.DistilBertTextClassifier.from_preset(path, num_classes=n_classes, preprocessor=None,
                                                                activation="softmax")
    net.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True)
    train_ds = (tf.data.Dataset.from_tensor_slices((x["train"], y["train"]))
                .shuffle(len(y["train"]), seed=seed, reshuffle_each_iteration=True).batch(BATCH_SIZE))
    val_ds = tf.data.Dataset.from_tensor_slices((x["val"], y["val"])).batch(BATCH_SIZE)

    start = time.time()
    history = net.fit(train_ds, validation_data=val_ds, epochs=1 if dev else EPOCHS, callbacks=[stop], verbose=2)
    train_time = time.time() - start
    best_epoch = stop.best_epoch + 1                  # EarlyStopping restored the weights of this epoch

    def predict(inputs):
        return net.predict(inputs, batch_size=PREDICT_BATCH, verbose=0)

    def evaluate(k):
        return net.evaluate(x[k], y[k], batch_size=PREDICT_BATCH, verbose=0)   # [loss, accuracy]

    (train_loss, train_acc), (val_loss, val_acc), (test_loss, test_acc) = (evaluate(k) for k in ["train", "val", "test"])
    p_val = predict(x["val"]).argmax(axis=1)
    start = time.time()
    prob_test = predict(x["test"])
    predict_ms_1000 = (time.time() - start) / len(y["test"]) * 1000 * 1000
    p_test = prob_test.argmax(axis=1)
    val_metrics = evaluation.metrics(y["val"], p_val, n_classes)
    test_metrics = evaluation.metrics(y["test"], p_test, n_classes)

    def extra_set(part, lang="en"):
        """Selection set or extended test (CARER texts never trained on)."""
        texts_extra, labels = load_carer_extra(part, lang=lang)
        texts_extra, labels = cut(texts_extra), cut(labels)
        p = predict(encode(tokenizer, texts_extra, cache)).argmax(axis=1)
        return labels, p, evaluation.metrics(labels, p, n_classes)

    _, _, sel_metrics = extra_set("select")
    y_xtest, p_xtest, xtest_metrics = extra_set("xtest")

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "study": study, "model": model, "family": "pretrained transformer", "variant": "base",
        "seed": seed, "machine": platform.node(), "optimizer": "adam", "lr": LEARNING_RATE,
        "params": net.count_params(), "epochs": len(history.history["loss"]), "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1), "predict_ms_1000": round(predict_ms_1000, 1),
        "train_acc": round(train_acc, 4), "train_loss": round(train_loss, 4),
        "val_acc": round(val_acc, 4), "val_loss": round(val_loss, 4), "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "sel_acc": round(sel_metrics["accuracy"], 4), "sel_macro_f1": round(sel_metrics["macro_f1"], 4),
        "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4), "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "xtest_acc": round(xtest_metrics["accuracy"], 4), "xtest_macro_f1": round(xtest_metrics["macro_f1"], 4),
    }
    for name, f1 in zip(config.CLASSES_EN, test_metrics["f1"]):
        row[f"f1_{name}"] = round(f1, 4)

    title = f"{model}, sėkla {seed}"
    evaluation.plot_learning_curves(history.history, best_epoch, title, _path(out, "curves", f"{run_id}.png"))
    evaluation.plot_confusion_matrix(evaluation.confusion_matrix(y["test"], p_test, n_classes), config.CLASSES_LT,
                                     f"{title}, testavimo imtis", _path(out, "matrices", f"{run_id}.png"))

    # The same prediction keys as train.py writes for S1, so the analysis treats DistilBERT like our models
    predictions = {"y_test": y["test"], "p_test": p_test, "prob_test": prob_test,
                   "y_xtest": y_xtest.astype(np.uint8), "p_xtest": p_xtest.astype(np.uint8)}
    x_lt, _, y_lt, lt_index = load_lt_test()
    prob = predict(encode(tokenizer, cut(x_lt), cache))
    predictions.update(y_lt_test=cut(y_lt), p_lt_test=prob.argmax(axis=1), prob_lt_test=prob, lt_test_index=cut(lt_index))
    lt_metrics = evaluation.metrics(cut(y_lt), prob.argmax(axis=1), n_classes)
    print(f"Lietuviška testavimo imtis: bendras tikslumas {lt_metrics['accuracy']:.4f}, "
          f"makro F-matas {lt_metrics['macro_f1']:.4f}")
    if study == "S14":
        # GoEmotions (S6) and the keyword-test copies (S13). Every word can be split into word pieces, so every
        # text with an emotion word counts (no vocabulary check); the analysis compares models on common texts
        x_go, y_go = load_goemotions()
        prob = predict(encode(tokenizer, cut(x_go), cache))
        predictions.update(y_go=cut(y_go), p_go=prob.argmax(axis=1), prob_go=prob)
        kw = keyword_variants(cut(texts["x_test"]))
        for name in ["without", "only", "i_feel", "control"]:
            prob = predict(encode(tokenizer, kw[name], cache))
            predictions.update({f"p_kw_{name}": prob.argmax(axis=1), f"prob_kw_{name}": prob})
        predictions["kw_has"] = kw["has_keyword"]
        # Negation templates (S13): probabilities for the rows of data.negation_templates(train), in that order
        templates = negation_templates(texts["x_train"], texts["y_train"])
        predictions["prob_neg"] = predict(encode(tokenizer, templates["text"].to_numpy(), cache))
    else:
        # S14b: the Lithuanian extended test too, without any Lithuanian training (zero-shot)
        y_lt_x, p_lt_x, lt_x_metrics = extra_set("xtest", lang="lt")
        predictions.update(y_lt_xtest=y_lt_x.astype(np.uint8), p_lt_xtest=p_lt_x.astype(np.uint8))
        print(f"Lietuviška išplėstinė testavimo imtis: bendras tikslumas {lt_x_metrics['accuracy']:.4f}, "
              f"makro F-matas {lt_x_metrics['macro_f1']:.4f}")
    np.savez_compressed(_path(out, "predictions", f"{run_id}.npz"), **predictions)
    _append_row(os.path.join(out, "results.csv"), row)
    if study == "S14" and seed == 1:
        # One saved model is enough for timing.py (S8). Weights only (265 MB; the full .keras file keeps the Adam
        # state too, 804 MB). To load: DistilBertTextClassifier.from_preset(load_preset("S14"), num_classes=6,
        # preprocessor=None, activation="softmax"), then load_weights
        net.save_weights(_path(out, "models", f"{run_id}.weights.h5"))

    print(f"Testavimo bendras tikslumas {test_acc:.4f}, makro F-matas {test_metrics['macro_f1']:.4f}, "
          f"epochų {row['epochs']} (geriausia {best_epoch}), mokymo laikas {train_time / 60:.0f} min")
    keras.backend.clear_session()
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT competitor (S14, S14b).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study", choices=sorted(PRESETS), default="S14")
    parser.add_argument("--dev", action="store_true", help=f"quick check: {DEV_TEXTS} texts, 1 epoch, results/dev/")
    args = parser.parse_args()
    run(args.seed, args.study, args.dev)
