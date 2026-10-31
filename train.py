"""One training run -> one row in results.csv + learning curve + confusion matrix + predictions.

Run: python train.py --model lstm --seed 1 --study S1 [--size 64] [--no-mask | --mask] [--padding pre]
                     [--class-weight] [--loss focal] [--fraction 0.25] [--task 2] [--optimizer sgd --lr 0.1]
                     [--dropout 0.5] [--l2 0.0001] [--layers 2] [--embedding glove-frozen] [--lang lt]
                     [--without-keyword] [--corpus goemotions] [--dev]
Everything not given on the command line comes from config.py (PLAN.md, section 4).
"""
import argparse
import csv
import os
import platform
import time
from datetime import datetime

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import config
import evaluation
from data import (encode, keyword_variants, load_carer_extra, load_emotion, load_goemotions, load_goemotions_splits,
                  load_lt_test, vectorize)
from models_a import MODELS_A
from models_b import MODELS_B

MODELS = {**MODELS_A, **MODELS_B}

# S9: the optimisers from lecture 4 (SGD, Momentum, Nesterov, Adagrad, Adadelta, RMSprop, Adam, Nadam) plus AdamW
OPTIMIZERS = {
    "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr),
    "momentum": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    "nesterov": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
    "adagrad": lambda lr: keras.optimizers.Adagrad(learning_rate=lr),
    "adadelta": lambda lr: keras.optimizers.Adadelta(learning_rate=lr),
    "rmsprop": lambda lr: keras.optimizers.RMSprop(learning_rate=lr),
    "adam": lambda lr: keras.optimizers.Adam(learning_rate=lr),
    "nadam": lambda lr: keras.optimizers.Nadam(learning_rate=lr),
    "adamw": lambda lr: keras.optimizers.AdamW(learning_rate=lr),
}

FAMILIES = {
    "logreg": "feedforward", "mlp": "feedforward", "embedding": "feedforward",
    "conv1d": "convolutional", "transformer": "attention",
    "simplernn": "recurrent", "lstm": "recurrent", "gru": "recurrent", "bilstm": "recurrent",
    "bilstm_attention": "recurrent + attention",
}

# What a study may change; everything else stays as in config.py. None = the protocol value.
DEFAULTS = {
    "size": None,             # T0: hidden units / filters / feed-forward units; None = config.MODEL_SIZES or the middle T0 value
    "mask": None,             # S2: Embedding(mask_zero=True); None = True, except config.NO_MASK_MODELS;
                              #   the Transformer also takes "attention" or "pooling" (the mask in one place only)
    "padding": "post",        # S2: zeros after ("post") or before ("pre") the text
    "class_weight": False,    # S3: inverse class frequency weights
    "loss": "crossentropy",   # S3: "focal" = focal loss
    "fraction": 1.0,          # S4: stratified share of train
    "task": "6",              # S5: "2" = positive/negative
    "optimizer": None,        # S9, T0: name from OPTIMIZERS; None = the model's choice from T0b
                              #   (config.MODEL_OPTIMIZERS), or config.OPTIMIZER before T0b
    "lr": None,               # S9, T0: learning rate; None = the model's rate from T0b (config.MODEL_LEARNING_RATES),
                              #   or config.LEARNING_RATE before T0b
    "dropout": None,          # S10: Dropout rate before the output layer; None = the model's own default
    "l2": None,               # S10: L2 penalty on the weights
    "layers": None,           # S11: number of Conv1D / recurrent layers; None = 1
    "embedding": "learned",   # S12: "learned100", "glove-frozen", "glove-tuned" or "glove-extended" (all 100-d;
                              #   extended = frozen GloVe, vocabulary + the 40,000 most frequent GloVe words)
    "lang": "en",             # S15: "lt" = train, validate and test on the Lithuanian translation
    "without_keyword": False, # S16: the emotion word replaced by data.UNKNOWN_WORD in train and validation texts
    "corpus": "carer",        # S17: "goemotions" = train on GoEmotions (its own split), test on it and on CARER
}
GLOVE_DIM = 100
NO_XTEST_STUDIES = ["dev", "S9a", "S9", "T0a", "T0b", "S4"]   # no extended-test predictions (choices, or a curve only)

# Columns of results.csv (PLAN.md, section 7). Per-class F1 uses the English class names;
# 6-class runs fill f1_sadness ... f1_surprise, binary runs (S5) fill f1_negative and f1_positive.
F1_COLUMNS = [f"f1_{name}" for name in config.CLASSES_EN] + ["f1_negative", "f1_positive"]
COLUMNS = [
    "date", "study", "model", "family", "variant", "seed", "machine", "optimizer", "lr",
    "params", "epochs", "best_epoch", "train_time_s", "predict_ms_1000",
    "train_acc", "train_loss", "val_acc", "val_loss", "val_macro_f1", "sel_acc", "sel_macro_f1",
    "test_acc", "test_loss", "test_macro_f1", "xtest_acc", "xtest_macro_f1",
] + F1_COLUMNS

# --dev: a quick check that everything runs; small train set, few epochs, results in results/dev/
DEV_FRACTION = 0.125    # 2,000 train texts
DEV_EPOCHS = 3


def run(model, seed, study, dev=False, **options):
    """Train one model once and save everything about the run. Returns the results.csv row as a dict.

    options: any key of DEFAULTS, e.g. run("gru", 1, "S10", dropout=0.5).
    """
    build, mode, t0_values = MODELS[model]
    o, variant = settings(model, **options)
    run_id = f"{study}_{model}_{variant}_seed{seed}"
    out = os.path.join(config.RESULTS_DIR, "dev") if dev else config.RESULTS_DIR
    epochs = DEV_EPOCHS if dev else config.EPOCHS
    fraction = min(o["fraction"], DEV_FRACTION) if dev else o["fraction"]
    n_classes = 2 if o["task"] == "2" else 6
    class_names = config.CLASSES_BINARY_LT if o["task"] == "2" else config.CLASSES_LT
    print(f"\n=== {run_id} ===")

    # The same seed gives the same train subset, initial weights and batch order; op determinism makes every
    # optimiser repeat bit for bit (without it SGD, Adagrad, Adadelta and RMSprop did not: sparse embedding updates)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    if o["corpus"] == "goemotions":
        texts = load_goemotions_splits(o["task"])                      # S17
    else:
        texts = load_emotion(o["task"], fraction, seed, o["lang"])
    if o["without_keyword"]:
        # S16: the emotion word becomes the placeholder everywhere it occurs (as in the test copy). It is replaced
        # before the vocabulary is built, so the placeholder is a word of its own there (3rd most frequent), not
        # [UNK]: the model knows a word stood there, not which, and [UNK] (rare words, leans to joy) stays apart
        texts = dict(texts, x_train=keyword_variants(texts["x_train"])["without"],
                     x_val=keyword_variants(texts["x_val"])["without"])
    # S12 "glove-extended": the most frequent GloVe words join the vocabulary, so words never seen in train
    # (e.g. in GoEmotions) get a GloVe vector instead of [UNK]
    extra_words = None
    if o["embedding"] == "glove-extended":
        from data import glove_words                 # needs data/glove/ (download_glove.py)
        extra_words = glove_words()
    d = vectorize(texts, mode, o["padding"], extra_words)

    net = build(size=o["size"], mask=o["mask"], n_classes=n_classes, **_model_extras(o, d["vectorizer"]))
    # Focal loss needs one-hot labels; cross-entropy works with the class numbers directly.
    # alpha=1: Keras multiplies by alpha = 0.25 by default, which is not in the formula (PLAN.md, section 6)
    if o["loss"] == "focal":
        loss = keras.losses.CategoricalFocalCrossentropy(alpha=1.0, gamma=2.0)
        y = {k: keras.utils.to_categorical(d[k], n_classes) for k in ["y_train", "y_val", "y_test"]}
    else:
        loss = "sparse_categorical_crossentropy"
        y = {k: d[k] for k in ["y_train", "y_val", "y_test"]}
    net.compile(optimizer=OPTIMIZERS[o["optimizer"]](o["lr"]), loss=loss, metrics=["accuracy"])
    # Stop when validation loss has not improved for PATIENCE epochs and go back to the best epoch
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True)

    start = time.time()
    history = net.fit(d["x_train"], y["y_train"], validation_data=(d["x_val"], y["y_val"]),
                      epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=[stop],
                      class_weight=_class_weights(d["y_train"], n_classes) if o["class_weight"] else None,
                      verbose=2)
    train_time = time.time() - start
    # The epoch EarlyStopping went back to (lowest val_loss). Taken from Keras, not from the history: when a large
    # learning rate makes the loss NaN (S9a), np.argmin would point at the first NaN epoch instead
    best_epoch = stop.best_epoch + 1

    # All three sets with the restored (best) weights; train vs validation shows overfitting
    train_loss, train_acc = net.evaluate(d["x_train"], y["y_train"], batch_size=config.BATCH_SIZE, verbose=0)
    val_loss, val_acc = net.evaluate(d["x_val"], y["y_val"], batch_size=config.BATCH_SIZE, verbose=0)
    test_loss, test_acc = net.evaluate(d["x_test"], y["y_test"], batch_size=config.BATCH_SIZE, verbose=0)
    p_val = _predict(net, d["x_val"]).argmax(axis=1)
    start = time.time()
    prob_test = _predict(net, d["x_test"])
    predict_ms_1000 = (time.time() - start) / len(d["x_test"]) * 1000 * 1000   # ms per 1,000 texts
    p_test = prob_test.argmax(axis=1)                # predicted class = the one with the highest probability

    val_metrics = evaluation.metrics(d["y_val"], p_val, n_classes)
    test_metrics = evaluation.metrics(d["y_test"], p_test, n_classes)

    # Selection set (every run: all choices use it) and extended test (confidence intervals), both CARER texts
    # never trained on, in the language of the run; --dev uses their first 2,000 texts
    def extra_set(part):
        x, labels = load_carer_extra(part, o["task"], o["lang"])
        if dev:
            x, labels = x[:2000], labels[:2000]
        p = _predict(net, encode(d["vectorizer"], x, mode, o["padding"])).argmax(axis=1)
        return labels, p, evaluation.metrics(labels, p, n_classes)

    _, _, sel_metrics = extra_set("select")
    if study not in NO_XTEST_STUDIES:
        y_xtest, p_xtest, xtest_metrics = extra_set("xtest")

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "study": study, "model": model, "family": FAMILIES.get(model, ""), "variant": variant,
        "seed": seed, "machine": platform.node(), "optimizer": o["optimizer"], "lr": o["lr"],
        "params": net.count_params(), "epochs": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1), "predict_ms_1000": round(predict_ms_1000, 1),
        "train_acc": round(train_acc, 4), "train_loss": round(train_loss, 4),
        "val_acc": round(val_acc, 4), "val_loss": round(val_loss, 4), "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "sel_acc": round(sel_metrics["accuracy"], 4), "sel_macro_f1": round(sel_metrics["macro_f1"], 4),
        "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4), "test_macro_f1": round(test_metrics["macro_f1"], 4),
    }
    if study not in NO_XTEST_STUDIES:
        row.update(xtest_acc=round(xtest_metrics["accuracy"], 4), xtest_macro_f1=round(xtest_metrics["macro_f1"], 4))
    f1_names = ["negative", "positive"] if o["task"] == "2" else config.CLASSES_EN
    for name, f1 in zip(f1_names, test_metrics["f1"]):
        row[f"f1_{name}"] = round(f1, 4)

    # Files of this run, all named by run_id
    title = f"{model} ({variant}), sėkla {seed}"
    evaluation.plot_learning_curves(history.history, best_epoch, title, _path(out, "curves", f"{run_id}.png"))
    evaluation.plot_confusion_matrix(evaluation.confusion_matrix(d["y_test"], p_test, n_classes), class_names,
                                     f"{title}, testavimo imtis", _path(out, "matrices", f"{run_id}.png"))

    predictions = {"y_test": d["y_test"], "p_test": p_test, "prob_test": prob_test}
    if study not in NO_XTEST_STUDIES:
        predictions.update(y_xtest=y_xtest.astype(np.uint8), p_xtest=p_xtest.astype(np.uint8))   # 50 kB, not 2 MB
    if study in ("S1", "S12"):
        # S6 and S12 need predictions for GoEmotions and the Lithuanian test set; the texts are encoded
        # with the vocabulary of this run (built from train only)
        x_go, y_go = load_goemotions(o["task"])
        x_lt_test, _, y_lt_test, lt_test_index = load_lt_test(o["task"])   # English side = the test set itself
        for name, other, labels in [("go", x_go, y_go), ("lt_test", x_lt_test, y_lt_test)]:
            prob = _predict(net, encode(d["vectorizer"], other, mode, o["padding"]))
            predictions.update({f"y_{name}": labels, f"p_{name}": prob.argmax(axis=1), f"prob_{name}": prob})
        predictions["lt_test_index"] = lt_test_index   # row in the test set, to pair LT and EN predictions
    if study in ("S1", "S12", "S16"):
        # S13 keyword test: the same test texts without the emotion word, with only it, as "i feel <word>",
        # and the control (another word unknown); a text counts only if its emotion word is in this run's vocabulary.
        # S16 replaces every emotion word, as in its training texts: its vocabulary lacks 54 test emotion words,
        # which would otherwise stay in the text and reach the model as [UNK] instead of the placeholder
        vocabulary = None if o["without_keyword"] else set(d["vectorizer"].get_vocabulary())
        kw = keyword_variants(texts["x_test"], vocabulary=vocabulary)
        for name in ["without", "only", "i_feel", "control"]:
            prob = _predict(net, encode(d["vectorizer"], kw[name], mode, o["padding"]))
            predictions.update({f"p_kw_{name}": prob.argmax(axis=1), f"prob_kw_{name}": prob})
        predictions["kw_has"] = kw["has_keyword"]       # metrics of the copies use only these texts
    if o["corpus"] == "goemotions":
        # S17: the model trained on GoEmotions, tested on the CARER test set too
        carer = load_emotion(o["task"])
        prob = _predict(net, encode(d["vectorizer"], carer["x_test"], mode, o["padding"]))
        predictions.update(y_carer=carer["y_test"], p_carer=prob.argmax(axis=1), prob_carer=prob)
    if study in ("S1", "S15"):
        # the model and its vocabulary (one word per line, line i = index i), for the app and later analysis
        net.save(_path(out, "models", f"{run_id}.keras"))
        with open(_path(out, "models", f"{run_id}_vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(d["vectorizer"].get_vocabulary()))
    np.savez_compressed(_path(out, "predictions", f"{run_id}.npz"), **predictions)

    _append_row(os.path.join(out, "results.csv"), row)
    print(f"Testavimo bendras tikslumas {test_acc:.4f}, makro F-matas {test_metrics['macro_f1']:.4f}, "
          f"epochų {row['epochs']} (geriausia {row['best_epoch']}), mokymo laikas {train_time:.0f} s")
    keras.backend.clear_session()                    # free memory before the next run (run_study.py)
    return row


def settings(model, **options):
    """All options of a run with the defaults filled in, and the variant name.

    Returns (options dict, variant). run_study.py uses it too, to find the runs already in results.csv.
    """
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown options: {unknown}")
    o = {**DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
    # S9 and T0 set the optimiser or learning rate themselves; that shows in the variant name
    optimizer_given = o["optimizer"] is not None or o["lr"] is not None
    default_mask = model not in config.NO_MASK_MODELS
    o["mask"] = default_mask if o["mask"] is None else o["mask"]
    o["size"] = _pick_size(model, o["size"], MODELS[model][2])
    o["optimizer"] = o["optimizer"] or config.MODEL_OPTIMIZERS.get(model, config.OPTIMIZER)
    o["lr"] = o["lr"] or config.MODEL_LEARNING_RATES.get(model, config.LEARNING_RATE)
    return o, _variant(o, optimizer_given, default_mask)


def _model_extras(o, vectorizer):
    """Options passed on to the model function; only the ones a study changes, so every model
    function needs to accept only what its studies use (PLAN.md, section 7)."""
    extras = {k: o[k] for k in ["dropout", "l2", "layers"] if o[k] is not None}
    if o["embedding"] == "learned100":
        extras["embedding_dim"] = GLOVE_DIM          # learned vectors of the same size as GloVe
    elif o["embedding"] in ("glove-frozen", "glove-tuned", "glove-extended"):
        from data import glove_matrix                # step 3.5 (S12); downloads nothing by itself
        extras["embedding_matrix"] = glove_matrix(vectorizer, GLOVE_DIM)   # one row per vocabulary word
        extras["trainable"] = o["embedding"] == "glove-tuned"
    return extras


def _pick_size(model, size, t0_values):
    """Size given on the command line, else the one chosen in T0 (config.MODEL_SIZES), else the middle T0 value."""
    if size is not None:
        return size
    if model in config.MODEL_SIZES:
        return config.MODEL_SIZES[model]
    return t0_values[len(t0_values) // 2] if t0_values else None


def _variant(o, optimizer_given, default_mask):
    """Short name for what this run changes, e.g. "size64", "size64-nomask-pre" or "size64-sgd-lr0.1";
    also used in file names. The mask appears only when it differs from the model's default ("nomask", or
    "mask" / "attentionmask" / "poolingmask" for the Transformer); the optimiser and learning rate only when the
    run set them (S9, T0); the ones actually used are always in the optimizer and lr columns of results.csv."""
    parts = [f"size{o['size']}" if o["size"] is not None else "base"]
    if o["mask"] != default_mask:
        parts.append(f"{o['mask']}mask" if isinstance(o["mask"], str) else "mask" if o["mask"] else "nomask")
    if o["padding"] == "pre":
        parts.append("pre")
    if o["class_weight"]:
        parts.append("cw")
    if o["loss"] == "focal":
        parts.append("focal")
    if o["fraction"] < 1.0:
        parts.append(f"frac{o['fraction']:g}")
    if o["task"] == "2":
        parts.append("task2")
    if optimizer_given:
        parts.append(f"{o['optimizer']}-lr{o['lr']:g}")
    if o["dropout"] is not None:
        parts.append(f"do{o['dropout']:g}")
    if o["l2"] is not None:
        parts.append(f"l2{o['l2']:g}")
    if o["layers"] is not None:
        parts.append(f"layers{o['layers']}")
    if o["embedding"] != "learned":
        parts.append(o["embedding"])
    if o["lang"] != "en":
        parts.append(o["lang"])
    if o["without_keyword"]:
        parts.append("nokw")
    if o["corpus"] != "carer":
        parts.append(o["corpus"])
    return "-".join(parts)


def _class_weights(y, n_classes):
    """Inverse class frequency, w_c = n / (k * n_c): rare classes (surprise 3.6 %) weigh more in the loss (S3)."""
    counts = np.bincount(y, minlength=n_classes)
    return {c: len(y) / (n_classes * counts[c]) for c in range(n_classes)}


def _predict(net, x):
    return net.predict(x, batch_size=config.BATCH_SIZE, verbose=0)


def _path(out, folder, filename):
    os.makedirs(os.path.join(out, folder), exist_ok=True)
    return os.path.join(out, folder, filename)


def _append_row(path, row):
    new_file = not os.path.exists(path)
    if not new_file:
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != COLUMNS:
            # COLUMNS gained new columns (e.g. sel_*, xtest_* in review 2): rewrite the file once with the
            # new header, older rows get empty cells there
            pd.read_csv(path).reindex(columns=COLUMNS).to_csv(path, index=False)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="One training run (PLAN.md, section 7).")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study", default="dev", help="S9a, S9, T0a, T0b, S1, ...; goes into results.csv and file names")
    parser.add_argument("--size", type=int, help="tuned setting; default: config.MODEL_SIZES or the middle T0 value")
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction,
                        help="--no-mask: Embedding without mask_zero; --mask: Transformer with the mask (S2)")
    parser.add_argument("--padding", choices=["post", "pre"], help="zeros after or before the text (S2)")
    parser.add_argument("--class-weight", action="store_true", help="inverse class frequency weights (S3)")
    parser.add_argument("--loss", choices=["crossentropy", "focal"], help="focal loss (S3)")
    parser.add_argument("--fraction", type=float, help="stratified share of train (S4)")
    parser.add_argument("--task", choices=["6", "2"], help="6 emotions or positive/negative (S5)")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), help="default: config.OPTIMIZER (S9)")
    parser.add_argument("--lr", type=float, help="learning rate; default: config.LEARNING_RATE (S9)")
    parser.add_argument("--dropout", type=float, help="Dropout rate before the output layer (S10)")
    parser.add_argument("--l2", type=float, help="L2 penalty on the weights (S10)")
    parser.add_argument("--layers", type=int, help="number of Conv1D / recurrent layers (S11)")
    parser.add_argument("--embedding", choices=["learned", "learned100", "glove-frozen", "glove-tuned", "glove-extended"],
                        help="S12")
    parser.add_argument("--lang", choices=["en", "lt"], help="lt: the Lithuanian translation (S15)")
    parser.add_argument("--without-keyword", action="store_true", help="train without the emotion word (S16)")
    parser.add_argument("--corpus", choices=["carer", "goemotions"], help="goemotions: train on GoEmotions (S17)")
    parser.add_argument("--dev", action="store_true", help="quick check: 2,000 texts, 3 epochs, results/dev/")
    args = vars(parser.parse_args())
    model, seed, study, dev = args.pop("model"), args.pop("seed"), args.pop("study"), args.pop("dev")
    args["class_weight"] = args.pop("class_weight") or None
    args["without_keyword"] = args.pop("without_keyword") or None
    run(model, seed, study, dev, **args)


if __name__ == "__main__":
    main()
