"""One training run -> one row in results.csv + learning curve + confusion matrix + predictions.

Run: python train.py --model lstm --seed 1 --study S1 [--size 64] [--no-mask] [--padding pre]
                     [--class-weight] [--fraction 0.25] [--task 2] [--optimizer sgd --lr 0.1] [--dev]
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

import config
import evaluation
from data import encode, load_emotion, load_goemotions, load_lt, vectorize
from models_a import MODELS_A
from models_b import MODELS_B

MODELS = {**MODELS_A, **MODELS_B}

# S9: the optimisers from lecture 5 (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam) plus Adagrad and AdamW
OPTIMIZERS = {
    "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr),
    "momentum": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    "nesterov": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True),
    "adagrad": lambda lr: keras.optimizers.Adagrad(learning_rate=lr),
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

# Columns of results.csv (PLAN.md, section 7). Per-class F1 uses the English class names;
# 6-class runs fill f1_sadness ... f1_surprise, binary runs (S5) fill f1_negative and f1_positive.
F1_COLUMNS = [f"f1_{name}" for name in config.CLASSES_EN] + ["f1_negative", "f1_positive"]
COLUMNS = [
    "date", "study", "model", "family", "variant", "seed", "machine", "optimizer", "lr",
    "params", "epochs", "train_time_s", "predict_ms_1000",
    "val_acc", "val_loss", "val_macro_f1",
    "test_acc", "test_loss", "test_macro_f1",
] + F1_COLUMNS

# --dev: a quick check that everything runs; small train set, few epochs, results in results/dev/
DEV_FRACTION = 0.125    # 2,000 train texts
DEV_EPOCHS = 3


def run(model, seed, study, size=None, mask=True, padding="post", class_weight=False,
        fraction=1.0, task="6", optimizer=None, lr=None, dev=False):
    """Train one model once and save everything about the run. Returns the results.csv row as a dict."""
    build, mode, t0_values = MODELS[model]
    size, optimizer, lr, variant = settings(model, size, mask, padding, class_weight, fraction, task, optimizer, lr)
    run_id = f"{study}_{model}_{variant}_seed{seed}"
    out = os.path.join(config.RESULTS_DIR, "dev") if dev else config.RESULTS_DIR
    epochs = DEV_EPOCHS if dev else config.EPOCHS
    if dev:
        fraction = min(fraction, DEV_FRACTION)
    n_classes = 2 if task == "2" else 6
    class_names = config.CLASSES_BINARY_LT if task == "2" else config.CLASSES_LT
    print(f"\n=== {run_id} ===")

    # The same seed gives the same train subset, initial weights and batch order
    keras.utils.set_random_seed(seed)
    d = vectorize(load_emotion(task, fraction, seed), mode, padding)

    net = build(size=size, mask=mask, n_classes=n_classes)
    net.compile(optimizer=OPTIMIZERS[optimizer](lr),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Stop when validation loss has not improved for PATIENCE epochs and go back to the best epoch
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True)

    start = time.time()
    history = net.fit(d["x_train"], d["y_train"], validation_data=(d["x_val"], d["y_val"]),
                      epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=[stop],
                      class_weight=_class_weights(d["y_train"], n_classes) if class_weight else None,
                      verbose=2)
    train_time = time.time() - start

    val_loss, val_acc = net.evaluate(d["x_val"], d["y_val"], batch_size=config.BATCH_SIZE, verbose=0)
    test_loss, test_acc = net.evaluate(d["x_test"], d["y_test"], batch_size=config.BATCH_SIZE, verbose=0)
    p_val = _predict(net, d["x_val"]).argmax(axis=1)
    start = time.time()
    prob_test = _predict(net, d["x_test"])
    predict_ms_1000 = (time.time() - start) / len(d["x_test"]) * 1000 * 1000   # ms per 1,000 texts
    p_test = prob_test.argmax(axis=1)                # predicted class = the one with the highest probability

    val_metrics = evaluation.metrics(d["y_val"], p_val, n_classes)
    test_metrics = evaluation.metrics(d["y_test"], p_test, n_classes)

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "study": study, "model": model, "family": FAMILIES.get(model, ""), "variant": variant,
        "seed": seed, "machine": platform.node(), "optimizer": optimizer, "lr": lr,
        "params": net.count_params(), "epochs": len(history.history["loss"]),
        "train_time_s": round(train_time, 1), "predict_ms_1000": round(predict_ms_1000, 1),
        "val_acc": round(val_acc, 4), "val_loss": round(val_loss, 4), "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4), "test_macro_f1": round(test_metrics["macro_f1"], 4),
    }
    f1_names = ["negative", "positive"] if task == "2" else config.CLASSES_EN
    for name, f1 in zip(f1_names, test_metrics["f1"]):
        row[f"f1_{name}"] = round(f1, 4)

    # Files of this run, all named by run_id
    title = f"{model} ({variant}), sėkla {seed}"
    evaluation.plot_learning_curves(history.history, title, _path(out, "curves", f"{run_id}.png"))
    evaluation.plot_confusion_matrix(evaluation.confusion_matrix(d["y_test"], p_test, n_classes), class_names,
                                     f"{title}, testavimo imtis", _path(out, "matrices", f"{run_id}.png"))

    predictions = {"y_test": d["y_test"], "p_test": p_test, "prob_test": prob_test}
    if study == "S1":
        # S6 needs predictions for GoEmotions and the LT/EN sentences; the texts are encoded with
        # the vocabulary of this run (built from train only)
        x_go, y_go = load_goemotions(task)
        x_lt, x_en, y_lt = load_lt(task)
        for name, texts, y in [("go", x_go, y_go), ("lt", x_lt, y_lt), ("en", x_en, y_lt)]:
            prob = _predict(net, encode(d["vectorizer"], texts, mode, padding))
            predictions.update({f"y_{name}": y, f"p_{name}": prob.argmax(axis=1), f"prob_{name}": prob})
        net.save(_path(out, "models", f"{run_id}.keras"))
    np.savez_compressed(_path(out, "predictions", f"{run_id}.npz"), **predictions)

    _append_row(os.path.join(out, "results.csv"), row)
    print(f"Testavimo bendras tikslumas {test_acc:.4f}, makro F-matas {test_metrics['macro_f1']:.4f}, "
          f"epochų {row['epochs']}, mokymo laikas {train_time:.0f} s")
    keras.backend.clear_session()                    # free memory before the next run (run_study.py)
    return row


def settings(model, size=None, mask=True, padding="post", class_weight=False, fraction=1.0, task="6",
             optimizer=None, lr=None):
    """Fill in the defaults from config.py; returns (size, optimizer, lr, variant name).
    run_study.py uses it too, to find the runs that are already in results.csv."""
    size = _pick_size(model, size, MODELS[model][2])
    optimizer = optimizer or config.OPTIMIZER
    lr = lr or config.LEARNING_RATE
    return size, optimizer, lr, _variant(size, mask, padding, class_weight, fraction, task, optimizer, lr)


def _pick_size(model, size, t0_values):
    """Size given on the command line, else the one chosen in T0 (config.MODEL_SIZES), else the middle T0 value."""
    if size is not None:
        return size
    if model in config.MODEL_SIZES:
        return config.MODEL_SIZES[model]
    return t0_values[len(t0_values) // 2] if t0_values else None


def _variant(size, mask, padding, class_weight, fraction, task, optimizer, lr):
    """Short name for what this run changes, e.g. "size64", "size64-nomask-pre" or "size64-sgd-lr0.1";
    also used in file names."""
    parts = [f"size{size}" if size is not None else "base"]
    if not mask:
        parts.append("nomask")
    if padding == "pre":
        parts.append("pre")
    if class_weight:
        parts.append("cw")
    if fraction < 1.0:
        parts.append(f"frac{fraction:g}")
    if task == "2":
        parts.append("task2")
    if optimizer != config.OPTIMIZER or lr != config.LEARNING_RATE:
        parts.append(f"{optimizer}-lr{lr:g}")
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
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="One training run (PLAN.md, section 7).")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study", default="dev", help="T0, S1, S2, ...; goes into results.csv and file names")
    parser.add_argument("--size", type=int, help="tuned setting; default: config.MODEL_SIZES or the middle T0 value")
    parser.add_argument("--no-mask", action="store_true", help="Embedding without mask_zero (S2)")
    parser.add_argument("--padding", choices=["post", "pre"], default="post", help="zeros after or before the text (S2)")
    parser.add_argument("--class-weight", action="store_true", help="inverse class frequency weights (S3)")
    parser.add_argument("--fraction", type=float, default=1.0, help="stratified share of train (S4)")
    parser.add_argument("--task", choices=["6", "2"], default="6", help="6 emotions or positive/negative (S5)")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), help="default: config.OPTIMIZER (S9)")
    parser.add_argument("--lr", type=float, help="learning rate; default: config.LEARNING_RATE (S9)")
    parser.add_argument("--dev", action="store_true", help="quick check: 2,000 texts, 3 epochs, results/dev/")
    args = parser.parse_args()
    run(args.model, args.seed, args.study, args.size, not args.no_mask, args.padding,
        args.class_weight, args.fraction, args.task, args.optimizer, args.lr, args.dev)


if __name__ == "__main__":
    main()
