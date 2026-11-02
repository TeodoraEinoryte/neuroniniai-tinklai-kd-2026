"""S8, inference time: how long each saved S1 model and DistilBERT (S14) take to classify 1,000 texts.

Run: python timing.py      (~3 min; all models on one machine, so the times can be compared)
For every model, seed 1: load the saved model, encode the first 1,000 test texts beforehand (so only the model is
timed), one warm-up prediction (Keras builds its graph then), then 5 timed ones; batch 64 and 8 CPU threads fixed.
A model whose saved files are missing on this machine is skipped, and the script says which.
Writes results/timing.csv (raw numbers, rewritten on every run); compare.py S8 turns it into the report table and
the quality-vs-cost figure.
"""
import os
import platform
import statistics
import time
from datetime import datetime

import tensorflow as tf

THREADS = 8
tf.config.threading.set_intra_op_parallelism_threads(THREADS)   # must be set before TensorFlow runs anything
tf.config.threading.set_inter_op_parallelism_threads(1)

import keras                                      # noqa: E402 (after the thread settings above)
import keras_hub                                  # noqa: E402
import pandas as pd                               # noqa: E402

import config                                     # noqa: E402
from competitor import encode as encode_word_pieces, load_preset   # noqa: E402
from data import encode, load_emotion             # noqa: E402
from generalization import MODELS_DIR, load_saved  # noqa: E402
from train import MODELS                          # noqa: E402

TEXTS = 1000
BATCH = config.BATCH_SIZE                         # 64, as in training and in competitor.py
REPEATS = 5
SEED = 1                                          # the inference time does not depend on the seed
DISTILBERT_WEIGHTS = os.path.join(MODELS_DIR, f"S14_distilbert_base_seed{SEED}.weights.h5")


def our_model(model, texts):
    """A saved S1 model, its 1,000 encoded texts and its embedding size (the Embedding layer; 0 for the
    multi-hot models, logreg and MLP)."""
    variant = config_variant(model)
    net, vectorizer = load_saved("S1", model, variant, SEED)
    embedding = sum(layer.count_params() for layer in net.layers if isinstance(layer, keras.layers.Embedding))
    return net, encode(vectorizer, texts, MODELS[model][1]), embedding


def config_variant(model):
    """S1 variant name of a model: "size" + its T0 size, "base" for logreg (as train.settings names it)."""
    size = config.MODEL_SIZES.get(model)
    return f"size{size}" if size else "base"


def distilbert(texts):
    """DistilBERT (S14, seed 1): the pretrained architecture with the saved fine-tuned weights, as competitor.py
    describes; texts encoded into word pieces the same way as in training. Embedding = its word-piece embedding."""
    preset = load_preset("S14")
    tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(preset)
    net = keras_hub.models.DistilBertTextClassifier.from_preset(preset, num_classes=len(config.CLASSES_EN),
                                                                preprocessor=None, activation="softmax")
    net.load_weights(DISTILBERT_WEIGHTS)
    return net, encode_word_pieces(tokenizer, texts, {}), net.backbone.token_embedding.count_params()


def time_model(net, x):
    """Milliseconds to predict all 1,000 texts, REPEATS times after one warm-up."""
    net.predict(x, batch_size=BATCH, verbose=0)   # warm-up: the first call also builds the graph
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        net.predict(x, batch_size=BATCH, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
    return times


if __name__ == "__main__":
    texts = load_emotion()["x_test"][:TEXTS]
    rows = []
    for model in [m for m in config.MODEL_NAMES_LT if m in MODELS] + ["distilbert"]:
        if model == "distilbert":
            missing = not os.path.exists(DISTILBERT_WEIGHTS)
        else:
            missing = not os.path.exists(os.path.join(MODELS_DIR, f"S1_{model}_{config_variant(model)}_seed{SEED}.keras"))
        if missing:
            print(f"Praleista (šiame kompiuteryje nėra išsaugoto modelio): {model}")
            continue
        net, x, embedding = distilbert(texts) if model == "distilbert" else our_model(model, texts)
        times = time_model(net, x)
        rows.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "machine": platform.node(), "model": model,
                     "seed": SEED, "params": net.count_params(), "embedding_params": embedding, "texts": TEXTS,
                     "batch": BATCH, "threads": THREADS, "ms_median": round(statistics.median(times), 1),
                     "ms_min": round(min(times), 1), "ms_max": round(max(times), 1)})
        print(f"{model}: {rows[-1]['ms_median']} ms / {TEXTS} tekstų (mediana iš {REPEATS}; "
              f"{rows[-1]['ms_min']}–{rows[-1]['ms_max']}), parametrų {rows[-1]['params']:,}")
        keras.backend.clear_session()
    path = os.path.join(config.RESULTS_DIR, "timing.csv")
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    print("Išsaugota:", path)
