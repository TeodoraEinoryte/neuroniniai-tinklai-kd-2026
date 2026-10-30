"""A's models: logreg, MLP, Embedding, Conv1D, Transformer (PLAN.md, section 5).

Every function returns an UNCOMPILED keras.Model; train.py compiles and trains it.
All take the same arguments so train.py can call any model the same way:
    size       the setting tuned in T0 (hidden units, embedding size, filters, heads); unused by logreg
    mask       Embedding(mask_zero=True) for sequence models; unused by multi-hot models
    n_classes  6 emotions, or 2 for the binary version (S5)
"""
import keras
from keras import layers

import config


def build_logreg(size=None, mask=True, n_classes=6):
    """Logistic regression: multi-hot vector -> softmax, no hidden layer (the baseline)."""
    return keras.Sequential([
        keras.Input(shape=(config.VOCAB_SIZE,)),
        layers.Dense(n_classes, activation="softmax"),
    ])


# name: (function, input type for data.vectorize, T0 values; empty = not tuned)
MODELS_A = {
    "logreg": (build_logreg, "multi_hot", []),
}
