"""A's models: logreg, MLP, Embedding, Conv1D, Transformer.

Every function returns an UNCOMPILED keras.Model; train.py compiles and trains it.
All take the same arguments so train.py can call any model the same way:
    size       the setting tuned in T0 (hidden units, embedding size, filters, feed-forward units); unused by logreg
    mask       Embedding(mask_zero=True) for sequence models; unused by multi-hot models and by Conv1D
               (a convolution looks at fixed windows, so it has no use for the padding mask)
    n_classes  6 emotions, or 2 for the binary version (S5)
Conv1D, a side model, also takes (train.py passes them only when a study changes them):
    dropout           Dropout rate before the output layer; default 0, no Dropout layer (S10)
    l2                L2 penalty on all weights: Embedding (if trained), Conv1D and output layers (S10)
    layers            number of stacked Conv1D layers; default 1 (S11)
    embedding_dim     size of learned word vectors; default config.EMBEDDING_DIM (S12 uses 100)
    embedding_matrix  GloVe vectors (one row per vocabulary word x 100) to start the Embedding from, with
    trainable         False = frozen, True = fine-tuned (S12)

Run: python models_a.py   prints each model and checks its parameter count against the formulas.
"""
import keras
from keras import ops, regularizers
from keras.layers import (Conv1D, Dense, Dropout, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, Input,
                          Layer, LayerNormalization, MultiHeadAttention)

import config

KERNEL_SIZE = 5           # Conv1D window: 5 words (Kim, 2014 uses 3-5)
HEADS = 2                 # Transformer attention heads (fixed: with key_dim = 64 / heads they do not change capacity)


def build_logreg(size=None, mask=True, n_classes=6):
    """Logistic regression: multi-hot vector -> softmax, no hidden layer (the baseline)."""
    return keras.Sequential([
        Input(shape=(config.VOCAB_SIZE,)),
        Dense(n_classes, activation="softmax"),
    ])


def build_mlp(size=64, mask=True, n_classes=6):
    """Multilayer perceptron on the multi-hot vector: one hidden ReLU layer of `size` units and Dropout 0.5."""
    return keras.Sequential([
        Input(shape=(config.VOCAB_SIZE,)),
        Dense(size, activation="relu"),
        Dropout(0.5),
        Dense(n_classes, activation="softmax"),
    ])


def build_embedding(size=64, mask=True, n_classes=6):
    """Word vectors of `size` numbers, averaged over the words of the text (padding left out when masked),
    then one hidden ReLU layer: the word order is ignored."""
    return keras.Sequential([
        Input(shape=(config.SEQUENCE_LENGTH,)),
        Embedding(config.VOCAB_SIZE, size, mask_zero=mask),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])


def build_conv1d(size=128, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
                 embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """Convolutional network: `size` filters slide over windows of KERNEL_SIZE words, and the strongest
    response of each filter anywhere in the text is kept (GlobalMaxPooling): a learned keyword detector."""
    reg = regularizers.l2(l2) if l2 else None
    if embedding_matrix is not None:
        # one row per vocabulary word: VOCAB_SIZE, or more with the extended GloVe vocabulary (S12)
        embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], trainable=trainable,
                              embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                              embeddings_regularizer=reg if trainable else None)   # frozen vectors cannot change
    else:
        embedding = Embedding(config.VOCAB_SIZE, embedding_dim, embeddings_regularizer=reg)
    model = keras.Sequential([Input(shape=(config.SEQUENCE_LENGTH,)), embedding])
    for _ in range(layers):
        model.add(Conv1D(size, KERNEL_SIZE, activation="relu", kernel_regularizer=reg))
    model.add(GlobalMaxPooling1D())
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation="softmax", kernel_regularizer=reg))
    return model


def build_transformer(size=128, mask=True, n_classes=6):
    """Transformer encoder block (Vaswani et al., 2017) with a feed-forward layer of `size` units:
    word + position vectors -> self-attention with HEADS heads -> add & LayerNorm -> feed-forward -> add & LayerNorm
    -> average over the words -> softmax. Its own Dropout (0.1, 0.1, 0.3) is part of the block, as in the Keras example.
    mask (the protocol uses False, config.NO_MASK_MODELS, which was better on validation; S2 measures the others):
        True         attention and the average use only the real words, not the padding
        "attention"  only the attention ignores the padding;  "pooling"  only the average ignores it
    """
    d = config.EMBEDDING_DIM
    inputs = Input(shape=(config.SEQUENCE_LENGTH,))
    x = Embedding(config.VOCAB_SIZE, d)(inputs)
    x = PositionEmbedding(name="positions")(x)                        # adds a learned vector for each position
    padding = ops.not_equal(inputs, 0)                                 # True for real words
    attention_mask = ops.expand_dims(padding, 1) if mask in (True, "attention") else None   # attend to real words
    attention = MultiHeadAttention(num_heads=HEADS, key_dim=d // HEADS)(x, x, attention_mask=attention_mask)
    x = LayerNormalization()(x + Dropout(0.1)(attention))
    ffn = Dense(d)(Dense(size, activation="relu")(x))
    x = LayerNormalization()(x + Dropout(0.1)(ffn))
    x = GlobalAveragePooling1D()(x, mask=padding if mask in (True, "pooling") else None)   # average of real words
    outputs = Dense(n_classes, activation="softmax")(Dropout(0.3)(x))
    return keras.Model(inputs, outputs, name="transformer")


@keras.saving.register_keras_serializable(package="kursinis")
class PositionEmbedding(Layer):
    """Adds a learned vector for each position 0 ... SEQUENCE_LENGTH-1 to the word vectors,
    so the Transformer knows the word order (attention itself does not)."""

    def build(self, input_shape):
        self.positions = self.add_weight(name="positions", shape=(input_shape[1], input_shape[2]),
                                         initializer="uniform")

    def call(self, x):
        return x + self.positions


# name: (function, input type for data.vectorize, T0 values; empty = not tuned)
MODELS_A = {
    "logreg": (build_logreg, "multi_hot", []),
    "mlp": (build_mlp, "multi_hot", [32, 64, 128]),
    "embedding": (build_embedding, "sequence", [32, 64, 128]),
    "conv1d": (build_conv1d, "sequence", [64, 128, 256]),
    "transformer": (build_transformer, "sequence", [64, 128, 256]),
}


if __name__ == "__main__":
    # Parameter counts by hand (V = vocabulary, e = embedding size, k = classes):
    # Dense n*u + u; Embedding V*e; Conv1D kernel*e*filters + filters;
    # MultiHeadAttention 4 * (e*e + e) when heads * key_dim = e; LayerNorm 2e
    V, e, k = config.VOCAB_SIZE, config.EMBEDDING_DIM, 6
    T = config.SEQUENCE_LENGTH
    expected = {
        "logreg": V * k + k,
        "mlp": (V * 64 + 64) + (64 * k + k),
        "embedding": V * 64 + (64 * 64 + 64) + (64 * k + k),
        "conv1d": V * e + (KERNEL_SIZE * e * 128 + 128) + (128 * k + k),
        "transformer": V * e + T * e + 4 * (e * e + e) + 2 * (2 * e)
                       + (e * 128 + 128) + (128 * e + e) + (e * k + k),
    }
    for name, (build, _, values) in MODELS_A.items():
        sizes = values or [None]
        for size in sizes:
            net = build(size=size) if size is not None else build()
            note = f"(pagal formulę {expected[name]:>9,})" if size == (values[1] if values else None) else ""
            print(f"{name:12s} dydis {str(size):>4s} parametrų: {net.count_params():>9,} {note}")
            if note:
                assert net.count_params() == expected[name]
    print("Parametrų skaičiai sutampa su formulėmis.")
