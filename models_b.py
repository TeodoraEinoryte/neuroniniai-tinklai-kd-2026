"""B's models: SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention (PLAN.md, section 5).

Same rules as models_a.py: every function returns an UNCOMPILED keras.Model; train.py compiles and trains it.
All five take the same arguments:
    size              units of the recurrent layer (per direction for BiLSTM), tuned in T0: 32 / 64 / 128
    mask              Embedding(mask_zero=True): the recurrent layer skips the padding zeros (S2 turns it off)
    n_classes         6 emotions, or 2 for the binary version (S5)
    dropout           Dropout rate before the output layer; default 0 = no Dropout layer (S10)
    l2                L2 penalty on all weights: Embedding (if trained), input and recurrent kernels of the
                      recurrent layers, output layer (S10); not on the attention pooling
    layers            number of stacked recurrent layers (S11)
    embedding_dim     size of learned word vectors (S12 uses 100)
    embedding_matrix  GloVe vectors (one row per vocabulary word x 100) to start the Embedding from, with
    trainable         False = frozen, True = fine-tuned (S12)

Run: python models_b.py   prints each model and checks its parameter count against the lecture 8 formulas.
"""
import keras
from keras import ops, regularizers
from keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Embedding, Input, Layer, SimpleRNN

import config


def build_simplernn(size=64, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
                    embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """Simple recurrent network: h_t = tanh(W x_t + U h_(t-1) + b), the last h_t goes to softmax."""
    return _recurrent_model(SimpleRNN, False, size, mask, n_classes, dropout, l2, layers,
                            embedding_dim, embedding_matrix, trainable)


def build_lstm(size=64, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
               embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """LSTM: cell state C_t with forget, input and output gates (Hochreiter, Schmidhuber, 1997)."""
    return _recurrent_model(LSTM, False, size, mask, n_classes, dropout, l2, layers,
                            embedding_dim, embedding_matrix, trainable)


def build_gru(size=64, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
              embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """GRU: reset and update gates, no separate cell state (Cho et al., 2014)."""
    return _recurrent_model(GRU, False, size, mask, n_classes, dropout, l2, layers,
                            embedding_dim, embedding_matrix, trainable)


def build_bilstm(size=64, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
                 embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """Bidirectional LSTM: one LSTM reads the text left to right, another right to left; their
    last states are joined (2 x size) (Schuster, Paliwal, 1997)."""
    return _recurrent_model(LSTM, True, size, mask, n_classes, dropout, l2, layers,
                            embedding_dim, embedding_matrix, trainable)


def build_bilstm_attention(size=64, mask=True, n_classes=6, dropout=0.0, l2=None, layers=1,
                           embedding_dim=config.EMBEDDING_DIM, embedding_matrix=None, trainable=True):
    """BiLSTM + attention pooling: instead of only the last state, a weighted sum of the states of
    all words, with one weight per word (Yang et al., 2016). The weights are the heatmap of S7.

    Uses the functional API, because the attention layer has two outputs (sum and weights).
    """
    reg = regularizers.l2(l2) if l2 else None
    inputs = Input(shape=(config.SEQUENCE_LENGTH,))
    x = _embedding(mask, embedding_dim, embedding_matrix, trainable, reg)(inputs)
    for _ in range(layers):
        lstm = LSTM(size, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)
        x = Bidirectional(lstm)(x)                                                         # (None, 50, 2*size)
    context, _ = AttentionPooling(name="attention")(x)                                    # (None, 2*size)
    if dropout:
        context = Dropout(dropout)(context)
    outputs = Dense(n_classes, activation="softmax", kernel_regularizer=reg)(context)
    return keras.Model(inputs, outputs, name="bilstm_attention")


@keras.saving.register_keras_serializable(package="kursinis")
class AttentionPooling(Layer):
    """Attention pooling over the word states h_1 ... h_T:
        u_t = tanh(W h_t + b)                  hidden representation of word t
        e_t = u_t · v                          score of word t
        a_t = exp(e_t) / sum_k exp(e_k)        weight of word t (softmax over the words; padding gets 0)
        c   = sum_t a_t h_t                    weighted sum, goes to the output layer
    W, b and v are learned together with the rest of the network. Returns [c, a].
    """

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(d,), initializer="glorot_uniform")

    def call(self, h, mask=None):
        u = ops.tanh(ops.matmul(h, self.W) + self.b)          # (batch, T, d)
        e = ops.sum(u * self.v, axis=-1)                       # (batch, T)
        if mask is not None:
            e = ops.where(mask, e, -1e9)                       # padding positions get weight 0 after softmax
        a = ops.softmax(e, axis=-1)                            # (batch, T), sums to 1 over the words
        c = ops.sum(h * ops.expand_dims(a, -1), axis=1)        # (batch, d)
        return [c, a]

    def compute_mask(self, inputs, mask=None):
        return [None, None]                                    # the outputs no longer have a word axis


def attention_weights(model, x):
    """Word weights a_t of a trained BiLSTM + attention model for encoded texts x: shape (n, SEQUENCE_LENGTH)."""
    weights_model = keras.Model(model.inputs, model.get_layer("attention").output[1])
    return weights_model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)


def _recurrent_model(cell, bidirectional, size, mask, n_classes, dropout, l2, layers,
                     embedding_dim, embedding_matrix, trainable):
    """Embedding -> 1 or more recurrent layers -> (Dropout) -> Dense softmax, as in the lecture 8 example."""
    reg = regularizers.l2(l2) if l2 else None
    model = keras.Sequential([
        Input(shape=(config.SEQUENCE_LENGTH,)),
        _embedding(mask, embedding_dim, embedding_matrix, trainable, reg),
    ])
    for i in range(layers):
        # all but the last layer return the whole sequence, so the next layer gets one vector per word
        rnn = cell(size, return_sequences=i < layers - 1, kernel_regularizer=reg, recurrent_regularizer=reg)
        model.add(Bidirectional(rnn) if bidirectional else rnn)
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation="softmax", kernel_regularizer=reg))
    return model


def _embedding(mask, embedding_dim, embedding_matrix, trainable, reg=None):
    """Learned word vectors, or GloVe vectors (S12) that are frozen or fine-tuned.
    reg (S10) penalises the vectors only when they are trained; frozen ones cannot change."""
    if embedding_matrix is not None:
        # one row per vocabulary word: VOCAB_SIZE, or more with the extended GloVe vocabulary (S12)
        return Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], mask_zero=mask, trainable=trainable,
                         embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                         embeddings_regularizer=reg if trainable else None)
    return Embedding(config.VOCAB_SIZE, embedding_dim, mask_zero=mask, embeddings_regularizer=reg)


# name: (function, input type for data.vectorize, T0 values)
MODELS_B = {
    "simplernn": (build_simplernn, "sequence", [32, 64, 128]),
    "lstm": (build_lstm, "sequence", [32, 64, 128]),
    "gru": (build_gru, "sequence", [32, 64, 128]),
    "bilstm": (build_bilstm, "sequence", [32, 64, 128]),
    "bilstm_attention": (build_bilstm_attention, "sequence", [32, 64, 128]),
}


if __name__ == "__main__":
    # Parameter counts by the lecture 8 formulas (n = input size, u = units):
    # Embedding V*e, SimpleRNN u(n+u+1), LSTM 4u(n+u+1), GRU 3u(n+u+2) (Keras reset_after=True), Dense n*k+k
    V, e, u, k = config.VOCAB_SIZE, config.EMBEDDING_DIM, 64, 6
    emb = V * e
    lstm = 4 * u * (e + u + 1)
    expected = {
        "simplernn": emb + u * (e + u + 1) + (u * k + k),
        "lstm": emb + lstm + (u * k + k),
        "gru": emb + 3 * u * (e + u + 2) + (u * k + k),
        "bilstm": emb + 2 * lstm + (2 * u * k + k),
        # attention: W (2u x 2u) + b (2u) + v (2u)
        "bilstm_attention": emb + 2 * lstm + (2 * u * 2 * u + 2 * u + 2 * u) + (2 * u * k + k),
    }
    for name, (build, _, _) in MODELS_B.items():
        net = build(size=u)
        print(f"{name:17s} parametrų: {net.count_params():>9,} (pagal formulę {expected[name]:>9,})")
        assert net.count_params() == expected[name]
    print("Visi parametrų skaičiai sutampa su formulėmis.")
