"""B's models: SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention (PLAN.md, section 5).

Same rules as models_a.py: every function returns an UNCOMPILED keras.Model and takes (size, mask, n_classes).
Side-study arguments, as described in models_a.py: GRU takes dropout, l2, layers, embedding_dim,
embedding_matrix and trainable (S10, S11, S12); LSTM takes layers (S11).
"""

# name: (function, input type for data.vectorize, T0 values); filled at step 2.2
MODELS_B = {}
