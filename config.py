# Shared settings, the same for every run (PLAN.md, section 4). Change only after both agree.

# Class names by label index: English as in the dataset, Lithuanian for printed output and the report
CLASSES_EN = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CLASSES_LT = ["liūdesys", "džiaugsmas", "meilė", "pyktis", "baimė", "nuostaba"]

# Model names in report tables and figures, in the order of PLAN.md section 5 (+ the DistilBERT competitors)
MODEL_NAMES_LT = {"logreg": "Logistinė regresija", "mlp": "MLP", "embedding": "Embedding", "conv1d": "Conv1D",
                  "transformer": "Transformeris", "simplernn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU",
                  "bilstm": "BiLSTM", "bilstm_attention": "BiLSTM + dėmesys", "distilbert": "DistilBERT",
                  "distilbert_multi": "Daugiakalbis DistilBERT"}

# Binary version (S5): positive = joy, love, surprise (label 1); negative = sadness, anger, fear (label 0)
POSITIVE_CLASSES = [1, 2, 5]
CLASSES_BINARY_LT = ["negatyvi", "pozityvi"]

# Vocabulary size: train has ~15k unique words; the 10k most frequent cover ~98% of all
# word occurrences, and over half of the words appear only once (see vocab_coverage.csv).
VOCAB_SIZE = 10000

# Sequence length: 98.7% of train texts have <= 50 words (longest is 66), so only a few
# texts lose their last words; longer sequences would slow down RNN training (see length_coverage.csv).
SEQUENCE_LENGTH = 50

EMBEDDING_DIM = 64

# Sequence models use the padding mask (Embedding(mask_zero=True)), except these. The Transformer was better
# without it on validation (90.8 % vs 85.9 % and 88.6 % with the mask, seeds 1 and 2; step 2.1); S2 measures it.
NO_MASK_MODELS = ["transformer"]

# Default optimiser and learning rate: T0a runs with them, and a model keeps them unless T0b finds a pair that is
# better by more than TIE_MARGIN on the selection set (PLAN.md, section 4)
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
TIE_MARGIN = 0.005      # macro-F1; the default (middle T0a size, Adam 0.001) stays unless beaten by more
BATCH_SIZE = 64
EPOCHS = 30             # upper limit; EarlyStopping usually stops earlier
PATIENCE = 3            # EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
SEEDS = [1, 2, 3, 4, 5] # for keras.utils.set_random_seed

# T0a edge rule: models whose best size was the largest of the three get one more size (seeds 1-2 again)
T0A_EDGE_SIZES = {"lstm": [256], "gru": [256], "bilstm": [256, 512]}  # 128 best at first; BiLSTM then 256 best

# Size chosen in T0a for each tuned model, e.g. {"lstm": 64}; filled at step 2.4.
# Until then train.py uses the middle T0 value. S1 and later studies refuse to start until it is filled.
MODEL_SIZES = {"mlp": 64, "embedding": 64, "conv1d": 128, "transformer": 128,          # T0a, confirmed 2026-10-31
               "simplernn": 64, "lstm": 128, "gru": 128, "bilstm": 256, "bilstm_attention": 64}

# Optimiser and learning rate chosen in T0b for every model (all 10, logreg too); filled at step 2.4.
# Models without an entry would use OPTIMIZER and LEARNING_RATE; S1 and later studies refuse to start then.
MODEL_OPTIMIZERS = {"logreg": "nesterov", "mlp": "adam", "embedding": "adam", "conv1d": "adam",        # T0b, confirmed
                    "transformer": "adam", "simplernn": "adam", "lstm": "adam", "gru": "adam",          # 2026-10-31
                    "bilstm": "adam", "bilstm_attention": "nesterov"}
MODEL_LEARNING_RATES = {"logreg": 0.1, "mlp": 0.001, "embedding": 0.001, "conv1d": 0.001, "transformer": 0.001,
                        "simplernn": 0.0003, "lstm": 0.01, "gru": 0.01, "bilstm": 0.01, "bilstm_attention": 0.3}

# T0b grid: the best adaptive and the best momentum optimiser of S9, each around its S9 rate (Adam 0.001 is the
# T0a run itself). Edge rule: if a model's best rate is at an end of a list, add one more x 3 step there and rerun T0b.
T0B_GRID = {"adam": [0.0003, 0.003, 0.01], "nesterov": [0.03, 0.1, 0.3]}
T0B_EDGE_RATES = {"simplernn": [("adam", 0.0001)], "lstm": [("adam", 0.03)], "gru": [("adam", 0.03)],
                  "bilstm": [("adam", 0.03)], "bilstm_attention": [("nesterov", 1.0)]}   # best pair at an end at first

# S2: learning rate of the Transformer with the mask, chosen from its own short search (step 3.2): mean selection-set
# macro-F1 of seeds 1-2 at Adam 0.0003 / 0.001 / 0.003 / 0.01 (edge step) = 0.804 / 0.829 / 0.861 / 0.853
S2_MASKED_TRANSFORMER_LR = 0.003

# S9: best learning rate of each optimiser from the S9a grid (highest mean validation macro-F1 of Conv1D and GRU,
# seed 1; step 2.3)
S9_LEARNING_RATES = {"sgd": 1.0, "momentum": 0.1, "nesterov": 0.1, "adagrad": 0.1, "adadelta": 1.0,
                     "rmsprop": 0.01, "adam": 0.001, "nadam": 0.01, "adamw": 0.001}

DATA_DIR = "data"
RESULTS_DIR = "results"
