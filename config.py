# Shared settings, the same for every run (PLAN.md, section 4). Change only after both agree.

# Class names by label index: English as in the dataset, Lithuanian for printed output and the report
CLASSES_EN = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CLASSES_LT = ["liūdesys", "džiaugsmas", "meilė", "pyktis", "baimė", "nuostaba"]

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
LEARNING_RATE = 0.001   # Adam
BATCH_SIZE = 64
EPOCHS = 30             # upper limit; EarlyStopping usually stops earlier
PATIENCE = 3            # EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
SEEDS = [1, 2, 3, 4, 5] # for keras.utils.set_random_seed

DATA_DIR = "data"
RESULTS_DIR = "results"
