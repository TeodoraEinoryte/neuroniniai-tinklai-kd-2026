# Kursinis darbas: plan

**Topic:** Neuroninių tinklų modelių palyginimas atpažįstant emocijas iš teksto
**Team:** Teodora Einorytė (**A**), Edgaras Legus (**B**)
**Deliverables:**
1. A technical solution: this repo (code, experiments, results, a small demo).
2. A written report of **22–26 pages** (.docx; 1800 characters with spaces per page; single spacing; Times New Roman 12).
3. A 7–10 min talk with slides.

**Where things live:** code and results in this repo; the report text in the shared Google Doc.
**Fixed dates:**
- report to Moodle by **2026-11-03**
- talk on 2026-11-04, -09 or -10
- slides uploaded one day before the talk

Contents:
1. Research questions
2. Beyond the labs
3. Data
4. Protocol
5. Models
6. Experiments
7. Code
8. Who does what
9. Steps
10. Writing the report
11. Talk
12. Literature
13. Open decisions
14. Review 2 (2026-10-31)

---

## 1. What we want to find out

A lab compares a few models on one task. The coursework is a **study** with four research questions:

1. **Which architecture recognises emotions in short texts best, at what cost, and which differences are real?** 10 models from 4 families, each tuned by the same procedure (size, optimiser, learning rate), 5 seeds each; differences judged with confidence intervals on 52,000 test texts (the official 2,000 + 50,000 more from the same corpus) and corrected for multiple comparisons; parameters and time; a fine-tuned DistilBERT run by us as the competitor.
2. **Which training choices matter for text?** Optimiser and learning rate (9 optimisers), padding/masking, class weights and focal loss for uneven classes, regularisation (Dropout, L2), depth, pretrained word vectors (GloVe), training-set size, 6 vs 2 classes.
3. **Do the models generalise, across corpora and across languages?** Models trained on CARER tested on the human-labelled GoEmotions, and the other way round; the same 10 models trained on the Lithuanian translation of the same texts, so English and Lithuanian are compared on identical texts and labels.
4. **What do the models actually learn: keywords or context?** The labels come from the emotion word, so the study measures how far that explains the results: results per keyword group (pure / ambiguous / no keyword), the keyword test (emotion word removed or kept alone) against a model trained on context only, negation templates, label noise and an audit of the errors every model makes, attention weights.

The dataset is easy and the labels follow the emotion word (section 3): where the word is unambiguous every model is right (GRU: 99.3 %), and 93 % of the errors fall in the 39 % of texts where it is ambiguous or missing. So the architectures may end up within a point of each other. If they do, that is a finding, not a failure: the report shows which differences the confidence intervals can resolve, and questions 3 and 4 carry the study.

**Contributions** (what the report can claim beyond "lab 2 on another dataset"):
1. A quantified diagnosis of a widely used benchmark: how much of it one word explains and how much label noise it carries.
2. An architecture comparison where it matters: per keyword group, with confidence intervals, at equal tuning effort.
3. How much transfers across corpora (CARER ↔ GoEmotions) and across languages (English ↔ Lithuanian on identical texts).

## 2. How this goes beyond lab 1 and lab 2

| | Lab 1 | Lab 2 | Coursework |
|---|---|---|---|
| Task | MNIST | MNIST + IMDB, 2 classes | 6 emotions, uneven classes (3.6 %–33.5 %) |
| Architectures | Dense | 5 (Conv2D, Dense, Conv1D, LSTM, GRU) | 10 in 4 families. New vs lab 2: logistic regression, Transformer, SimpleRNN, BiLSTM, BiLSTM + attention |
| Trained runs | ~30, 1 each | ~21–24, 1 each | ~790: tuning (size, optimiser and learning rate per model) on a 20,000-text selection set, then 5 seeds → mean ± std, paired-bootstrap confidence intervals and McNemar, Holm correction |
| Metrics | accuracy, loss | accuracy, loss, time | per-class precision/recall/F1, macro-F1, confusion matrix, train vs validation gap, parameters (with and without the embedding), train and inference time, results per keyword group |
| Context | – | – | baselines (majority 34.8 %, keyword rule 73.8 %) and competitor: DistilBERT fine-tuned by us, same metrics and test sets |
| Data | ready | ready | own data-quality analysis (keyword labels and their purity, conflicting labels, near-duplicates); 70,000 extra texts from the same corpus for selection and testing |
| Generalisation | – | – | GoEmotions (6,535 texts) in both directions; Lithuanian: the same texts machine-translated, models trained in Lithuanian |
| Extra studies | depth, width, batch, epochs, activations, 3 optimisers × 4 rates (MNIST, Dense) | pooling, one-hot vs Embedding, LSTM/GRU units, layers, activation, optimiser (1 run each) | 9 optimisers × 5 rates on text models (5 seeds), optimiser tuned per model, masking, class weights vs focal loss, Dropout vs L2, depth, GloVe frozen vs fine-tuned, training-set size, binary vs 6 classes |
| Interpretation | – | – | results per keyword group, keyword test against a context-only model, negation templates, audit of the hard errors by both of us, attention heatmaps with an erasure check |
| Product | – | – | `predict.py` + a small web app for English and Lithuanian: sentence in → emotion probabilities, the words the model looked at, and a warning when many words are unknown |

## 3. Data

### dair-ai/emotion (main)
- Saravia et al., CARER, EMNLP 2018, DOI 10.18653/v1/D18-1404. English tweets, labels 0–5: sadness, joy, love, anger, fear, surprise → **liūdesys, džiaugsmas, meilė, pyktis, baimė, nuostaba**.
- Official split: train 16,000 / validation 2,000 / test 2,000.
- Classes (train): joy 33.5 %, sadness 29.2 %, anger 13.5 %, fear 12.1 %, love 8.2 %, surprise 3.6 %.
- Texts: lowercase, no punctuation, 19.2 words on average (median 17, 95th percentile 41, 99th percentile 52, max 66). 15,212 distinct words; the top 10,000 cover 98.3 % of word occurrences.

**Quality issues (these go in the report, not under the rug):**
- 97.3 % of texts contain *feel/feeling/felt*. The labels were assigned from the emotion word, so "i didnt feel humiliated" is *sadness*.
- A one-line rule ("word after *feel*" → most common label in train) gets **73.8 %** on test (macro-F1 0.672). The word after *feel* skips intensifiers ("so", "very", "slightly" …); when it is a pronoun or conjunction ("i feel like i …") the text has no emotion word, which is the case for about a quarter of the texts (`data.emotion_word`, lists checked on validation).
- **How pure the emotion word is** (share of its train texts with its most common label): 56.5 % of test texts have a word that is ≥ 95 % pure, 13.9 % an ambiguous word, 24.9 % no emotion word, 4.7 % a word never seen as an emotion word in train. Pilot GRU (S9, Adam): 99.3 % right on the pure group, 70.0 % on the ambiguous one, 88.6 % without a word; 93 % of its errors are in the ambiguous and no-word groups (`data.keyword_groups`, step 1.11).
- **Label noise:** 51 texts appear more than once with **different** labels. All 11 test texts that also appear in train have a **different** label there (5 of them joy ↔ love; "i feel cared for and accepted" is love in test, joy in train); the 5 validation duplicates likewise. Near-duplicates (word-set Jaccard ≥ 0.7) add 38 test texts, only 42 % with the same label. In the whole CARER corpus 22,295 texts carry two different labels. So the duplicates are not a leak but a floor of label noise.
- Text length says nothing about the class: class means 18.4–20.7 words; a length-only rule is no better than the majority class.
- **Decision:** keep the official split unchanged so our numbers are comparable with published results; report the issues and their size.

### GoEmotions (generalisation in both directions)
- Demszky et al., ACL 2020. Reddit comments, human-labelled, 28 labels.
- We take the single-label texts whose label is one of our 6 emotions: **6,535 texts** (love 1,760, anger 1,265, joy 1,052, sadness 1,003, surprise 902, fear 553); by its own split 5,272 train / 625 validation / 638 test.
- Preprocess like our data (lowercase, strip punctuation).
- **The labels mean different things:** "love" is the most common class (26.9 %, so that is the majority baseline here, not joy) and 1,228 of its 1,760 texts contain the word "love", while in CARER "love" is a joy word as often as a love word; "wow" marks 135 surprise texts but occurs in 5 CARER train texts. Part of the drop is therefore fixed by label definitions before any model runs, which the report says, with per-class results.
- **Used two ways:** CARER-trained models tested on all 6,535 texts (and on the 638-text test split, for a like-for-like number), and models trained on the 5,272 GoEmotions train texts tested on CARER (S17).

### Lithuanian (the same texts translated: training data for S15 and test sets)
- Source: [SkyWater21/lt_twitter_emotions](https://huggingface.co/datasets/SkyWater21/lt_twitter_emotions) (MIT; derived dataset: CARER → the Kaggle "emotions" copy → translated with the free Google Translate API), the whole CARER corpus, i.e. dair-ai/emotion "unsplit", 416,809 texts, machine-translated into Lithuanian. It keeps the English text and the same 6 labels.
- All our texts are in it, matched by English text and label: **all 16,000 train, 2,000 validation and 2,000 test texts** (checked 2026-10-31). The test texts are in `data/lt_test.csv` (`test_index`, `text_lt`, `text_en`, `label`); train and validation follow at step 1.9.
- **Lithuanian is harder for these models** (same texts, our cleaning, train split): 29,043 word forms vs 15,212 in English (61 % appear once, vs 51 %); the 10,000 most frequent cover 91.0 % of word occurrences vs 98.3 %; 11.9 % of validation words are outside that vocabulary vs 3.6 %; texts are shorter (14.3 vs 19.2 words, no articles and fewer pronouns).
- **Why it matters (S15):** the same 10 models trained on the Lithuanian texts with the same protocol give a controlled comparison: identical texts and labels, only the language changes (morphology plus translation noise). Does the ranking of the architectures hold for a language with rich inflection? It also lets the web app take Lithuanian.
- **English models on Lithuanian** (the old plan): 92.1 % of the words are unknown to the English vocabulary, and the words they do know are Lithuanian function words that happen to be English entries (bet, man, kai, su, to, per …), so the predictions come from false friends. One sentence in the report, as a sanity check.
- **Limits, stated in the report:** machine translation, not natural Lithuanian ("translationese"); a translation can lose or change the emotion word ("i feel so strongly in my beliefs …" → "Aš taip tvirtai tikiu …" has no "feel").
- **Own sentences dropped (2026-10-30):** the plan first had 60 hand-written Lithuanian sentences. With 60 texts any accuracy has a ±12-point uncertainty, so the translated sets replace them.

### Extra CARER texts (selection set and extended test, never trained on)
- The same translated corpus holds **394,670 more CARER rows** (373,874 distinct texts) whose text is not among our 20,000 (after cleaning). Texts repeat in the corpus, often with different labels (20,220 of them; 63 % of those have an ambiguous emotion word).
- **How the official split was drawn matters.** It behaves like random *rows* of the corpus: our test texts occur twice as often in the corpus as other texts (11.8 % vs 5.8 % occur more than once), and random rows have the same emotion-word profile as the official test (pure 56.1 %, ambiguous 12.3 %, none 26.4 %, unseen 5.1 % vs 56.5 / 13.9 / 24.9 / 4.7 %). Drawing *unique* texts and dropping two-label ones (our first version) gave half as many ambiguous texts (7.0 %), and every model scored ~7 macro-F1 points higher on it than on validation (Conv1D 0.971 vs 0.904 in the first T0a run). So the extra sets are drawn as random rows.
- **Why:** the official test set is too small for the differences we expect. One run's test macro-F1 varies by 0.010 (bootstrap sd over texts), more than across seeds (0.004 for GRU + Adam), and a 2,000-text validation set is too small to choose between settings: in S9, GRU with Nesterov beat Adam by 0.017 on validation but lost by 0.0025 on test.
- **What we take** (`download_carer_extra.py`, step 1.9): random rows of the other CARER rows, stratified to the official test's class shares, no text in both sets; their emotion-word profile matches the official sets (selection set: pure 56.9 %, ambiguous 12.2 %, none 25.8 %, unseen 5.2 %; extended test 57.0 / 11.9 / 25.8 / 5.3 %; validation 56.8 / 12.4 / 25.5 / 5.3 %):
  - **selection set, 20,000 texts:** all choices in T0 (sizes, optimiser, learning rate) and the "best model" choices are made on it; EarlyStopping still watches the official validation set;
  - **extended test, 50,000 texts:** the confidence intervals and significance tests between models (section 6). It narrows the interval of a difference about 5× (~±0.25 macro-F1 points instead of ±1.2) and has ~1,650 surprise texts instead of 66.
- Both samples keep their Lithuanian translation, so S15 gets the same selection set and extended test in Lithuanian.
- **Headline numbers stay on the official test set** (comparable with published results); the report shows that the extended test agrees with it (class shares, keyword groups, accuracy).

### Binary version (study S5)
Positive = joy, love, surprise; negative = sadness, anger, fear (45.2 % / 54.8 % in train). Surprise is debatable; say so in the report (fear → surprise errors count as crossing the valence only because surprise is mapped to positive).
- **Free first step:** collapse the 6-class predictions of S1 to positive/negative. On the S9 runs this already gives 97.4–98.1 %, and 68–73 % of the 6-class errors stay inside one valence.
- **Trained binary models** only for logreg and the best non-recurrent and recurrent model of S1 (15 runs instead of 50): is a model trained on 2 classes better than the collapsed 6-class one?

## 4. Fixed protocol (`config.py`), the same for every run

| Setting | Value |
|---|---|
| Tokenisation | `TextVectorization`, vocabulary 10,000, adapted on **train only** |
| Sequence length | 50 tokens (98.7 % of train texts fit whole; `length_coverage.csv`) |
| Embedding size | 64, learned (S12: 100-d learned or GloVe) |
| Masking | `Embedding(mask_zero=True)` for sequence models (except where S2 changes it). Exceptions: Conv1D (fixed windows, no use for a mask) and the Transformer, which was better without the mask on validation (step 2.1); listed in `config.NO_MASK_MODELS` |
| Optimiser | **per model**, tuned in T0b: Adam or SGD + Nesterov momentum 0.9 (the best adaptive and the best momentum optimiser of S9), in `config.MODEL_OPTIMIZERS`. Adam 0.001 (`config.OPTIMIZER`, `LEARNING_RATE`) is the default: T0a runs with it, and it stays unless another pair is clearly better (tie rule below) |
| Learning rate | per model, tuned in T0b together with the optimiser: Adam 0.0003 / 0.001 / 0.003 / 0.01, Nesterov 0.03 / 0.1 / 0.3; in `config.MODEL_LEARNING_RATES` |
| Loss | sparse categorical cross-entropy, softmax output (S3: focal loss, γ = 2, α = 1, i.e. exactly the formula in section 6) |
| Regularisation | none beyond EarlyStopping, except the MLP's own Dropout 0.5 and the Transformer block's own Dropout (0.1 after attention and after the feed-forward layer, 0.3 before the output, as in the Keras example). S10 adds Dropout or L2 |
| Depth | 1 Conv1D / recurrent layer, 1 Transformer block (S11: 2 layers) |
| Batch | 64 |
| Epochs | up to 30, `EarlyStopping(val_loss, patience=3, restore_best_weights=True)` on the official validation set; runs whose best epoch is ≥ 28 are flagged as capped (still improving), not as worse |
| Seeds | 1, 2, 3, 4, 5 (`keras.utils.set_random_seed` + `tf.config.experimental.enable_op_determinism()`, so every optimiser repeats bit for bit; without it SGD, Adagrad, Adadelta and RMSprop did not) |
| Evaluation sets | official validation (2,000: EarlyStopping); **selection set** (20,000 CARER texts: every choice); official test (2,000: headline numbers); **extended test** (50,000 CARER texts: confidence intervals and tests between models); GoEmotions; Lithuanian sets (section 3) |
| Model choice | by **selection-set** macro-F1 (mean of the seeds). **Tie rule:** the default (middle T0a size, Adam 0.001) stays unless another setting is better by more than 0.005; a choice at the edge of a grid gets one more step in that direction. The test sets are used only for final numbers |

**Machines:** every study runs entirely on one machine. S1 is where training times are compared, so it runs on B's machine.

## 5. Models (10) and comparison points

| # | Model | Family | Owner | Tuned (3 values) |
|---|---|---|---|---|
| 0 | Logistic regression (multi-hot → softmax) | feedforward | A | – (baseline) |
| 1 | MLP (multi-hot → Dense → Dropout 0.5 → softmax) | feedforward | A | hidden units 32/64/128 |
| 2 | Embedding + GlobalAveragePooling + Dense | feedforward | A | embedding size 32/64/128 |
| 3 | Conv1D + GlobalMaxPooling | convolutional | A | filters 64/128/256 |
| 4 | Transformer encoder (word + position embedding, MultiHeadAttention with 2 heads, FFN, LayerNorm) | attention | A | feed-forward units 64/128/256 (heads 1/2/4 would not change capacity: 677,062 parameters each) |
| 5 | SimpleRNN | recurrent | B | units 32/64/128 |
| 6 | LSTM | recurrent | B | units 32/64/128 |
| 7 | GRU | recurrent | B | units 32/64/128 |
| 8 | BiLSTM | recurrent | B | units 32/64/128 |
| 9 | BiLSTM + attention pooling (one weight per word → heatmap) | recurrent + attention | B | units 32/64/128 |

Not models we train, but points of comparison:
- **Majority class** (always "joy"): 34.8 % on test (GoEmotions: always "love", 26.9 %).
- **Keyword rule** (word after *feel* → label): 73.8 % on test; 97.6 % on the texts whose word is ≥ 95 % pure, ~47 % on the ambiguous ones.
- **Competitor:** published fine-tuned transformers on this same dataset. DistilBERT reports 93.8 % accuracy (93.79 F1) and, elsewhere on the same model page, 92.7 % / macro-F1 0.883. BERT-base reports 94.05 %. They have ~66M+ parameters vs our ~0.7M. These numbers come from model pages, not peer-reviewed papers, and use other metrics, so **S14 fine-tunes DistilBERT ourselves** and measures it exactly like our models (same splits, macro-F1, McNemar, GoEmotions, keyword test, time).

**Pilot run (2026-10-29, 1 seed, untuned).** Test accuracy / macro-F1:

| Model | Result |
|---|---|
| MLP | 89.0 % / 0.844 |
| Embedding | 87.5 % / 0.808 |
| Conv1D | 92.1 % / 0.876 |
| Transformer | 89.9–90.5 % (2 seeds) |
| SimpleRNN (masked) | 81.0 % / 0.689 |
| LSTM (masked) | 90.3 % / 0.852 |
| GRU (masked) | 92.4 % / 0.881 |
| BiLSTM, unmasked | 88.9 % / 0.834 |
| BiLSTM + attention, unmasked | 89.1 % / 0.834 |

- **Without masking:** LSTM collapsed to 34.9 % and SimpleRNN to 51.0 %. That's the origin of S2.
- **The Transformer goes the other way (step 2.1, full data):** with the padding mask (attention only to real words, average over real words) validation 85.9 % and 88.6 % (seeds 1, 2), test 86.5 % / 0.808 and 88.0 %; without it validation 90.8 %, test 90.0 % / 0.851 (seed 1). The reason is not yet known. Text length is ruled out (it says nothing about the class, section 3). A possible one, to be tested: without the mask each padding position attends to the real words and is then averaged in, so the padding positions act as extra summary slots. S2 tests it with a 2 × 2 design (attention mask × pooling mask). The protocol runs the Transformer without the mask, and S2 measures it with the mask (5 seeds). **S2's answer (step 3.2):** mostly the learning rate. With its own rate (Adam 0.003) the masked Transformer is only 0.006 below the unmasked one on the extended test.
- **The attention family is one small model** (1 block, 2 heads); conclusions about "attention models" are stated for that model, with DistilBERT as the large reference.
- **Time:** all models together train in ~3 min on CPU.

## 6. Experiments

Studies are listed in the order they run. "Side models" = Conv1D and GRU, the best non-recurrent and the best recurrent model of the pilot run; side studies use them so that each finding is checked on both families at a manageable cost. Where a study's baseline is the protocol itself, it is **not run again**: `compare.py` takes it from S1.

| ID | What changes (everything else fixed) | Baseline | Runs | Runs on | Owner | Report |
|---|---|---|---|---|---|---|
| S9a | Optimiser grid: 9 optimisers × learning rates 1.0 / 0.1 / 0.01 / 0.001 / 0.0001, side models, seed 1; each optimiser's best rate by validation macro-F1 | – | 9 × 5 × 2 = 90 ✅ | B | B | 4.2 |
| S9 | Optimisers, each at its best S9a rate, side models, 5 seeds; picks the two candidates for T0b (the best adaptive and the best momentum optimiser) | – | 9 × 2 × 5 = 90 ✅ | B | B | 4.2 |
| T0a | Tuning, size: one capacity setting × 3 values per model (Transformer: feed-forward units), Adam 0.001, seeds 1–2; logreg runs once at its only setting (its "× 1" for T0b); pick by selection-set macro-F1 with the tie rule (section 4) | – | 9 × 3 × 2 + 2 = 56, + 8 edge = 64 ✅ | B | each picks for own models | 4.3 |
| T0b | Tuning, optimiser and learning rate at the chosen size: Adam 0.0003 / 0.003 / 0.01 and SGD + Nesterov 0.03 / 0.1 / 0.3 (Adam 0.001 is the T0a run), all 10 models, seeds 1–2; tie rule and edge rule | Adam 0.001 = T0a | 10 × 6 × 2 = 120, + 10 edge = 130 ✅ | B | each picks for own models | 4.3 |
| **S1** | **Main comparison:** 10 tuned models. Saves predictions for the official and the extended test, GoEmotions, the Lithuanian test set and the keyword-test copies (S13); saves every model with its vocabulary. Refuses to start until T0 is filled in for every model | – | 10 × 5 = 50 | B | both | 4.4 |
| S8 | Cost: parameters with and without the embedding, s/epoch and training time (median over seeds), inference time per 1,000 texts from `timing.py` (after a warm-up, median of 5, fixed batch and threads, model only, DistilBERT likewise) → quality vs cost chart | – | from S1 | – | B | 4.4 |
| S14 | Competitor: DistilBERT (`distil_bert_base_en_uncased`, 66M parameters) fine-tuned on the same train split, 3 epochs, learning rate 2e-5; same predictions as S1; in the keyword-test copies the emotion word becomes DistilBERT's own [UNK] token | – | 3 seeds | B (same machine as S1, for the cost chart) | A | 4.4, 4.8, 4.9 |
| S14b | Optional, zero-shot to Lithuanian: multilingual DistilBERT (KerasHub preset, check the name, e.g. `distil_bert_base_multi`) fine-tuned on English train, tested on the Lithuanian test sets | – | 1 seed | B | A | 4.9 |
| S2 | Padding/masking: RNNs (SimpleRNN, LSTM, GRU, BiLSTM) post padding without mask / pre padding without mask; Transformer **with** the mask (its protocol is without), after its own learning-rate search (Adam × 3 rates + the edge step, seeds 1–2); Transformer 2 × 2: attention mask only / pooling mask only; RNN accuracy by text length (free) | protocol = S1 | 4 × 2 × 5 + 8 + 3 + 2 × 5 = 61 | B | B | 4.5 |
| S3 | Uneven classes: class weights (inverse frequency) for all 10 models; focal loss (γ = 2, α = 1) for the side models | neither = S1 | 10 × 5 + 2 × 5 = 60 | B | B | 4.6 |
| S10 | Regularisation: Dropout 0.2 / Dropout 0.5 before the output layer / L2 1e-4 on **all** weights (Embedding, Conv1D or input and recurrent kernels, output layer), side models; compared by train − validation gap and macro-F1 (val_loss includes the penalty) | none = S1 | 3 × 2 × 5 = 30 | B | B | appendix, 1 sentence in 4.7 |
| S11 | Depth: 2 stacked Conv1D / recurrent layers instead of 1, for Conv1D, LSTM, GRU | 1 layer = S1 | 3 × 5 = 15 | B | B | appendix, 1 sentence in 4.7 |
| S12 | Word vectors, all 100-dimensional: learned / GloVe frozen / GloVe fine-tuned / GloVe frozen with the vocabulary extended to the 40,000 most frequent GloVe words (so words unseen in train get vectors too), side models; GloVe coverage of train, GoEmotions and the emotion words; also saves GoEmotions, LT test and keyword-test predictions | learned 64-d = S1 | 4 × 2 × 5 = 40 | A | A | 4.7 + appendix |
| S4 | Training-set size: 1k, 2k, 4k, 8k, 16k (stratified) for logreg, the best non-recurrent and the best recurrent model of S1 (chosen on the selection set: Conv1D 0.8905; GRU 0.8973 ties with BiLSTM + attention 0.8974, GRU taken as simpler, 2.4 times faster and the side studies' model); plus the keyword rule rebuilt from each size (free) | – | 5 × 3 × 5 = 75 | A | A | 4.7 + appendix |
| S5 | 2 classes vs 6 classes: S1 predictions collapsed to positive/negative (free) + binary models for logreg and the best non-recurrent and recurrent model (Conv1D, GRU, as in S4) | 6 classes = S1 | 3 × 5 = 15 | A | A | 4.7 + appendix |
| S15 | **Lithuanian:** all 10 models trained on the Lithuanian translation of the same train split, with their English-tuned settings; vocabulary fitted on Lithuanian train; tested on the Lithuanian official and extended test; also saves keyword groups by the English text | English = S1 | 10 × 5 = 50 | A | A | 4.9 |
| S16 | **Context-only reference:** Conv1D, GRU and BiLSTM + attention trained on train texts with the emotion word replaced by the placeholder `xxunknownxx` (every occurrence; a word of its own in the vocabulary, not [UNK]), tested on the "without" copy of test | – | 3 × 5 = 15 | A | A | 4.8 |
| S17 | **Cross-corpus:** logreg, Conv1D, GRU and BiLSTM + attention trained on GoEmotions train (5,272 texts; its validation split for EarlyStopping), tested on GoEmotions test and on CARER test | CARER → GoEmotions from S1 | 4 × 5 = 20 | A | A | 4.9 |
| S13 | **Keywords or context?** No retraining; reads S1, S12, S14 and S16 predictions. Results per keyword group; the keyword-test copies (emotion word → [UNK] everywhere it occurs; emotion word alone, and "i feel <word>"); an [UNK] control on a random other word; negation templates; attention share and an erasure check (details below) | original test | 0 | – | A | 4.8 |
| S6 | Generalisation analysis: S1, S12, S14 and S17 predictions on GoEmotions (per class, majority = love, 5-class check without love), S15 vs S1 on identical texts, English models on Lithuanian as a sanity check | – | 0 | – | A | 4.9 |
| S7 | Errors: confusion matrices, most-confused pairs (joy ↔ love is the largest in every S9 model), audit of the texts every model gets wrong (both of us label a sample), attention heatmaps | – | 0 | – | B | 4.10 |

About 790 training runs (180 of them done: S9a, S9), roughly 8–12 hours of CPU time for our models (side-model runs take ~10–90 s each) plus several hours for the DistilBERT runs.

**Core and side studies.** S1 (with S8, S14), S2, S3, S13 (with S16), S6 (with S15, S17) and S7 answer the research questions directly; they get their own report sections and are discussed in depth. S9 gets a short section (it chose the two optimiser candidates). S10, S11, S12, S4 and S5 are side studies: one shared report section (4.7) with a summary table, full tables and figures in the appendices, which do not count towards the 22–26 pages; S10 and S11 get one sentence each, since their expected effects are below what the test sets can resolve.

**S9, optimisers (runs first, before T0):**
- **Optimisers (9):** all eight from lecture 4 (SGD, SGD + Momentum 0.9, Nesterov, Adagrad, Adadelta, RMSprop, Adam, Nadam; lecture 5 compares six of them in a chart), plus AdamW (Adam with decoupled weight decay, standard for Transformers). Keras defaults otherwise.
- **Why a learning-rate grid:** lab 1 (task 6.3) showed that SGD needs a much larger learning rate than Adam or RMSprop, and Adadelta is normally used with 1.0. Comparing all of them at 0.001 would be unfair, so every optimiser gets the same 5-value grid (S9a) and its own best rate before the 5-seed comparison (S9).
- **Why the side models:** recurrent networks are where plain SGD is expected to struggle (gradients pass through 50 time steps), so the two families may react differently.
- **Sizes:** the middle T0 value (Conv1D 128 filters, GRU 64 units), because T0 runs after S9.
- **The first decision rule** was one optimiser for every model: keep Adam if it is within one standard deviation of the best optimiser's mean validation macro-F1. By that rule Adam 0.001 stayed (0.898 vs Nesterov 0.902 ± 0.009).
- **What S9 showed and what we decided (2026-10-31):**
  - The optimiser and the model interact on validation: Conv1D is best with Adam/AdamW (0.904), GRU with Nesterov (0.909 vs Adam 0.893), larger than some differences between architectures.
  - **But it did not hold on test:** GRU Nesterov − Adam was +0.017 on validation and −0.0025 on test. The 2,000-text validation set is too small to choose on, and S9 picked each optimiser's rate on that same set (so its S9 numbers are optimistic).
  - So the optimiser is tuned **per model**, as a precaution and by the same procedure for every model: T0b tries the best adaptive optimiser (Adam) and the best momentum optimiser (Nesterov) of S9, chosen on the 20,000-text selection set with the tie rule (Adam 0.001 stays unless another pair is better by more than 0.005). The report presents it as that, not as "GRU works better with Nesterov".
  - Other S9 facts for 4.2: SGD at 1.0 failed on Conv1D in 4 of 5 seeds (1 diverged, 3 got stuck at 9–30 % macro-F1), so 1.0 is at the edge of stability; 35 of 90 S9a runs hit the 30-epoch cap (31 with their best epoch at 30); SGD's best rate is the top of the grid.

**T0, procedure per model (the same for all 10):** sizes first (T0a, Adam 0.001), then optimiser and learning rate together at that size (T0b), because the right rate depends on the optimiser (S9a). Each setting is judged by the mean selection-set macro-F1 of seeds 1–2, with the tie rule and the edge rule (section 4); a choice at the edge (Adam 0.01 or 0.0003, Nesterov 0.3 or 0.03) gets one more × 3 step. The budget is the same for every model (3 sizes, 7 optimiser–rate pairs, 2 seeds), which the report shows in a table. S1 then compares each architecture at its own tuned size, optimiser and rate, under the same tuning budget.

**S2, masking (details):**
- **RNNs:** without the mask the hidden state also passes the padding, so post padding should hurt most on short texts; the accuracy-by-length plot (free, from the saved predictions) checks that.
- **Transformer:** its protocol runs without the mask; S2 adds the masked version after its own short learning-rate search (Adam 0.0003 / 0.001 / 0.003, seeds 1–2, selection set, with the edge rule of section 4: 0.003 was best, so 0.01 too), so the comparison is not tuned vs untuned. The 2 × 2 design separates the two places the mask acts: the attention mask (each word attends only to real words) and the pooling mask (average over real words only).

**S10–S12, side studies (run after S1, with the T0 settings):**
- **S10 regularisation (lecture 6):** Dropout drops each unit with probability p during training (lecture 6, §3); L2 adds λ·Σw² to the loss (§1.1) for **all** weights: the Embedding (640k of ~670k parameters, where most memorising happens), the Conv1D or input and recurrent kernels, and the output layer. Compare the train − validation gap and macro-F1, not raw val_loss, because the reported loss includes the penalty; EarlyStopping stays on in every arm, so the question is what they add to it.
- **S11 depth (lab 2, task 2.3):** two stacked layers (`return_sequences=True` on the first recurrent layer). Shows whether a deeper network pays off on 19-word texts.
- **S12 pretrained word vectors (lecture 6, §6.2):** GloVe 6B 100-d (Pennington et al., 2014). The learned arm is also 100-d, so only the origin of the vectors changes. Our vocabulary is fitted on train, so GloVe normally gets vectors only for words already seen in training; the fourth arm extends the vocabulary with the 40,000 most frequent GloVe words, so words unseen in train (7.9 % of GoEmotions word occurrences) get vectors too. Questions: do pretrained vectors help with 16,000 training texts, on GoEmotions (another domain) and on the 94 test texts whose emotion word never occurs as one in train? Report GloVe's coverage first (our texts merge contractions: "didnt", "im"). The GloVe file (~330 MB) is downloaded by `download_glove.py`, kept out of git.
- **Focal loss (lecture 4, §2.1):** FL = −(1 − p_t)^γ · log p_t with γ = 2: well-classified texts contribute less, so the rare classes weigh more without explicit weights. Keras's `CategoricalFocalCrossentropy` multiplies by α = 0.25 by default, which would be a 4× smaller step for SGD; we set α = 1, so the loss is exactly this formula.

**S13, keywords or context? (the most direct answer to research question 4):**
- **Why the plain keyword test is not enough:** the gold label was made from the emotion word, so every model must lose accuracy without it and keep it with the word alone. S13 therefore measures *how much* one word explains, against two references: the keyword rule (pure lookup) and S16 (a model that never saw the word).
- **Keyword groups** (`data.keyword_groups`, step 1.11; purity = share of the word's train texts with its most common label): pure (≥ 95 %, 56.5 % of test), ambiguous (13.9 %), no emotion word (24.9 %), word unseen as an emotion word in train (4.7 %). Every core table (S1, S14, S15, S16) gets a column per group. This is where the architectures can differ: on the pure group everything is ~99 % right, while on the ambiguous group the S9 models beat the lookup by ~23 points.
- **Modified copies of the test set**, built in `data.keyword_variants`. The emotion word (`data.emotion_word`) is the first word after *feel / feels / feeling / felt*, skipping intensifiers and small words ("so", "very", "slightly" …), the same word the keyword rule (73.8 %) uses. A text counts as a keyword text only if the word is in the run's vocabulary (1,459 of the 1,502 test texts with a word). All models are compared on the same texts: these 1,459 of S1 (`kw_has` of an S1 run).
  - *Without the keyword*: **every** occurrence of the word becomes a word outside the vocabulary, so it is encoded as [UNK]. "i feel so **humiliated** today" → "i feel so **[UNK]** today"; "i feel crappy i eat crappy" loses both.
  - *Only the keyword*: "humiliated"; and *"i feel" + keyword*: "i feel humiliated", so the input is not unusually short. Metrics on keyword texts only (the others would be all padding).
  - *[UNK] control*: a random other word of the same text becomes [UNK], to separate "lost the emotion word" from "got an [UNK]" ([UNK] itself leans to joy: 43 % of the train texts with an out-of-vocabulary emotion word are joy).
- **Negation templates** (CheckList, Ribeiro et al., 2020): 10 pure emotion words per class × "i feel X" / "i dont feel X" / "i never feel X". No labels are needed: we measure how often the prediction changes and how much the probability of X's class drops. Run on DistilBERT too: if it also ignores negation, the labels taught the shortcut. This replaces accuracy on a negation subset, which is scored against labels that ignore negation (the keyword rule is right on 85 % of the 48 negated test texts).
- **What we measure**, per model: accuracy and macro-F1 on each copy and group vs the original test set; the drop without the keyword read against what context alone supports (S16). Plotted for all models, the keyword rule, S16 and DistilBERT in one chart.
- **Link to attention:** for the BiLSTM + attention model, the share of attention weight on the keyword, and an erasure check: removing the most-attended word vs a random word, how much the prediction changes (attention as explanation is debated: Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).
- **No retraining:** S1 (and S12, S14, S16) save predictions for the copies, like for GoEmotions; the templates are scored by `generalization.py` with the saved models.

**S16, context-only reference:** Conv1D, GRU and BiLSTM + attention trained on train texts with the emotion word replaced by the placeholder `xxunknownxx` (same function as the test copy), tested on the "without" copy of test. The placeholder is replaced before the vocabulary is built, so it is a word of its own there (the 3rd most frequent), not [UNK]: the model knows a word stood there but not which, and it gets no hint from [UNK], which leans to joy. Its test copy replaces every emotion word, as in its training texts (its vocabulary lacks 54 test emotion words). The context-only result is read from the predictions `p_kw_without`; the `test_*` columns of its rows in results.csv are on the original test texts. It answers "how much can context alone support?", the analogue of hypothesis-only baselines (Gururangan et al., 2018; Poliak et al., 2018), so "a model loses X points without the keyword" can be read against "context alone reaches Y".

**S15, Lithuanian (translate-train):** the same 10 models, trained on the Lithuanian translations of our train split with each model's English-tuned size, optimiser and rate (one change: the language); the vocabulary (10,000) and sequence length (50) stay, fitted on Lithuanian train. Tested on the Lithuanian official test and extended test. Compared with S1 text by text (paired bootstrap), per architecture and per keyword group (groups from the English original). Report the vocabulary coverage next to English (91.0 % vs 98.3 %). Expected: models that read word order and subword-like patterns lose less than bag-of-words models, but it is an open question. The best Lithuanian model goes into the app.

**S17, cross-corpus:** four models (logreg, Conv1D, GRU, BiLSTM + attention) trained on the 5,272 GoEmotions train texts (its validation split for EarlyStopping; settings from T0), tested on GoEmotions test (in-domain) and CARER test. With S1 this gives a 2 × 2 table (train corpus × test corpus, as in Bostan & Klinger, 2018) and a second ranking of the architectures on human labels: the cleanest check of whether keyword labels flatten the differences between architectures.

**S6, analysis (details):** per-class recall and the confusion matrix on GoEmotions (love is the majority there, 26.9 %), a 5-class check without love, CARER → GoEmotions both on all 6,535 texts and on its 638-text test split (like-for-like with S17); S15 vs S1; the English models on Lithuanian as one sanity sentence.

**S7, errors (details):** confusion matrices and most-confused pairs; **audit**: the test texts that ≥ 80 % of the S1 runs get wrong, of which each of us labels the same random 100 independently into categories (gold label wrong or ambiguous, ambiguous emotion word, no emotion word, negation, sarcasm, real model error); report the shares and our agreement (Cohen's κ). It estimates how many of the remaining errors are not the models' fault. Attention heatmaps of typical cases.

**S14, competitor (DistilBERT fine-tuned by us):**
- KerasHub `DistilBertClassifier` from the `distil_bert_base_en_uncased` preset (Sanh et al., 2019), fine-tuned on our train split with our validation set and EarlyStopping; texts cut at 64 word pieces; Adam 2e-5, batch 32, up to 3 epochs (the usual BERT fine-tuning settings), seeds 1–3.
- Uses its own tokenizer, so it skips our `TextVectorization`; everything after the predictions (metrics, confidence intervals, McNemar, cost, GoEmotions, LT test, keyword test) is shared with our models. In the keyword-test copies the emotion word becomes DistilBERT's own [UNK] token (our `xxunknownxx` would be split into real word pieces).
- **S14b (optional):** the multilingual DistilBERT preset fine-tuned the same way on English, tested on the Lithuanian sets without any Lithuanian training (zero-shot transfer, Pires et al., 2019). Our models cannot do this at all (92 % unknown words), so it shows what multilingual pretraining adds.
- Downloads the pretrained weights (~250 MB) once; kept out of git. Answers "how far are our 0.7M-parameter models from a 66M-parameter pretrained transformer, and at what cost?"

**Label noise, not a leak:** the 11 test texts that also appear in train all have a different label there (section 3), so a model that memorised train gets them wrong. The report presents them, the near-duplicates and the 22,295 two-label CARER texts as a floor of label noise (with the S7 audit), not as leakage; the main table needs no "without duplicates" column.

**What stays fixed, and why (one sentence each in report 3.1 / 3.7):**
- **Hold-out validation, not k-fold:** the dataset has an official split; lecture 6 (§5) notes that k-fold is rare for neural networks because of training cost, and 5 seeds give the spread instead.
- **Batch size 64:** a common middle value (lecture 4, §3.7: too small is noisy, too large is slow); lab 1 (task 6.1) already studied batch size, so it is not repeated.
- **Activations:** ReLU in Dense/Conv1D hidden layers (best in lab 1, task 6.2), tanh inside recurrent cells (the Keras and lecture 8 default), softmax output.
- **Binary task (S5) uses a 2-unit softmax:** with 2 classes it is mathematically equivalent to 1 sigmoid unit + binary cross-entropy, and keeps one output type for all studies.
- **EarlyStopping on val_loss, choices on selection-set macro-F1:** the loss is smooth and moves every epoch, so it is a stable stopping signal; macro-F1 is what we report, so it decides between settings, on a set ten times larger than validation (S9 showed why).
- **Pooling:** GlobalMaxPooling after Conv1D (Kim, 2014); lab 2 (task 1.2) already compared max and average pooling.
- **Transformer position embedding:** learned, as in the Keras example; sine/cosine encoding (lecture 9, §3.9) gives similar results for short texts.
- **Sequence length 50:** covers 98.7 % of train texts (`length_coverage.csv`). It is also above the IQR outlier bound of the word counts (lecture 1, §8.4): Q1 = 11, Q3 = 25, Q3 + 1.5·IQR = 46, so only outlier-length texts (2.3 % are longer than 46 words) get cut.

**Statistics: which differences are real.**
- **Why more than mean ± sd:** the spread over seeds shows only training noise. The test texts are a sample too, and on 2,000 texts that noise is larger (one run's macro-F1: bootstrap sd 0.010 over texts vs 0.004 over seeds for GRU + Adam). The 95 % interval of a difference between two 5-seed means is about ±1.2 macro-F1 points on the official test, while the pilot's top three were 0.3 points apart.
- **Main test: paired bootstrap over test texts** (`evaluation.bootstrap_diff`; Efron & Tibshirani, 1993; Berg-Kirkpatrick et al., 2012): draw the extended test texts with replacement 1,000 times; each time compute both models' macro-F1 (and accuracy) for each seed, average over the seeds, and take the difference. The 2.5 % and 97.5 % points give the 95 % confidence interval; p = twice the share of differences on the other side of 0. Implemented in numpy and checked on a small example like the other metrics.
- **Reference model:** the best by selection-set macro-F1, not by test. Each model (and DistilBERT) is compared with it; Holm–Bonferroni correction per table (`evaluation.holm`, Holm, 1979): sort the m p-values, compare the i-th smallest with 0.05 / (m − i + 1), stop at the first that fails.
- **Wording rule:** "better" when the interval excludes 0 and survives Holm; "practically equal" when the whole interval lies within ±1 macro-F1 point; otherwise "no clear difference". The main table groups the models that cannot be told apart instead of a strict ranking.
- **McNemar, the second test:** per seed on the official test, χ² = (|b − c| − 1)² / (b + c), p from the χ² distribution with 1 degree of freedom (`math.erfc`), b and c printed. Rule declared in advance: a difference counts if it is significant after Holm in at least 4 of the 5 seeds. It tests accuracy, the bootstrap tests macro-F1; the report says when they disagree.
- **Seeds:** if the top group's members are within 0.5 points, they get 3 more seeds (+15 runs), because with 52,000 test texts the seed noise becomes the limit.

**Not in the plan (only if both agree later):**
- A CNN + BiLSTM hybrid.

## 7. Code

```
neuroniniai-tinklai-kd-2026/
  README.md, PLAN.md, requirements.txt, .gitignore
  config.py            shared settings (section 4), class names in English and Lithuanian; learning rates from
                       S9a; sizes, optimisers and learning rates from T0 are filled in here
  download_data.py     A: one-off script that created data/*.csv from Hugging Face
  download_lt_test.py  B: one-off script: downloads the Lithuanian translation of the CARER corpus and keeps
                       our 2,000 test texts → data/lt_test.csv (the 56 MB download itself is not kept)
  download_carer_extra.py  B: one-off, same corpus: Lithuanian train and validation (our texts), and the selection
                       set (20,000) and extended test (50,000) from the other CARER texts, English + Lithuanian
  download_glove.py    A: one-off download of GloVe 6B 100-d into data/glove/ (git-ignored, ~330 MB) for S12
  data.py              A: loading (English or Lithuanian), vectorising, pre/post padding, binary version, subsets,
                       GoEmotions, LT, selection set and extended test, GloVe matrix (optionally extended vocabulary),
                       keyword-test copies and keyword groups (S13), negation templates
  data_analysis.py     A: data analysis figures + quality checks + baselines + label-noise numbers (report 4.1)
  models_a.py          A: logreg, MLP, Embedding, Conv1D, Transformer
  models_b.py          B: SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention
  evaluation.py        B: metrics in numpy, confusion matrix, McNemar, Holm correction, paired bootstrap, plots
  train.py             B: one run → one CSV row + curve + matrix + predictions (+ saved model and vocabulary for S1, S15)
  run_study.py         B: runs a whole study: python run_study.py S1
  competitor.py        A: S14 (and S14b), fine-tunes DistilBERT (KerasHub) and writes the same results row and predictions
  compare.py           report tables and figures, one function per study; tables ready for the report (Lithuanian
                       headers, decimal comma, "vidurkis ± st. nuokrypis"), figures with Lithuanian model names
                       (B: S9, T0, S1 + S14 + bootstrap + Holm, S8, S2, S3, S10, S11, S7; A: S12, S4, S5)
  generalization.py    A: S6, S13, S15, S16, S17 tables and figures; negation templates and the erasure check
  timing.py            B: S8 inference time of the saved S1 models and DistilBERT (warm-up, median of 5)
  predict.py           A: command-line demo
  app.py               A: small web app (Gradio): English or Lithuanian sentence in → bar chart of emotion
                       probabilities, the sentence with words shaded by attention weight, and a warning when
                       many words are unknown to the model; choice of model (best overall / attention model)
  data/                emotion_{train,validation,test}.csv, goemotions_{train,validation,test}.csv, lt_test.csv,
                       lt_{train,validation}.csv, carer_select.csv, carer_xtest.csv
                       (original texts; cleaning and filtering happen in data.py)
  results/             results.csv, curves/, matrices/, predictions/, tables/, figures/,
                       models/: only the app's models are committed (best English, best BiLSTM + attention,
                       best Lithuanian), each with its vocabulary file; the rest and dev/ are git-ignored
```

### Interfaces: agree on these first, then both of us can code in parallel

```python
# data.py (A)
load_emotion(task="6", fraction=1.0, seed=1) -> dict
    # x_train, y_train, x_val, y_val, x_test, y_test (np.array of str / int); "fraction" = stratified share of train (S4)
vectorize(d, mode="sequence", padding="post") -> dict
    # mode "sequence" → (n, 50) int; "multi_hot" → (n, 10000); vocabulary from train only; padding "pre" for S2
    # also returns the fitted "vectorizer"; encode(vectorizer, texts, mode, padding) encodes any other texts
load_goemotions(task="6") -> (x, y)
load_lt_test(task="6") -> (x_lt, x_en, y, test_index)   # the 2,000 test texts translated (data/lt_test.csv), cleaned
                                                #   like the rest; test_index = row in emotion_test.csv, to pair LT and EN
load_emotion(..., lang="lt")                    # S15: the same splits in Lithuanian (lt_train/lt_validation/lt_test)
load_carer_extra(part, lang="en") -> (x, y)     # part "select" (20,000) or "xtest" (50,000); never trained on
glove_matrix(vectorizer, dim=100) -> np.array   # (vocabulary, 100); row i = GloVe vector of word i, zeros if missing (S12)
keyword_variants(texts, vocabulary) -> dict     # S13: "without" (every occurrence → out-of-vocabulary word), "only",
                                                #   "i_feel" ("i feel <word>"), "control" (a random other word → [UNK]),
                                                #   "keyword" (the word or ""), "has_keyword" (word found and in vocabulary)
keyword_groups(texts, train) -> np.array        # "pure" / "ambiguous" / "none" / "unseen" per text (purity ≥ 95 % in train)
negation_templates() -> DataFrame               # S13: word, class, template ("i feel X" / "i dont feel X" / "i never feel X")
test_in_train(d) -> np.array                    # bool per test text: the same cleaned text is in train (11 texts)

# models_a.py / models_b.py: every function returns an UNCOMPILED keras.Model
build_lstm(size=64, mask=True, n_classes=6) -> keras.Model
MODELS_B = {"lstm": (build_lstm, "sequence", [32, 64, 128]), ...}   # function, input type, T0 values
# Side-study arguments, passed only when a study changes them (see models_a.py):
#   Conv1D and GRU: dropout, l2, layers, embedding_dim, embedding_matrix, trainable (S10, S11, S12)
#   LSTM: layers (S11); the B models take all of them

# evaluation.py (B)
metrics(y, p, n_classes) -> dict      # accuracy, macro_f1, precision/recall/f1 per class
confusion_matrix(y, p, n_classes) -> np.array
mcnemar(y, p1, p2) -> (chi2, p_value)
holm(p_values, alpha=0.05) -> np.array   # bool per test: still significant after the Holm–Bonferroni correction
bootstrap_diff(y, preds1, preds2, metric, n=1000, seed=0) -> (diff, ci_low, ci_high, p_value)
                                         # preds = one prediction array per seed; paired: the same resampled texts
```

```bash
python train.py --model lstm --seed 1 --study S1 [--size 64] [--no-mask | --mask] [--padding pre]
                [--class-weight] [--loss focal] [--fraction 0.25] [--task 2] [--optimizer sgd --lr 0.1]
                [--dropout 0.5] [--l2 0.0001] [--layers 2] [--embedding glove-frozen] [--dev]
python run_study.py S2           # all runs of one study (S9a, S9, T0a, T0b, S1–S5, S10–S12, S15–S17)
python competitor.py --seed 1    # S14, one DistilBERT run
python compare.py                # tables and figures for the report
python predict.py "i feel so alone tonight"
python app.py                    # web demo on http://localhost:7860
```

**`results.csv`, one row per run:**
- run info: `date`, `study`, `model`, `family`, `variant`, `seed`, `machine`, `optimizer`, `lr`
- model and training: `params`, `epochs`, `best_epoch` (the one EarlyStopping restored), `train_time_s`, `predict_ms_1000`
- train (best weights, evaluated after training): `train_acc`, `train_loss`; the train − validation gap shows overfitting (lecture 5)
- validation: `val_acc`, `val_loss`, `val_macro_f1`
- selection set: `sel_acc`, `sel_macro_f1` (every run; all choices use these)
- test: `test_acc`, `test_loss`, `test_macro_f1`, `f1_sadness` … `f1_surprise` (binary runs: `f1_negative`, `f1_positive`)
- extended test: `xtest_acc`, `xtest_macro_f1` (S1, S15 and the studies compared with them)

Predictions for every text are saved per run in `predictions/` (test; in S1, S12, S14, S15 and S16 also the extended test, GoEmotions, the LT test set and the keyword-test copies), which S6, S7, S13, the bootstrap and McNemar need. Extended-test predictions are stored as class numbers only (uint8, 50 kB per run). `optimizer` and `lr` hold what was actually used, so the per-model choices from T0b show in every row; `run_study.py` also uses them to recognise finished runs.

New packages: `keras-hub` (S14) and `gradio` (app); both go into `requirements.txt` with exact versions.

## 8. Who does what (summary)

| | A — Teodora | B — Edgaras |
|---|---|---|
| Models (5 each) | logreg, MLP, Embedding, Conv1D, Transformer | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention |
| Shared code | `data.py`, `data_analysis.py`, `generalization.py`, `competitor.py`, `predict.py`, `app.py` | `evaluation.py`, `train.py`, `run_study.py`, `timing.py`, `download_carer_extra.py`, most of `compare.py` |
| Studies | data analysis, S13 keyword test, S16 context-only, S14 DistilBERT (run on B's machine), S12 GloVe, S4, S5, S6, S15 Lithuanian, S17 cross-corpus, demo app | S9 optimisers, running T0a/T0b and S1, S2, S3 (+ focal loss), S10 regularisation, S11 depth, S7 (+ the error audit, both label), S8, statistics (bootstrap, Holm) |
| Report text | ~23,600 characters (section 10) | ~21,600 characters + drafts of 4.3, 4.4, 3.5.11, conclusions (section 10) |
| Talk | ~half of the slides | ~half of the slides |
| Also | repo setup, Google Doc owner: outline, styles, final formatting | number check, character/page count |

## 9. Steps

Phases overlap. Writing that needs no results (literature, model theory) starts in phase 1. **"Needs"** lists the steps that must be finished first.

### Phase 0 — Agree and set up
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 0.1 | Read this plan, settle the open decisions (section 13) | both | – | plan merged in `main` | ✅ |
| 0.2 | Repo setup: `requirements.txt`, `.gitignore`, `config.py`, data CSVs (emotion, GoEmotions) | A | 0.1 | `data/` in repo | ✅ |
| 0.3 | Google Doc: outline from section 10, styles (TNR 12, single spacing, Heading 1–3, captions), character budget per section as a comment, working list of sources | A | 0.1 | doc matches section 10 | ✅ budgets as "[owner · ~N ženklų]" before each section note (not comments); 26 starter sources in LITERATŪROS SĄRAŠAS |
| 0.4 | Register the talk date | both | – | date booked |  |

### Phase 1 — Shared code
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 1.1 | `data.py`: `load_emotion`, `vectorize` (post/pre padding, multi-hot), binary, stratified subsets, GoEmotions, LT | A | 0.2 | functions match section 7 | ✅ |
| 1.2 | `evaluation.py`: metrics in numpy, confusion matrix, McNemar, plots in Lithuanian; checked on a small hand-computed example | B | – | example matches hand calculation | ✅ `python evaluation.py` checks a 10-text example against the hand calculation and Keras `F1Score` |
| 1.3 | `train.py` + `run_study.py` | B | 1.1, 1.2 | `python train.py --model logreg --seed 1 --dev` writes a row, curve and matrix | ✅ also tested: S1 extras (GoEmotions, LT/EN, saved model), no mask, pre padding, class weights, binary task, `run_study.py` skips finished runs. `models_a.py` starts with logreg; model keys in `train.FAMILIES` |
| 1.4 | `data_analysis.py`: class and length plots, quality checks, majority and keyword baselines | A (finished by B) | 1.1 | figures and numbers for 4.1 | ✅ tables: classes, vocabulary, length, dataset overview, GoEmotions filter (A); `quality_checks.csv`, `baselines.csv`, `emotion_words.csv` and figures `class_distribution.png`, `length_distribution.png` (B). All section 3 numbers reproduced: 97.3 % feel-words, 18.5 % negations, 51 conflicting texts, 11 / 5 overlaps, majority 34.8 %, keyword rule 73.8 % (macro-F1 0.672; first version 72.4 %, see 1.6). The emotion word is `data.emotion_word` |
| 1.5 | ~~LT sentences: 30 each + English translations~~ | both | – | – | dropped 2026-10-30: replaced by the 2,000-text LT test set (1.8), see section 3 |
| 1.6 | `data.py`: `keyword_variants` (S13, using `data.emotion_word` so it removes the same word the keyword rule uses) and `test_in_train` (duplicates check), each checked on a few hand-made examples | A (done by B) | 1.1, 1.4 | functions match section 7 | ✅ `python data.py` checks both on hand-made examples (incl. "i feel very very sad" → "sad", "feeling slightly more optimistic" → "optimistic", "i feel like i …" → none, the replacement word encodes as [UNK]); skip lists extended after looking at the examples and chosen on validation (keyword rule 71.0 % → 73.0 % on validation); 75.1 % of test texts have an emotion word, the same word the keyword rule uses for every one of them; `test_in_train` finds the 11 overlapping test texts; S1 `--dev` runs save the keyword-test predictions with the real function |
| 1.7 | Code for the review additions: `evaluation.holm` (checked on a hand example), per-model learning rate in `train.py` (`config.MODEL_LEARNING_RATES`), T0a/T0b in `run_study.py`, keyword-test predictions in S1/S12 | B | 1.6 for the keyword predictions | tested in `--dev` | ✅ Holm checked by hand in `python evaluation.py`; S1 variant names stay the same when the T0b rate is used (the rate is in the `lr` column); T0b refuses to start before `MODEL_SIZES` is filled; S1/S12 save `p_kw_without`, `p_kw_only`, `kw_has` and refuse to start until `data.keyword_variants` (1.6) exists. Tested with a stand-in for 1.6 |
| 1.8 | LT test set: `download_lt_test.py` → `data/lt_test.csv` (our test texts matched by English text and label, unmatched ones counted), `data.load_lt_test`, and S1/S12 save its predictions | B (small `data.py` function agreed with A) | 1.1 | ~2,000 rows, a few rows checked by eye; `--dev` S1 run saves `p_lt_test` | ✅ all 2,000 test texts found with the same label (235 appear twice in the corpus, one kept); one example per emotion checked by eye, no empty or untranslated rows; 92.1 % of the Lithuanian words are unknown to our English vocabulary (English test: 3.6 %); S1/S12 save `p_lt_test`, `prob_lt_test`, `y_lt_test`, `lt_test_index` |
| 1.9 | Extra data: `download_carer_extra.py` → Lithuanian train and validation (our texts, matched by English text and label), selection set (20,000) and extended test (50,000) drawn as random rows of the other CARER rows like the official split, stratified to its class shares, no text in both sets, English + Lithuanian; `data.load_carer_extra`, `load_emotion(lang="lt")` | B (data.py functions agreed with A) | 1.8 | counts and class shares printed and match section 3; a few rows checked by eye | ✅ all 16,000 train and 2,000 validation texts matched (same row order as the English files). **Fixed 2026-10-31:** the first version drew unique texts and dropped two-label ones, which left half as many ambiguous texts (7.0 % vs 13.9 %), and the first T0a run scored ~7 points higher on it than on validation; now random rows, emotion-word profile as the official sets (section 3). T0a was rerun |
| 1.10 | Review-2 fixes (section 14.1): keyword-test copies replace every occurrence and check the vocabulary; focal α = 1; L2 on all weights; `run_study.py` guards and finished-run key with the actual optimiser and rate; logreg T0a run; `enable_op_determinism`; the Transformer's T0 setting = feed-forward units 64/128/256 | B (A's files agreed with A) | – | each checked in `--dev` and on a hand example | ✅ `python data.py` checks the new copies on hand examples ("i feel crappy i eat crappy" loses both; a word outside the vocabulary does not count; the control changes exactly one other word); focal α = 1; L2 also on the Embedding (unless frozen) and the recurrent kernels; S1 and later refuse to start until T0 is filled in (`--dev` skips the check), finished runs are recognised by study, model, variant, seed, optimiser and rate; T0a has 56 runs with logreg; `enable_op_determinism`: two SGD dev runs now give identical losses; Transformer sizes 64 / 128 / 256 feed-forward units (668,806 / 677,062 / 693,574 parameters) and the mask modes "attention" / "pooling" for S2 |
| 1.11 | Analysis code: `evaluation.bootstrap_diff` (checked on a hand example), `data.keyword_groups`, `data.negation_templates`, selection-set metrics and extended-test predictions in `train.py`, vocabulary saved with S1/S15 models, `config.MODEL_OPTIMIZERS` and the T0b grid in `run_study.py`, S15–S17 in `run_study.py` | B | 1.9, 1.10 | tested in `--dev` | ✅ `python evaluation.py` checks the bootstrap by hand (difference −0.3 on the 10-text example, identical models → p = 1, always right vs always wrong → p = 0, mean over seeds); keyword groups reproduce 56.5 / 13.9 / 24.9 / 4.7 %; every run writes `sel_acc`, `sel_macro_f1`, studies from S1 on also `xtest_*` and `p_xtest` (old rows get empty cells); S1 and S15 save the model with its vocabulary; `config.MODEL_OPTIMIZERS` and `T0B_GRID` (T0b: 120 runs); S2 (Transformer arms), S5, S15, S16, S17 in `run_study.py`. One `--dev` run each of S1 (GRU, BiLSTM + attention), S15, S16, S17, S2 (mask, attention mask), S3 focal, S10 L2, S5 passes |

### Phase 2 — Models
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 2.1 | `models_a.py`: 5 models; Conv1D also takes the side-study arguments (section 7) | A (done by B) | 1.3 | each trains via `train.py --dev`, Conv1D also with `--dropout`, `--l2`, `--layers 2`, `--embedding learned100` | ✅ `python models_a.py` checks parameter counts against the formulas (logreg 60,006, MLP 640,454, Embedding 644,550, Conv1D 681,862, Transformer 677,062); `--dev` runs and saved-model loading pass, Conv1D also with the side-study options. Full-data check (seed 1), test accuracy / macro-F1: MLP 88.9 % / 0.837, Embedding 86.3 % / 0.805, Conv1D 91.7 % / 0.871, Transformer without mask 90.0 % / 0.851 vs with mask 86.5 % / 0.808 (seed 2: 88.0 %). So the Transformer runs without the mask (`config.NO_MASK_MODELS`, section 4; `train.py --mask` turns it on, and the variant name shows "mask" only when it differs from the model's default). Review 2: its T0 setting becomes the feed-forward size (step 1.10) |
| 2.2 | `models_b.py`: 5 models; the attention model can output word weights; GRU takes the side-study arguments, LSTM takes `layers` | B | 1.3 | each trains via `train.py --dev`, GRU also with the side-study options | ✅ all 5 take every side-study argument; `python models_b.py` checks parameter counts against the lecture 8 formulas; full-data check (seed 1): GRU 92.4 % / 0.881, BiLSTM + attention 92.2 % / 0.874; `attention_weights()` works on the saved model (padding gets weight 0) |
| 2.3 | S9a + S9 optimisers on Conv1D and GRU; best rates → `config.S9_LEARNING_RATES`; what the protocol uses (👤 both) | B | Conv1D from 2.1, GRU from 2.2 | 180 S9a/S9 rows, the optimiser decision recorded | ✅ S9a (90 runs, 70 min): best rate per optimiser by mean validation macro-F1 of Conv1D and GRU: SGD 1.0, Momentum 0.1, Nesterov 0.1, Adagrad 0.1, Adadelta 1.0, RMSprop 0.01, Adam 0.001, Nadam 0.01, AdamW 0.001. S9 (90 runs, 28 min), validation macro-F1 over 10 runs: Nesterov 0.902 ± 0.009, Momentum 0.900, Adam 0.898 ± 0.008, AdamW 0.898, RMSprop 0.898, Nadam 0.898, Adagrad 0.896, Adadelta 0.896, SGD 0.616 ± 0.377; Conv1D best with Adam/AdamW (0.904), GRU with Nesterov (0.909 vs Adam 0.893), but on test GRU Nesterov − Adam = −0.0025. **Decided 2026-10-31: the optimiser is tuned per model in T0b** (Adam vs SGD + Nesterov, selection set, tie rule; section 6). Fixed on the way: `best_epoch` from `EarlyStopping` (np.argmin pointed at the first NaN epoch) |
| 2.4 | T0 tuning: T0a sizes (56 runs), then T0b optimiser + learning rate (120 runs), chosen on the selection set with the tie and edge rules; B runs both for all models; each owner picks for own models → `config.MODEL_SIZES`, `MODEL_OPTIMIZERS`, `MODEL_LEARNING_RATES` | run B, pick each | 2.1, 2.2, 2.3, 1.9, 1.10, 1.11 | T0a and T0b rows in CSV, the three dicts filled for all 10 models, tuning-budget table | ✅ **T0a done** (64 runs, 38 min; first pass rerun after the step 1.9 fix). Mean selection-set macro-F1 of seeds 1–2, size proposed by the tie rule: logreg 0.822 (its only setting; both runs hit the 30-epoch cap); MLP 64 (0.854; 32 only +0.003); Embedding 64 (0.817; 128 +0.0049, just under the margin); Conv1D 128 (0.889; 256 +0.002); Transformer 128 feed-forward units (0.865; 256 +0.001); SimpleRNN 64 (0.721; unstable: 0.650 ± 0.089 at 32, 0.597 at 128); LSTM 128 (0.871; 256 +0.001); GRU 128 (0.886; 256 +0.002); BiLSTM 256 (0.870; edge rule: 128 → 256 +0.006, 512 worse, 0.867); BiLSTM + attention 64 (0.886). Sizes confirmed 2026-10-31, `config.MODEL_SIZES` filled. **T0b done** (130 runs, 105 min: 120 + 10 for the edge rule). Mean selection-set macro-F1 of seeds 1–2 against Adam 0.001 (T0a), pair proposed by the tie rule: logreg Nesterov 0.1 (0.853, +0.031; both runs hit the 30-epoch cap, Nesterov 0.3 stops before it and is 0.002 lower); MLP Adam 0.001 (Adam 0.0003 only +0.001); Embedding Adam 0.001 (Adam 0.003 +0.0046, just under the margin); Conv1D Adam 0.001 (Adam 0.0003 +0.001); Transformer Adam 0.001 (best itself); SimpleRNN Adam 0.0003 (0.743, +0.023; edge step 0.0001 worse, 0.690); LSTM Adam 0.01 (0.893, +0.022; 0.03 worse, 0.878); GRU Adam 0.01 (0.898, +0.012; 0.03 worse, 0.880; best epoch 1 in both seeds); BiLSTM Adam 0.01 (0.885, +0.015; 0.03 worse, 0.865); BiLSTM + attention Nesterov 0.3 (0.897, +0.011; 1.0 a little worse, 0.896). SGD + Nesterov does not train the Transformer at all (0.086–0.089: one class) nor SimpleRNN (0.11–0.38); one diverged run (Conv1D Nesterov 0.3 seed 1, NaN) and one stuck run (LSTM Nesterov 0.03 seed 2, 0.244) are kept as they are. Pairs confirmed 2026-10-31 (B; A to be told about her five: logreg, MLP, Embedding, Conv1D, Transformer), `config.MODEL_OPTIMIZERS` and `MODEL_LEARNING_RATES` filled, S1 and later studies start with them. Tuning budget: T0a 64 runs / 38 min, T0b 130 runs / 105 min; the budget table itself comes from results.csv in step 4.1 |

### Phase 3 — Experiments
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 3.1 | S1 main comparison (saves predictions incl. the extended test, keyword-test copies and LT test; models with their vocabulary) | B | 2.4, 1.9, 1.11 | 50 S1 rows, predictions committed | ✅ 50 runs, 23 min of training (2026-10-31). Mean macro-F1 over 5 seeds, test / extended test: GRU 0.885 / 0.903, BiLSTM + attention 0.890 / 0.902, LSTM 0.883 / 0.900, Conv1D 0.878 / 0.896, BiLSTM 0.879 / 0.895, Transformer 0.856 / 0.872, MLP 0.840 / 0.860, logreg 0.832 / 0.857, Embedding 0.812 / 0.827, SimpleRNN 0.720 / 0.732 (unstable: seed 5 0.618); keyword rule 0.672 on test. The spread over seeds on the extended test is about half of that on the 2,000-text test (e.g. GRU 0.003 vs 0.005 standard deviation), so the extended test separates models better. Every model scores 1.5–2 points higher on the extended test than on the test (to explain in 4.1). GRU's best epoch is 1 in all 5 seeds (Adam 0.01), logreg hits the 30-epoch cap in all 5. Hardest classes: surprise and love. Predictions (test, extended test, GoEmotions, LT test, 4 keyword copies) committed; the 50 models (383 MB) stay local in `results/models/` |
| 3.2 | S2 masking (61 runs, incl. the Transformer arms) and S3 class weights + focal loss (α = 1) | B | 3.1 | rows in CSV | ✅ **S2** (61 runs) and **S3** (60 runs) done, no failed runs (2026-10-31). **S2**, mean test macro-F1 over 5 seeds, post padding without mask / pre padding without mask (S1 = post padding with mask): SimpleRNN 0.427 / 0.700 (S1 0.720); LSTM 0.706 / 0.878 (S1 0.883; post: seed 4 collapsed to one class, 0.086, the other four 0.855–0.873 and 9–12 epochs instead of 5); GRU 0.884 / 0.884 (S1 0.885); BiLSTM 0.876 / 0.881 (S1 0.879). Padding after the text without the mask hurts the gateless SimpleRNN badly and LSTM in one seed; GRU and BiLSTM do not notice it; padding before the text works about as well as the mask. Accuracy by length (test, mean of 5 seeds, ≤ 10 / 11–20 / 21–35 / > 35 words, 501 / 729 / 590 / 180 texts): SimpleRNN S1 0.905 / 0.841 / 0.773 / 0.656, post without mask 0.756 / 0.596 / 0.510 / 0.421, so it drops at every length and the guess in section 6 (short texts hurt most) does not hold; a likely reason is that with post padding the last state is always 50 steps after the start, whatever the text's length. GRU without mask is flat (0.929 / 0.926 / 0.933 / 0.921). **Transformer with the mask**, Adam rate search (selection set, seeds 1–2): 0.0003 0.804, 0.001 0.829, 0.003 0.861, edge step 0.01 0.853 → 0.003 (confirmed 2026-10-31, `config.S2_MASKED_TRANSFORMER_LR`). At 0.003, 5 seeds: selection set 0.862, test 0.849, extended test 0.866 (best epoch 1 in all 5); without the mask (S1, Adam 0.001): 0.866, 0.856, 0.872. Masked − unmasked on the extended test (paired bootstrap, 1,000 draws, preliminary, no Holm): −0.006, 95 % interval −0.007 … −0.004, so practically equal by the ±0.01 rule. Most of the step-2.1 gap was the rate: the masked model needs a 3 times larger one. 2 × 2 at Adam 0.001 (selection set, 5 seeds): attention mask only 0.845, pooling mask only 0.848, both 0.829 (seeds 1–2), neither 0.866; these arms run at the unmasked model's rate. **S3**, mean test macro-F1, S1 → class weights: SimpleRNN 0.720 → 0.772 (and steadier: standard deviation 0.058 → 0.019), logreg 0.832 → 0.845, MLP 0.840 → 0.853, Embedding 0.812 → 0.823; the other six change by at most 0.005 (GRU 0.885 → 0.882, BiLSTM + attention 0.890 → 0.889, Conv1D 0.878 → 0.879). Mean F1 over the 10 models: love 0.775 → 0.797, surprise 0.689 → 0.723, sadness, joy and fear a little lower; test accuracy of the strong models drops (GRU 0.930 → 0.921), since rare classes are predicted more often. Focal loss: Conv1D 0.879 (S1 0.878), GRU 0.877 (S1 0.885; surprise F1 0.726 → 0.707). Significance (bootstrap, Holm) comes in step 4.1 |
| 3.3 | S4 training-set size (+ keyword-rule curve), S5 binary (collapsed S1 predictions + 15 runs) | A | 3.1 (S4 and S5 need the best models of S1) | rows in CSV |  |
| 3.4 | S10 regularisation (L2 on all weights), S11 depth | B | 2.4 | rows in CSV |  |
| 3.5 | S12 GloVe: `download_glove.py`, `data.glove_matrix` (also with the extended vocabulary), GloVe coverage, then `run_study.py S12` | A | 2.4 | 40 S12 rows, predictions committed |  |
| 3.6 | S14 competitor: `competitor.py` (KerasHub DistilBERT, same results row and predictions as `train.py`; DistilBERT's own [UNK] in the keyword copies), tested on a small subset first; the 3 full runs on B's machine | A writes, B runs | 1.6, 1.7, 1.9 | 3 S14 rows, predictions committed |  |
| 3.7 | S15 Lithuanian: all 10 models on the Lithuanian train split with their English settings | A | 2.4, 1.9 | 50 S15 rows, predictions committed, best LT model saved for the app |  |
| 3.8 | S16 context-only reference (Conv1D, GRU, BiLSTM + attention without the emotion word) | A | 2.4, 1.10 | 15 S16 rows |  |
| 3.9 | S17 cross-corpus: 4 models trained on GoEmotions, tested both ways | A | 2.4 | 20 S17 rows |  |
| 3.10 | Optional S14b: multilingual DistilBERT, English training, Lithuanian test | A writes, B runs | 3.6 | 1 row |  |

### Phase 4 — Analysis outputs
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 4.1 | `compare.py`: S9 (heatmap optimiser × rate + table), T0 (choices + budget table), S1 with S14 and baselines: bootstrap intervals, Holm, groups that cannot be told apart, keyword-group columns, McNemar (b, c, seeds significant); S8 (with and without the embedding); S2 (incl. accuracy by length), S3, S10 (gap), S11; summary table for 4.7. Tables in report format (Lithuanian headers, decimal comma), figures with decimal comma and Lithuanian model names | B | 2.3, 3.1, 3.2, 3.4, 3.6 | files in `tables/`, `figures/` |  |
| 4.2 | `compare.py` S12, S4, S5 functions (appendix tables + rows for the 4.7 summary); `generalization.py` S6 (per class, both directions), S13 (groups, copies, [UNK] control, negation templates, erasure check), S15, S16, S17 | A | 3.1, 3.3, 3.5, 3.6, 3.7–3.9 | files in `tables/`, `figures/` |  |
| 4.3 | S7: confusion matrices, most-confused pairs, the error audit (texts ≥ 80 % of S1 runs get wrong; both label the same random 100; categories and Cohen's κ), attention heatmaps | B (labelling: both) | 3.1 | figures + audit table |  |
| 4.4 | `predict.py` and `app.py` (Gradio: probabilities, words shaded by attention, unknown-word warning, model choice, English and Lithuanian models with their saved vocabulary); a test table of 6–8 English and Lithuanian inputs; screenshots and a 20–30 s screen recording for 4.11 and the talk | A | 3.1, 3.7 | works on any sentence, LT and EN |  |
| 4.5 | `timing.py` (S8): inference time of the saved S1 models and DistilBERT, warm-up, median of 5, fixed batch and threads | B | 3.1, 3.6 | `tables/timing.csv` |  |

### Phase 5 — Writing (details in section 10)
| # | Step | Owner | Needs | Status |
|---|---|---|---|---|
| 5.1 | Literature review sections; **systematic search** (one runs it, both screen): Web of Science Core Collection, Advanced search, query in section 10 (2.4), years 2019–2026, document types article / proceedings paper / review, English; inclusion criteria; counts found → refined → screened → included, PRISMA-style figure (Page et al., 2021), exact query and date in an appendix; plus Google Scholar "cited by" of Saravia et al. (2018) for results on the same dataset; recent surveys (2021–2025) | each own; search one, screen both | sources (start now) |  |
| 5.2 | Model theory sections | each own | – (start now) |  |
| 5.3 | Methodology: data, protocol, metrics, training | each own | 2.4 (final architectures) |  |
| 5.4 | Results sections | each own | phase 4 |  |
| 5.5 | Shared sections: main comparison, summaries | drafter, then the other adds | 4.1 |  |
| 5.6 | Product structure (3.2, ~1,200 characters: purpose and users, 3–4 requirements, the tool's architecture, why these models); comparable services table (3.6: NRC EmoLex, Hugging Face emotion models, cloud NLP APIs, our tool × emotion set, Lithuanian support, explanation, offline/privacy, licence/cost, reported accuracy, pros/cons; vendor claims cited with access dates) | A | 2.1 |  |
| 5.7 | Intro and conclusions (written last): the 8 tasks map 1:1 to conclusions; contribution paragraph (~700 characters) written from the actual record (git log, step owners), each of us able to defend what is credited to them; the detailed split (Table 1) in an appendix | A drafts intro, B drafts conclusions, both edit | 5.1–5.6 |  |

### Phase 6 — Assembly and checks
| # | Step | Owner | Done when | Status |
|---|---|---|---|---|
| 6.0 | Export dry run now: one formula, one table, one figure, one caption → .docx → open in Word; decide the citation style (IEEE: Zotero for Google Docs or renumber at the end) | B | works in Word |  |
| 6.1 | Cross-review: A reviews B's sections, B reviews A's (checklist in section 10) | both | all comments resolved |  |
| 6.2 | Number check: a script pulls every number from the exported .docx and looks it up in `results/tables/*.csv`; only the misses are checked by hand | B | no mismatches |  |
| 6.3 | Formatting: title page, Word captions and cross-references for figures, tables and numbered formulas, table of contents regenerated after export, page numbers, reference list | A | doc is consistent |  |
| 6.4 | Export .docx; count the body (prose from 1. Įvadas to 5. Bendros išvados, without tables, captions and formulas) with the counting script, and the physical pages; trim or extend to fit 22–26 | B | inside the limits |  |
| 6.5 | Freeze the results: tag the commit the report uses (hash in appendix 8); give the lecturer access to the repo (public or read access); read it through together and upload to Moodle | both | uploaded before the deadline |  |

### Phase 7 — Talk (section 11)
| # | Step | Owner | Status |
|---|---|---|---|
| 7.1 | Slides: 13 (section 11), each makes own part, one shared template; demo as a recording with a screenshot fallback | both |  |
| 7.2 | Defence notes: the likely questions in section 11 per person, 4–5 backup slides; rehearse to 8–9 min | both |  |
| 7.3 | Upload slides (PPTX or PDF) one day before | A |  |

## 10. Writing the report

### How long

- "22–26 pages" at 1800 characters per page means **39,600–46,800 characters with spaces** of body text. **Target ~45,200; hard caps: chapter 3 ≤ 18,000, total ≤ 46,800, each section within ±10 % of its budget.**
- **What counts as body text:** the prose from 1. Įvadas to 5. Bendros išvados, without tables, captions and formulas; measured on the exported .docx with a short counting script (step 6.4), which also reports the physical page count.
- **Physical pages:** with one architecture table (not ten), two-panel figures (class + length; best + worst confusion matrix) and ~20 figures and tables in total, about 25–27 pages from the title page to the references; the appendices are extra. Check it with a layout dry run (placeholder figures and tables) before writing chapter 4.
- **Check both numbers at step 6.4.**

### Outline with owner and size

Character counts are targets with spaces. "Drafter" writes the first version; the other person adds their own part and edits. Chapters are numbered exactly as in the lecturer's structure (1. Įvadas … 5. Išvados), whose grading weights are given per numbered chapter; page ranges come from the same table.

| Section | Owner | Characters | Content |
|---|---|---|---|
| **1. ĮVADAS** (2 p., 5 %) | drafter A, both | 3,400 | relevance (~800), problem, object, aim, tasks, methods, structure, contribution of each member (~700; the detailed Table 1 in an appendix) |
| **2. LITERATŪROS APŽVALGA** (3–5 p., 20 %) | | **7,600** | |
| 2.1 Emocijų atpažinimas iš teksto | B | 1,500 | emotion models (Ekman, Plutchik), applications, difficulties (sarcasm, short texts, negation), recent surveys (2021–2025) |
| 2.2 Teksto vaizdavimo metodai | A | 1,500 | one-hot, TF-IDF, word2vec, GloVe, learned embeddings |
| 2.3 Neuroniniai tinklai tekstams klasifikuoti | A + B | 1,600 | where the architectures came from and what the literature found about them for text; no formulas (those are in 3.5) |
| 2.4 Panašių tyrimų apžvalga | A + B | 2,400 + table + PRISMA-style figure | **systematic search** (lecturer's recommendation, step 5.1): Web of Science Core Collection, Advanced search, `TS=("emotion recognition" OR "emotion detection" OR "emotion classification") AND TS=(text* OR tweet* OR "social media") AND TS=("neural network*" OR "deep learning" OR LSTM OR GRU OR CNN OR convolutional OR transformer* OR BERT) AND PY=(2019-2026)`, refined to article / proceedings paper / review in English; inclusion: text only, several emotion classes, at least two neural architectures compared, accuracy or F1 on a named dataset; counts found → refined → screened → included; table: author, data, models, result (incl. results on dair-ai/emotion from Google Scholar "cited by" of Saravia et al.) |
| 2.5 Apžvalgos apibendrinimas | drafter A | 600 | what is missing, what this work does |
| **3. PRAKTINĖS DALIES ATLIKIMO METODIKA** (9–10 p., 30–35 %) | | **17,600** | opens with one sentence mapping each item of the lecturer's structure to its subsection |
| 3.1 Tyrimo eiga, protokolas ir programos algoritmas | B | 1,700 | pipeline diagram; the algorithm of one run (`train.py`) and of the study loop (`run_study.py`) as a flowchart or numbered steps; fixed settings, seeds, one change at a time within a study; the evaluation sets (selection set, extended test) and why; why hold-out validation, batch 64, stopping on val_loss vs choosing on macro-F1 (section 6) |
| 3.2 Kuriamo produkto struktūra | A | 1,200 + diagram | purpose and users, 3–4 requirements, the tool's architecture (input → cleaning → vectoriser → model → probabilities + attention + unknown-word share), English and Lithuanian models, why these models (accuracy vs interpretability, from S1) |
| 3.3 Duomenų rinkiniai ir jų kokybė | A | 1,700 | dair-ai/emotion, GoEmotions, the Lithuanian translation (train, validation, test), the selection set and extended test; which quality checks are done (the numbers are in 4.1) |
| 3.4 Teksto paruošimas ir kodavimas | A | 1,100 | tokenisation, sequence length 50 (coverage + IQR rule), padding, multi-hot, embedding, GloVe |
| 3.5.1–3.5.5 A's models | A | 5 × 550 | logreg, MLP, Embedding, Conv1D, Transformer: principle and formula; their layers are in the 3.5.11 table |
| 3.5.6–3.5.10 B's models | B | 5 × 550 | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention: principle and formula; their layers are in the 3.5.11 table |
| 3.5.11 Modelių suvestinė | drafter B | 600 + one table | one table for all 10 models: layers, output shapes, parameters with and without the embedding, pros and cons |
| 3.6 Palyginamosios paslaugos ir konkurentai | A | 1,900 + table | baselines (majority, keyword rule); DistilBERT (published numbers and our own fine-tuning, S14); **comparable services** table (NRC EmoLex, Hugging Face emotion models, cloud NLP APIs, our tool × emotion set, Lithuanian support, explanation, offline/privacy, licence/cost, reported accuracy, pros/cons; vendor claims cited with access dates) |
| 3.7 Mokymo parametrai, optimizatoriai ir reguliarizacija | B | 1,800 | loss (cross-entropy, focal with α = 1), class-weight formula, update rules of the 9 optimisers (compact table), Dropout and L2 formulas, early stopping, activations; the tuning procedure, the same for every model (T0a size, T0b optimiser × rate, selection set, tie rule) and its budget |
| 3.8 Vertinimo kriterijai ir statistinis palyginimas | B | 1,800 | accuracy, precision, recall, F-measure, macro-F1, paired bootstrap confidence interval, McNemar with Holm correction, all with formulas; the wording rule; keyword groups; cost measures (parameters with and without the embedding, s/epoch, ms per 1,000 texts); generalisation drop; train − validation gap |
| 3.9 Naudotos priemonės | A | 300 | Python, TensorFlow/Keras, KerasHub, Gradio, git, hardware |
| **4. PRAKTINĖ DALIS** (6–9 p., 35–40 %) | | **14,200** + figures | core studies get their own sections and depth; side studies share 4.7, full results in the appendices |
| 4.1 Duomenų analizė | A | 1,300 | class and length plots (one two-panel figure), the keyword finding with the purity groups, label noise (duplicates with different labels, two-label texts), baselines |
| 4.2 Optimizatoriaus ir mokymosi greičio parinkimas | B | 800 | S9a heatmap (optimiser × rate), S9 table; why two candidates go to T0b; the validation advantage that did not hold on test |
| 4.3 Hiperparametrų parinkimas | drafter B | 900 + table | chosen size, optimiser and rate per model, the tuning budget; full T0 tables in an appendix |
| 4.4 Modelių palyginimas | drafter B, A adds | 2,200 | main table (mean ± sd on the official test, the extended test, 95 % intervals vs the reference model, groups that cannot be told apart, optimiser and rate per model) with DistilBERT and the baselines; keyword-group columns; McNemar; quality vs cost (S8) |
| 4.5 Užpildymo ir užpildymo kaukės įtaka | B | 1,000 | S2 table, explanation via hidden state and the accuracy-by-length plot; the Transformer as the opposite case (2 × 2 mask check) |
| 4.6 Nesubalansuotų klasių įtaka | B | 900 | S3: per-class F1 without / with class weights / with focal loss |
| 4.7 Kitų mokymo pasirinkimų įtaka | B (S10, S11) + A (S12, S4, S5) | 1,200 + summary table | one table: each side study, its best setting and the change vs S1 (with intervals); 2–3 sentences on GloVe, training-set size (with the keyword-rule curve) and binary vs 6 classes, one each on regularisation and depth; figures and full tables in the appendices |
| 4.8 Raktiniai žodžiai ar kontekstas? | A | 1,800 | S13 per keyword group; the drop without the keyword vs the context-only model (S16), the keyword rule and DistilBERT; the [UNK] control; negation templates; attention share on the keyword and the erasure check |
| 4.9 Gebėjimas apibendrinti: kiti duomenys ir lietuvių kalba | A | 1,500 | GoEmotions both ways (S6, S17; per class, love as majority); models trained in Lithuanian vs English on identical texts (S15); English models on Lithuanian in one sentence; optional zero-shot (S14b) |
| 4.10 Klaidų analizė ir dėmesio vizualizacija | B | 1,300 | confusion matrices (joy ↔ love), the error audit (categories, our agreement), heatmaps |
| 4.11 Emocijų atpažinimo įrankis | A | 1,300 | implementation, two screenshots, a test table of 6–8 English and Lithuanian inputs with outputs, inference time, limitations (machine-translated training data, unknown words) |
| **5. BENDROS IŠVADOS** (1–2 p., 5 %) | drafter B, both | 2,400 | one numbered conclusion per task (8), each with a number |
| LITERATŪROS SĄRAŠAS | A formats | – | ~45 entries at most (1–2 pages) |
| PRIEDAI (not counted in the 22–26 pages) | both | – | 1: T0a/T0b full tables and the tuning budget (B); 2: S10 regularisation (B); 3: S11 depth (B); 4: S12 GloVe (A); 5: S4 learning curve (A); 6: S5 binary (A); 7: per-class F1 of every model in S1 (B); 8: link to the repo and the tagged commit; 9: the systematic search (query, date, counts); 10: detailed contribution table (Table 1) |
| **Total** | | **~45,200 (≈ 25.1 pages)** | A ≈ 23,600, B ≈ 21,600 (shared sections counted half each; B drafts more of the shared ones) |

Review 1 (S10–S14, focal loss) and review 2 (statistics, keyword groups, S15–S17, the product, the comparable services table, the systematic search) were paid for by shorter model sections (900 → 700 → 550 each, with one architecture table), a shorter 2.3, 4.2 and 4.7, and the side studies' details in the appendices. The total stays at ~45,200, ~1,600 below the 46,800 maximum.

### Draft aim, tasks and contributions (Lithuanian, final wording at step 5.7)

**Tikslas** – palyginti skirtingų architektūrų neuroninių tinklų modelius, skirtus emocijoms atpažinti iš trumpų tekstų, įvertinti jų tikslumą, skaičiavimo sąnaudas ir gebėjimą apibendrinti bei sukurti tyrime apmokytais modeliais paremtą emocijų atpažinimo įrankį.

**Uždaviniai:**
1. Išanalizuoti mokslinę literatūrą apie emocijų atpažinimą iš teksto ir tam naudojamus neuroninių tinklų modelius. → 2
2. Ištirti duomenų rinkinį ir jo kokybę, nustatyti bazinius tikslumo lygius. → 3.3, 4.1
3. Sukurti dešimt skirtingų architektūrų modelių ir kiekvienam ta pačia tvarka parinkti dydį, optimizatorių ir mokymosi greitį. → 3.5, 3.7, 4.2, 4.3
4. Palyginti modelius tarpusavyje ir su iš anksto apmokytu DistilBERT modeliu pagal kokybės kriterijus, skaičiavimo sąnaudas ir statistinį reikšmingumą. → 4.4
5. Ištirti užpildymo kaukės, nesubalansuotų klasių, reguliarizacijos, tinklo gylio, iš anksto apmokytų žodžių vektorių, mokymo imties dydžio ir klasių skaičiaus įtaką rezultatams. → 4.5–4.7
6. Nustatyti, ar modeliai remiasi emocijos žodžiu, ar kontekstu, ir išanalizuoti jų klaidas. → 4.8, 4.10
7. Įvertinti modelių gebėjimą apibendrinti kitame duomenų rinkinyje (GoEmotions) ir palyginti lietuviškais bei angliškais tekstais apmokytus modelius. → 4.9
8. Sukurti emocijų atpažinimo įrankį ir jį išbandyti. → 3.2, 4.11

**Darbų pasiskirstymas** (the lecturer asks for each member's contribution at the end of the introduction): written at step 5.7 from the actual record, not from the plan's owner columns. Several of A's steps were done by B (1.4, 1.6, 2.1; 1.8 and 1.9 are B's), so the text says who implemented, ran, analysed and wrote each part, and each of us must be able to explain at the defence everything credited to them. About 700 characters in the introduction, ending with "Išsamus pasiskirstymas pateiktas 10 priede"; the detailed table goes to appendix 10. The draft already pasted into the Google Doc must be replaced then.

### Writing rules

- **Every model section uses the same template:** principle → formula or diagram → what is special about it for text; its layers, output shapes, parameter counts, pros and cons are its row in the one table of 3.5.11.
- **Every results section uses the same template:** what we did (1–2 sentences) → table or figure → what we see (with numbers) → why → which task it answers.
- **Depth over breadth:** a core section explains *why* a result happens (hidden state for masking, keyword labels for S13, domain shift for GoEmotions), not only what the numbers are. Side studies in 4.7 get 2–3 sentences each; everything else goes to the appendices and is referred to ("3 priedas").
- **Differences are claimed only when they hold** (section 6, statistics): "better" ("geresnis") when the 95 % interval of the paired difference excludes 0 and survives Holm; "practically equal" ("praktiškai lygūs") when the whole interval lies within ±1 macro-F1 point; otherwise "no clear difference" ("aiškaus skirtumo nėra"). Rankings are given as groups.
- **Tasks map one-to-one** to sections and to conclusions. Each conclusion contains at least one number.
- **Figures and tables** are numbered, captioned in Lithuanian ("1 pav. …", "1 lentelė. …") and referred to in the text. Sources are given for anything not ours. Figures use Lithuanian model names and a decimal comma on the axes, like the text.
- **Formulas** are numbered "(1)", "(2)" …, with the symbols explained after "čia", using the lecture's symbols (η, h_t, C_t …).
- **Numbers come only from the generated tables** (`compare.py` writes them in report format). Use a decimal comma in Lithuanian text (0,881; 92,4 %).
- **Lithuanian wording:** "tekstams klasifikuoti", not "teksto klasifikavimui" (purpose with a verbal noun in the dative); "transformerio koduotuvas"; "su DistilBERT modeliu"; the lecturer's chapter titles word for word. In the Google Doc fix "Uždaviniai tiklsui" → "tikslui" and "Tikrino: Doc. Dr." → "doc. dr.", and use the 3.4 title from this plan.
- **Every claim from literature is cited.** Every source is checked in Google Scholar before it goes in. Papers for 2.4 come from the systematic Web of Science search (step 5.1).
- **Use the lecturer's terms** (from the course notes):

| English | In the report |
|---|---|
| accuracy | bendras tikslumas |
| precision | tikslumas |
| recall | pilnumas |
| F1 / macro-F1 | F-matas / makro F-matas |
| confusion matrix | maišaties matrica |
| embedding | vektorinis žodžių atvaizdavimas (Embedding) |
| padding / masking | užpildymas / užpildymo kaukė |
| pooling | surinkimo sluoksnis |
| gate | sklendė |
| attention | dėmesio mechanizmas |
| early stopping | ankstyvasis sustabdymas |
| overfitting | persimokymas |
| loss function | nuostolių funkcija |
| training / validation / test set | mokymo / validavimo / testavimo imtis |
| selection set / extended test | parinkimo imtis / išplėstinė testavimo imtis |
| learning rate | mokymosi greitis |
| optimiser | optimizatorius |
| batch size | paketo dydis |
| seed | sėkla |
| hidden state | paslėptoji būsena |
| class weights | klasių svoriai |
| vocabulary | žodynas |
| confidence interval | pasikliautinasis intervalas |
| Transformer | transformeris (transformerio koduotuvas) |

### Cross-review checklist (step 6.1)
- Terms match the table above.
- Every figure and table is referenced in the text and has a Lithuanian caption.
- Numbers match the generated tables.
- Each model section follows the template.
- Every literature claim has a citation.
- Lithuanian diacritics are correct, and no English is left in captions or axes.
- The section stays within ±10 % of its character budget; chapter 3 ≤ 18,000, total ≤ 46,800.
- Every "better" follows the wording rule, with its interval.

## 11. Talk (7–10 min, 13 slides, aim for 8–9 min)

| # | Slide | Presents |
|---|---|---|
| 1 | Title, team | A |
| 2 | Problem, aim and the 8 tasks (the same as in the report) | A |
| 3 | Data and its quality: the keyword finding (purity groups), label noise | A |
| 4 | A's models | A |
| 5 | B's models | B |
| 6 | Protocol: the same tuning procedure for every model (size, optimiser, rate), 5 seeds, 52,000 test texts, confidence intervals | B |
| 7 | Main comparison with DistilBERT and the baselines (groups that cannot be told apart) + quality vs cost as a second panel | B |
| 8 | Training choices: masking (RNNs vs Transformer), class weights vs focal loss, the 4.7 summary | B |
| 9 | Typical errors (the audit) and attention heatmaps | B |
| 10 | **Keywords or context?** Per keyword group; without the word vs the context-only model; negation templates | A |
| 11 | Generalisation: GoEmotions both ways; models trained in Lithuanian vs English on the same texts | A |
| 12 | Demo: the tool, a 20–30 s recording (screenshot as fallback; with a Lithuanian input) | A |
| 13 | Conclusions | B |

Three speaker changes (A → B → A → B). Each person must also be able to answer questions about the other's part at a basic level. 4–5 backup slides for questions.

**Likely questions** (prepare answers at step 7.2):
- Why are all models within 1–2 points of each other? (The labels follow the emotion word; the keyword rule alone gets 73.8 %; the groups show where models differ.)
- Is tuning the optimiser per model fair? Why only Adam and Nesterov?
- Why is the Transformer better without the mask?
- Why a bootstrap and McNemar instead of a t-test over the 5 seeds?
- Doesn't replacing a word with [UNK] hurt by itself? (The [UNK] control.)
- Are attention weights really an explanation? (The erasure check.)
- Why not translate Lithuanian into English first, or use a multilingual model? (S15, S14b.)
- Why count logistic regression as a neural network? (A network without a hidden layer.)
- What about the test texts that also appear in train? (Different labels: label noise.)
- Explain the other member's model: LSTM gates, attention, the Conv1D parameter count.

## 12. Starter literature (check each in Google Scholar before citing)

**Emotions and data**
- Ekman, P. (1992). An argument for basic emotions. *Cognition and Emotion*, 6(3–4), 169–200.
- Plutchik, R. (2001). The nature of emotions. *American Scientist*, 89(4), 344–350.
- Saravia, E. et al. (2018). CARER: Contextualized affect representations for emotion recognition. *EMNLP 2018*, 3687–3697.
- Demszky, D. et al. (2020). GoEmotions: A dataset of fine-grained emotions. *ACL 2020*.
- Mohammad, S., Turney, P. (2013). Crowdsourcing a word–emotion association lexicon. *Computational Intelligence*, 29(3). (NRC EmoLex; existing tools in 3.6)
- SkyWater21. *lt_twitter_emotions* (Lithuanian machine translation of the CARER corpus), Hugging Face dataset, MIT licence. https://huggingface.co/datasets/SkyWater21/lt_twitter_emotions (cite as a derived dataset, CARER → the Kaggle "emotions" copy → translated with the free Google Translate API, with the access date; matching our texts in it was our own check)
- Kusal, S. et al. (2023). A systematic review of applications of natural language processing and future challenges with special emphasis in text-based emotion detection. *Artificial Intelligence Review*, 56(12), 15129–15215. (recent systematic review)
- Deng, J., Ren, F. (2021). A survey of textual emotion recognition and its challenges. *IEEE Transactions on Affective Computing*. doi:10.1109/TAFFC.2021.3053275
- Al Maruf, A. et al. (2024). Challenges and opportunities of text-based emotion detection: A survey. *IEEE Access*.
- Bostan, L.-A.-M., Klinger, R. (2018). An analysis of annotated corpora for emotion classification in text. *COLING 2018*. (cross-corpus, S17)
- Kapočiūtė-Dzikienė, J., Damaševičius, R., Woźniak, M. (2019). Sentiment analysis of Lithuanian texts using traditional and deep learning approaches. *Computers*, 8(1), 4. (Lithuanian NLP, relevance and 4.9)

**Text representation**
- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space. arXiv:1301.3781.
- Pennington, J., Socher, R., Manning, C. (2014). GloVe: Global vectors for word representation. *EMNLP 2014*.
- Salton, G., Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523. (TF-IDF, 2.2)

**Models**
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP 2014*.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014*.
- Schuster, M., Paliwal, K. (1997). Bidirectional recurrent neural networks. *IEEE Trans. Signal Processing*, 45(11).
- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Yang, Z. et al. (2016). Hierarchical attention networks for document classification. *NAACL 2016*.
- Vaswani, A. et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*.
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Pires, T., Schlinger, E., Garrette, D. (2019). How multilingual is multilingual BERT? *ACL 2019*. (zero-shot transfer, S14b)
- Bengio, Y., Simard, P., Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166. (vanishing gradients: SimpleRNN, S2)
- The DistilBERT and BERT accuracies on this dataset come from Hugging Face model cards: cite them as web pages with access dates.

**Training and evaluation**
- Srivastava, N. et al. (2014). Dropout. *JMLR*, 15. (S10)
- Krogh, A., Hertz, J. (1992). A simple weight decay can improve generalization. *Advances in Neural Information Processing Systems 4*. (L2; S10)
- He, H., Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. (class imbalance; S3)
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*. (focal loss; S3)
- Zeiler, M. (2012). ADADELTA: An adaptive learning rate method. arXiv:1212.5701. (Adadelta; S9)
- Kingma, D., Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML 2013*. (Momentum, Nesterov; S9)
- Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *JMLR*, 12. (Adagrad; S9)
- Tieleman, T., Hinton, G. (2012). Lecture 6.5 – RMSprop. *COURSERA: Neural Networks for Machine Learning* (lecture slides; cite as such). (RMSprop; S9)
- Dozat, T. (2016). Incorporating Nesterov momentum into Adam. *ICLR 2016 Workshop*. (Nadam; S9)
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. (AdamW; S9)
- Zhang, J. et al. (2020). Why are adaptive methods good for attention models? *NeurIPS 2020*. (optimiser per model: the Transformer)
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2).
- Dietterich, T. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. (Holm correction)
- Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. (bootstrap)
- Berg-Kirkpatrick, T., Burkett, D., Klein, D. (2012). An empirical investigation of statistical significance in NLP. *EMNLP-CoNLL 2012*. (paired bootstrap in NLP)
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. *ACL 2018*.
- Reimers, N., Gurevych, I. (2017). Reporting score distributions makes a difference: Performance study of LSTM-networks for sequence tagging. *EMNLP 2017*. (seeds)
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. (κ in the error audit)

**Shortcuts, label artifacts and explanations (S13, S16, the data-quality findings)**
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673.
- Gururangan, S. et al. (2018). Annotation artifacts in natural language inference data. *NAACL 2018*.
- Poliak, A. et al. (2018). Hypothesis only baselines in natural language inference. *\*SEM 2018*. (the context-only reference, S16)
- Ribeiro, M. T., Wu, T., Guestrin, C., Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. *ACL 2020*. (negation templates)
- Jain, S., Wallace, B. C. (2019). Attention is not explanation. *NAACL 2019*.
- Wiegreffe, S., Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP 2019*.

**Systematic review and software**
- Page, M. J. et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ*, 372, n71.
- Abadi, M. et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI 2016*.
- Abid, A. et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.

**Textbooks (course list)**
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. (cite the edition actually read)

**For the final list:** the working list has 56 entries after review 2 (7 overlapping ones removed: the SemEval-2018 task, the two 2020 surveys, Chung 2014, Zhou 2016, Bouthillier 2021, Gardner 2020), about 2 pages; at step 6.3 cut what the text does not cite and add pages and DOIs where they exist (only Deng & Ren has one so far).

## 13. Open decisions (settle at step 0.1)

1. **Section numbering.** ✅ Decided 2026-10-30: follow the lecturer exactly (1. Įvadas, 2. Literatūros apžvalga, 3. Praktinės dalies atlikimo metodika, 4. Praktinė dalis, 5. Bendros išvados); section 10 and the Google Doc use it.
2. **Citation style.** Recommendation: IEEE numeric [1], compact for ~45 sources (Zotero's Google Docs plugin, or renumber at the end). APA (Author, year) also works. Decide at step 6.0.
3. **Talk date.** Recommendation: 11-09 or 11-10. Register it (step 0.4).
4. **Who uploads to Moodle.**
5. **Owners of the new studies** (review 2): S15, S16, S17 and S14b are proposed for A (generalisation and the keyword test are A's), the data, statistics and timing code for B. Confirm with A.
6. **Dates and a results freeze:** a date for each remaining phase and an hour after which no new numbers enter the report; a side study not finished by then gets one sentence in 4.7 and goes to "further work". 👤 both.

## 14. Review 2 (2026-10-31): findings and decisions

Three independent reviews (code, method and statistics, requirements and report) after S9. Every number below was re-checked on our data. **Status: all adopted 2026-10-31 (B; A to be told, items marked (A) touch A's models or studies).** The decisions are now part of sections 1–13 (data: section 3; protocol: 4; studies and statistics: 6; code: 7; steps 1.9–1.11, 2.4, 3.7–3.10, 4.5, 6.0: 9; report, talk, literature: 10–12); this section stays as the record of why. The code fixes are step 1.10.

### 14.1 Errors to fix in any case
1. **S13 "without" copy (critical):** only the first occurrence of the emotion word is replaced, so 25 of the 1,502 keyword test texts keep it ("i feel crappy i eat crappy"), and 43 keywords are not in the vocabulary, so their copy is unchanged; 4.5 % of the copies hide nothing and understate the keyword drop. Fix: replace every occurrence; count a text as a keyword text only if the word is in the run's vocabulary; compute "only" metrics on keyword texts only (the other 498 are all padding).
2. **Focal loss is 0.25 × the formula:** Keras `CategoricalFocalCrossentropy` defaults to alpha = 0.25. With SGD + Nesterov that is a 4× smaller step, confounding S3. Fix: `alpha=1.0`.
3. **L2 covers 2–6 % of the weights:** only the Conv1D / recurrent input kernel and the output layer are penalised, not the Embedding (640k of ~670k parameters) or the recurrent kernel. Fix: also `embeddings_regularizer` and `recurrent_regularizer`; compare S10 arms by train−validation gap and macro-F1, not raw val_loss (it includes the penalty).
4. **`run_study.py` guards:** S1 (and later studies) start even with empty `MODEL_SIZES`; the done-key ignores the learning rate taken from `config`, so a corrected rate would be skipped as "jau atlikta". Fix: S1 refuses to start until sizes, rates and optimisers are filled; the key includes the actual optimiser and rate.
5. **Logreg has no T0b "× 1" run** (T0a skips models without sizes). Fix: T0a runs logreg at its only setting (+2 runs).
6. **Transformer has its own Dropout (0.1, 0.1, 0.3)** while section 4 says "none except the MLP". Fix: document it (standard block, as in the Keras example).
7. **Repeatability:** seed 1 is bit-identical across S9a/S9 for Adam, AdamW, Nadam, Momentum, Nesterov, but not for SGD, Adagrad, Adadelta, RMSprop (sparse embedding updates). Fix: `tf.config.experimental.enable_op_determinism()` next to `set_random_seed`.
8. **PLAN wording:** "35 of 90 still improving" → 35 hit the cap, 31 had their best epoch at 30; "SGD diverged on Conv1D in 4 of 5 seeds" → 1 diverged, 3 got stuck; "CPU not repeatable" → item 7; the Transformer "position vectors carry the text length" reason is wrong (class mean lengths 18.4–20.7 words); tasks 3 and 6 out of date; the per-model optimiser decision still to be carried into sections 2, 4, 6, 10, 11.
9. **Timing (S8):** `predict_ms_1000` is one noisy call (GRU 80–184 ms over 45 runs). Fix: median over seeds plus a separate timing script (warm-up, median of 5).

### 14.2 Strengthen the core
1. **Statistics:** one run's test macro-F1 has a bootstrap sd of 0.010 over texts, more than its seed sd (0.004 for GRU + Adam), so mean ± sd over seeds understates the uncertainty, and McNemar tests accuracy while we rank by macro-F1. Fix: paired bootstrap over test texts (CI and p for each difference, Holm per table); reference model chosen on validation; report groups of models that cannot be told apart; McNemar stays as the second test with a rule declared in advance. 0 runs.
2. **Bigger evaluation sets from CARER:** the translated corpus holds 353,654 more CARER texts that are not in our 20k and have one label (class shares and length match our test). A stratified 50k extra test (no retraining: S1 saves models) narrows the CI of a difference ~5×; a 20k selection set makes T0 choices reliable. Evidence that 2k validation is too small: GRU Nesterov − Adam was +0.017 on validation but −0.0025 on test. Remove near-duplicates of train first.
3. **Keywords vs context:** 56.5 % of test texts have a keyword that is ≥ 95 % pure in train (GRU + Adam 99.3 % there); 93 % of its errors are in texts with an ambiguous keyword (13.9 % of test, 70.0 % correct) or none (24.9 %, 88.6 %). Fix: a keyword-subset column in every core table (free); a context-only reference (models trained with the keyword removed, 10–15 runs); a negation template set ("i feel X" / "i dont feel X", CheckList) instead of negation-subset accuracy; an [UNK] control (random non-keyword word); an "i feel <keyword>" variant; an attention erasure check; DistilBERT uses its own [UNK]/[MASK].
4. **Label noise:** all 11 test texts that also appear in train have a **different** label; 22,295 CARER texts carry two labels. Reframe the "leak" as a label-noise floor; both students label a sample of the texts every model gets wrong (categories, agreement) to explain the ceiling.
5. **T0b with per-model optimiser:** add Adam 0.01 (GRU + Adam peaked there in S9a) and an edge rule; tie-break: Adam 0.001 stays unless another pair is better by more than the noise (on the selection set); flag runs that hit the epoch cap. Describe per-model tuning as a precaution (the validation advantage did not hold on test).
6. **Lithuanian, trained:** all 16,000 train and 2,000 validation texts exist in Lithuanian, so the models can be trained on Lithuanian with the same protocol: a controlled EN vs LT comparison on identical texts and labels (Lithuanian: 29,043 word forms vs 15,212, top-10k coverage 91.0 % vs 98.3 %, validation OOV 11.9 % vs 3.6 %). The English-vocabulary LT test stays as one sentence (its predictions come from false friends such as "bet", "man", "kai"). Optional: multilingual DistilBERT fine-tuned on English, tested on LT (zero-shot, 1 seed). The app can then take Lithuanian.
7. **GoEmotions:** per-class recall and confusion matrix, the right majority baseline (love 26.9 %); label meanings differ ("love" is in 1,228 of its 1,760 love texts). Optional: train on GoEmotions (5,272 texts) and test both ways (15–20 runs).

### 14.3 Side studies: reshape (pays for 14.2)
- S5 (A): collapsing 6-class predictions to valence is free (97.4–98.1 % on S9 runs); train binary models for 3 models only, or drop the training.
- S10, S11: effects expected below the resolution; appendix only, one sentence each.
- S4 (A): add the keyword rule's own learning curve (free).
- S12 (A): GloVe is used only for words seen in train; add an arm with vocabulary = train ∪ top GloVe words, and report GloVe coverage (+10 runs).
- Transformer (A): heads 1/2/4 do not change capacity (677,062 parameters each); tune FFN size 64/128/256 instead. Mask: a 2 × 2 check (attention mask × pooling mask, 10 runs); the S2 masked arm gets its own learning-rate search (6 runs).
- S8: also report parameters without the embedding (Conv1D ~42k, Transformer ~37k, GRU ~25k).

### 14.4 Report, product, talk
- **Literature (20 %):** concrete Web of Science query, inclusion criteria, counts (found → screened → included), PRISMA-style figure; sources from 2021–2025 (none in section 12 is newer than 2020); attention-as-explanation (Jain & Wallace 2019; Wiegreffe & Pinter 2019), vanishing gradients, class imbalance, seeds and significance testing, Lithuanian NLP; 2.3 shorter, 2.4 longer; fix incomplete entries.
- **Product:** save the vocabulary and the app's model (nothing writes `best.keras` yet); unknown-word warning; model choice; 3.2 → ~1,200 and 4.11 → ~1,300 characters with a test table; a comparable-services table in 3.6 (criteria incl. Lithuanian support).
- **Aim and tasks:** 8 tasks mapping 1:1 to conclusions, the tool in the aim; contributions built around the keyword diagnosis, the architecture comparison where it matters, and transfer across corpora and languages.
- **Length:** one architecture table instead of ten, multi-panel figures, hard caps (chapter 3 ≤ 18,000, total ≤ 46,800), a counting script.
- **Getting to the .docx:** tag the results commit; `compare.py` writes tables with decimal commas; figures with decimal commas and Lithuanian model names; export dry run; number-check script; numbered formulas; citation style decided; give the lecturer access to the repo.
- **Contribution paragraph:** write it from the actual record (git log and step owners), so each person can defend what is credited to them.
- **Talk:** 13 slides, three speaker changes, recorded demo, backup slides; likely questions listed in the review.
- **Schedule:** dates for the remaining steps and a results freeze; register the talk date (step 0.4).
