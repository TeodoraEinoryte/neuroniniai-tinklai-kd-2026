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

---

## 1. What we want to find out

A lab compares a few models on one task. The coursework is a **study** with four research questions:

1. **Which architecture recognises emotions in short texts best, and at what cost?** 10 models from 4 families, 5 seeds each, statistical test, parameters and time.
2. **Which text-specific training choices matter?** Optimiser and learning rate (8 optimisers), padding/masking in recurrent networks, class weights for uneven classes, training-set size, 6 vs 2 classes.
3. **Do the models generalise?** Test on a different, human-labelled dataset (GoEmotions) and on Lithuanian sentences.
4. **What do the models actually learn: keywords or context?** Keyword baseline, attention weights, negation, error analysis.

## 2. How this goes beyond lab 1 and lab 2

| | Lab 1 | Lab 2 | Coursework |
|---|---|---|---|
| Task | MNIST | MNIST + IMDB, 2 classes | 6 emotions, uneven classes (3.6 %–33.5 %) |
| Architectures | Dense | 5 (Conv2D, Dense, Conv1D, LSTM, GRU) | 10 in 4 families. New vs lab 2: logistic regression, Transformer, SimpleRNN, BiLSTM, BiLSTM + attention |
| Trained runs | ~30, 1 each | ~21–24, 1 each | ~480: tuning on validation, then 5 seeds → mean ± std, McNemar test |
| Metrics | accuracy, loss | accuracy, loss, time | per-class precision/recall/F1, macro-F1, confusion matrix, parameters, train and inference time |
| Context | – | – | baselines (majority 34.8 %, keyword rule 72.4 %) and competitor (fine-tuned DistilBERT) |
| Data | ready | ready | own data-quality analysis (keyword labels, conflicts, leakage) |
| Generalisation | – | – | GoEmotions (6,535 texts) and Lithuanian sentences |
| Extra studies | hyperparameters | hyperparameters | 8 optimisers × 4 learning rates on text models, masking, class weights, training-set size, binary vs 6 classes |
| Interpretation | – | – | attention heatmaps, error examples, negation subset |
| Product | – | – | `predict.py`: sentence in → emotion probabilities out |

## 3. Data

### dair-ai/emotion (main)
- Saravia et al., CARER, EMNLP 2018, DOI 10.18653/v1/D18-1404. English tweets, labels 0–5: sadness, joy, love, anger, fear, surprise → **liūdesys, džiaugsmas, meilė, pyktis, baimė, nuostaba**.
- Official split: train 16,000 / validation 2,000 / test 2,000.
- Classes (train): joy 33.5 %, sadness 29.2 %, anger 13.5 %, fear 12.1 %, love 8.2 %, surprise 3.6 %.
- Texts: lowercase, no punctuation, 19.2 words on average (median 17, 95th percentile 41, 99th percentile 52, max 66). 15,212 distinct words; the top 10,000 cover 98.3 % of word occurrences.

**Quality issues (these go in the report, not under the rug):**
- 97.3 % of texts contain *feel/feeling/felt*. The labels were assigned from the emotion word, so "i didnt feel humiliated" is *sadness*.
- A one-line rule ("word after *feel*" → most common label in train) gets **72.4 %** on test.
- 51 texts appear more than once with **different** labels; 11 test texts also appear in train (5 validation texts likewise).
- **Decision:** keep the official split unchanged so our numbers are comparable with published results; report the issues and their size.

### GoEmotions (generalisation test only, never trained on)
- Demszky et al., ACL 2020. Reddit comments, human-labelled, 28 labels.
- We take the single-label texts whose label is one of our 6 emotions: **6,535 texts** (love 1,760, anger 1,265, joy 1,052, sadness 1,003, surprise 902, fear 553).
- Preprocess like our data (lowercase, strip punctuation).

### Lithuanian sentences (generalisation, fun part)
- **Each of us writes 30 short sentences (5 per emotion)** plus our own English translation, 60 in total.
- The trained models classify both versions. Expected result: the LT originals fail, the EN translations work. The model knows only its training language.

### Binary version (study S5)
Positive = joy, love, surprise; negative = sadness, anger, fear (45.2 % / 54.8 % in train). Surprise is debatable; say so in the report.

## 4. Fixed protocol (`config.py`), the same for every run

| Setting | Value |
|---|---|
| Tokenisation | `TextVectorization`, vocabulary 10,000, adapted on **train only** |
| Sequence length | 50 tokens (98.7 % of train texts fit whole; `length_coverage.csv`) |
| Embedding size | 64 |
| Masking | `Embedding(mask_zero=True)` for sequence models (except where S2 changes it) |
| Optimiser | Adam, learning rate 0.001 (`config.OPTIMIZER`). One optimiser for every run; S9 checks this choice before T0 and may replace it (step 2.3) |
| Loss | sparse categorical cross-entropy, softmax output |
| Batch | 64 |
| Epochs | up to 30, `EarlyStopping(val_loss, patience=3, restore_best_weights=True)` |
| Seeds | 1, 2, 3, 4, 5 (`keras.utils.set_random_seed`) |
| Model choice | by **validation** macro-F1; the test set is used only for final numbers |

**Machines:** every study runs entirely on one machine. S1 is where training times are compared, so it runs on B's machine.

## 5. Models (10) and comparison points

| # | Model | Family | Owner | Tuned (3 values) |
|---|---|---|---|---|
| 0 | Logistic regression (multi-hot → softmax) | feedforward | A | – (baseline) |
| 1 | MLP (multi-hot → Dense → Dropout → softmax) | feedforward | A | hidden units 32/64/128 |
| 2 | Embedding + GlobalAveragePooling + Dense | feedforward | A | embedding size 32/64/128 |
| 3 | Conv1D + GlobalMaxPooling | convolutional | A | filters 64/128/256 |
| 4 | Transformer encoder (word + position embedding, MultiHeadAttention, FFN, LayerNorm) | attention | A | heads 1/2/4 |
| 5 | SimpleRNN | recurrent | B | units 32/64/128 |
| 6 | LSTM | recurrent | B | units 32/64/128 |
| 7 | GRU | recurrent | B | units 32/64/128 |
| 8 | BiLSTM | recurrent | B | units 32/64/128 |
| 9 | BiLSTM + attention pooling (one weight per word → heatmap) | recurrent + attention | B | units 32/64/128 |

Not models we train, but points of comparison:
- **Majority class** (always "joy"): 34.8 % on test.
- **Keyword rule** (word after *feel* → label): 72.4 % on test.
- **Competitor:** published fine-tuned transformers on this same dataset. DistilBERT reports 93.8 % accuracy (93.79 F1) and, elsewhere on the same model page, 92.7 % / macro-F1 0.883. BERT-base reports 94.05 %. They have ~66M+ parameters vs our ~0.7M.

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
- **Time:** all models together train in ~3 min on CPU.

## 6. Experiments

| ID | What changes (everything else fixed) | Runs | Runs on | Owner | Report |
|---|---|---|---|---|---|
| S9a | Optimiser grid: 8 optimisers × learning rates 0.1 / 0.01 / 0.001 / 0.0001, for Conv1D and GRU, seed 1, pick each optimiser's best rate by validation macro-F1 | 8 × 4 × 2 = 64 | B's machine | B | 3.6 |
| S9 | Optimisers: each at its best learning rate, Conv1D and GRU, 5 seeds | 8 × 2 × 5 = 80 | B's machine | B | 3.6 |
| T0 | Tuning: one capacity setting × 3 values per model, seeds 1–2, pick by validation macro-F1 | 9 × 3 × 2 = 54 | B's machine | each picks for own models | 3.2 |
| **S1** | **Main comparison:** 10 tuned models. Also saves predictions for test, GoEmotions and LT/EN sentences | 10 × 5 = 50 | B's machine | both | 3.3 |
| S2 | Padding/masking: post-padding without mask / pre-padding without mask / with mask, for SimpleRNN, LSTM, GRU, BiLSTM | 4 × 3 × 5 = 60 | B | B | 3.4 |
| S3 | Class weights: none vs inverse class frequency | 10 × 5 (+ S1) | B | B | 3.5 |
| S4 | Training-set size: 1k, 2k, 4k, 8k, 16k (stratified) for logreg + best model of each family | 5 × 5 × 3 = 75 | A | A | 3.7 |
| S5 | 2 classes vs 6 classes | 10 × 5 | A | A | 3.8 |
| S6 | Generalisation: reads the S1 predictions for GoEmotions and LT/EN, no retraining | 0 | – | A | 3.9 |
| S7 | Errors: confusion matrices, most-confused pairs, misclassified examples, negation subset, attention heatmaps, McNemar | 0 | – | B | 3.10 |
| S8 | Cost: parameters, s/epoch, training time, inference time per 1,000 texts → quality vs cost chart | from S1 logs | – | B | 3.3 |

About 480 training runs, roughly 3–6 hours of CPU time in total.

**S9, optimisers (runs first, before T0):**
- **Optimisers (8):** the six from lecture 5 (SGD, SGD + Momentum 0.9, Nesterov, RMSprop, Adam, Nadam), plus Adagrad (made for sparse features such as words) and AdamW (Adam with weight decay, standard for Transformers). Keras defaults otherwise.
- **Why a learning-rate grid:** lab 1 (task 6.3) showed that SGD needs a much larger learning rate than Adam or RMSprop. Comparing all of them at 0.001 would be unfair to SGD, so each optimiser gets its own best rate (S9a) before the 5-seed comparison (S9).
- **Why Conv1D and GRU:** the best non-recurrent and the best recurrent model of the pilot run. Recurrent networks are where plain SGD is expected to struggle (gradients pass through 50 time steps), so the two families may react differently.
- **Sizes:** the middle T0 value (Conv1D 128 filters, GRU 64 units), because T0 runs after S9.
- **Decision rule for the main protocol (👤 both):** the protocol keeps one optimiser for every model, so the main comparison changes only the architecture. After S9, take the optimiser with the highest mean validation macro-F1 over both models. If Adam (0.001) is within one standard deviation of it, keep Adam. Otherwise switch `config.OPTIMIZER` and `config.LEARNING_RATE` before T0, so T0 and all later studies use the new choice. The test set is not used for this decision.

**McNemar test:** best model vs every other model, per seed. Report in how many of the 5 seeds the difference is significant (p < 0.05). Implemented ourselves in numpy: χ² = (|b − c| − 1)² / (b + c); p from the χ² distribution with 1 degree of freedom, using `math.erfc`.

**Not in the base plan (only if both agree later):**
- Pretrained GloVe embeddings vs learned ones.
- Running the real DistilBERT with our metrics.
- A CNN + BiLSTM hybrid.

## 7. Code

```
neuroniniai-tinklai-kd-2026/
  README.md, PLAN.md, requirements.txt, .gitignore
  config.py            shared settings (section 4), class names in English and Lithuanian; sizes from T0 and
                       learning rates from S9a are filled in here
  download_data.py     A: one-off script that created data/*.csv from Hugging Face
  data.py              A: loading, vectorising, pre/post padding, binary version, subsets, GoEmotions, LT
  data_analysis.py     A: data analysis figures + quality checks + baselines (report 3.1)
  models_a.py          A: logreg, MLP, Embedding, Conv1D, Transformer
  models_b.py          B: SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention
  evaluation.py        B: metrics in numpy, confusion matrix, McNemar, plots
  train.py             B: one run → one CSV row + curve + matrix + predictions (+ saved model for S1)
  run_study.py         B: runs a whole study: python run_study.py S1
  compare.py           report tables and figures, one function per study (B: T0, S1, S2, S3, S7, S8; A: S4, S5)
  generalization.py    A: S6 tables (GoEmotions, LT/EN)
  predict.py           A: demo
  data/                emotion_{train,validation,test}.csv, goemotions_{train,validation,test}.csv, lt_sentences.csv
                       (original texts; cleaning and filtering happen in data.py)
  results/             results.csv, curves/, matrices/, predictions/, tables/, figures/,
                       models/best.keras (only the best model is committed; the rest and dev/ are git-ignored)
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
load_lt() -> (x_lt, x_en, y)

# models_a.py / models_b.py: every function returns an UNCOMPILED keras.Model
build_lstm(size=64, mask=True, n_classes=6) -> keras.Model
MODELS_B = {"lstm": (build_lstm, "sequence", [32, 64, 128]), ...}   # function, input type, T0 values

# evaluation.py (B)
metrics(y, p, n_classes) -> dict      # accuracy, macro_f1, precision/recall/f1 per class
confusion_matrix(y, p, n_classes) -> np.array
mcnemar(y, p1, p2) -> (chi2, p_value)
```

```bash
python train.py --model lstm --seed 1 --study S1 [--size 64] [--no-mask] [--padding pre]
                [--class-weight] [--fraction 0.25] [--task 2] [--optimizer sgd --lr 0.1] [--dev]
python run_study.py S2           # all runs of one study (S9a, S9, T0, S1–S5)
python compare.py                # tables and figures for the report
python predict.py "i feel so alone tonight"
```

**`results.csv`, one row per run:**
- run info: `date`, `study`, `model`, `family`, `variant`, `seed`, `machine`, `optimizer`, `lr`
- model and training: `params`, `epochs`, `train_time_s`, `predict_ms_1000`
- validation: `val_acc`, `val_loss`, `val_macro_f1`
- test: `test_acc`, `test_loss`, `test_macro_f1`, `f1_sadness` … `f1_surprise`

Predictions for every text are saved per run in `predictions/` (test, plus GoEmotions and LT/EN in S1), which S6, S7 and McNemar need.

## 8. Who does what (summary)

| | A — Teodora | B — Edgaras |
|---|---|---|
| Models (5 each) | logreg, MLP, Embedding, Conv1D, Transformer | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention |
| Shared code | `data.py`, `data_analysis.py`, `generalization.py`, `predict.py` | `evaluation.py`, `train.py`, `run_study.py`, most of `compare.py` |
| Studies | data analysis, S4, S5, S6, demo | S9 optimisers, running T0 and S1, S2, S3, S7, S8 |
| Report text | ~22,700 characters (section 10) | ~21,700 characters + drafts of 3.2, 3.3, 2.4.10, conclusions (section 10) |
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
| 1.4 | `data_analysis.py`: class and length plots, quality checks, majority and keyword baselines | A | 1.1 | figures and numbers for 3.1 | 🟡 tables done (classes, vocabulary, length, dataset overview, GoEmotions filter); plots, quality checks (feel-words, conflicts, leakage) and baselines still to do |
| 1.5 | LT sentences: 30 each (5 per emotion) + English translations → `lt_sentences.csv` | both | – | 60 rows | 🟡 file format set, 2 example rows of 60 |

### Phase 2 — Models
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 2.1 | `models_a.py`: 5 models | A | 1.3 | each trains via `train.py --dev` |  |
| 2.2 | `models_b.py`: 5 models; the attention model can output word weights | B | 1.3 | each trains via `train.py --dev` |  |
| 2.3 | S9a + S9 optimisers on Conv1D and GRU; best rates → `config.S9_LEARNING_RATES`; main optimiser by the rule in section 6 (👤 both) | B | Conv1D from 2.1, GRU from 2.2 | 144 S9a/S9 rows, `config.OPTIMIZER` confirmed or changed |  |
| 2.4 | T0 tuning: B runs it for all models; each owner picks own sizes → `config.py` | run B, pick each | 2.1, 2.2, 2.3 | T0 rows in CSV, sizes in `config.py` |  |

### Phase 3 — Experiments
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 3.1 | S1 main comparison (saves predictions + best model) | B | 2.4, 1.5 | 50 S1 rows, predictions committed |  |
| 3.2 | S2 masking, S3 class weights | B | 3.1 | rows in CSV |  |
| 3.3 | S4 training-set size, S5 binary | A | 2.4 | rows in CSV |  |

### Phase 4 — Analysis outputs
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 4.1 | `compare.py`: T0, S1, S2, S3, S8 tables and figures; McNemar table | B | 3.1, 3.2 | files in `tables/`, `figures/` |  |
| 4.2 | `compare.py` S4, S5 functions; `generalization.py` S6 | A | 3.1, 3.3 | files in `tables/`, `figures/` |  |
| 4.3 | S7: confusion matrices, most-confused pairs, misclassified examples, negation subset, attention heatmaps | B | 3.1 | figures + example table |  |
| 4.4 | `predict.py` demo + screenshot | A | 3.1 | works on any sentence |  |

### Phase 5 — Writing (details in section 10)
| # | Step | Owner | Needs | Status |
|---|---|---|---|---|
| 5.1 | Literature review sections | each own | sources (start now) |  |
| 5.2 | Model theory sections | each own | – (start now) |  |
| 5.3 | Methodology: data, protocol, metrics, training | each own | 2.4 (final architectures) |  |
| 5.4 | Results sections | each own | phase 4 |  |
| 5.5 | Shared sections: main comparison, summaries | drafter, then the other adds | 4.1 |  |
| 5.6 | Intro and conclusions (written last) | A drafts intro, B drafts conclusions, both edit | 5.1–5.5 |  |

### Phase 6 — Assembly and checks
| # | Step | Owner | Done when | Status |
|---|---|---|---|---|
| 6.1 | Cross-review: A reviews B's sections, B reviews A's (checklist in section 10) | both | all comments resolved |  |
| 6.2 | Number check: every number in the doc matches the generated tables | B | no mismatches |  |
| 6.3 | Formatting: title page, table of contents, figure/table numbering and captions, page numbers, reference list | A | doc is consistent |  |
| 6.4 | Export .docx; count characters and pages; trim or extend to fit 22–26 | B | inside the limits |  |
| 6.5 | Read it through together and upload to Moodle | both | uploaded before the deadline |  |

### Phase 7 — Talk (section 11)
| # | Step | Owner | Status |
|---|---|---|---|
| 7.1 | Slides: each makes own half, one shared template | both |  |
| 7.2 | Defence notes: likely questions per person; rehearse to 8–9 min | both |  |
| 7.3 | Upload slides (PPTX or PDF) one day before | A |  |

## 10. Writing the report

### How long

- "22–26 pages" at 1800 characters per page means **39,600–46,800 characters with spaces** of body text (intro to conclusions). **Target ~44,000.**
- Physically this comes to about 24–26 pages from the title page to the references, once ~15–18 figures and tables are added. The appendices are extra.
- **Check both numbers at step 6.4.**

### Outline with owner and size

Character counts are targets with spaces. "Drafter" writes the first version; the other person adds their own part and edits.

| Section | Owner | Characters | Content |
|---|---|---|---|
| **ĮVADAS** | drafter A, both | 3,600 | relevance, problem, object, aim, tasks, methods, structure, contributions (drafts below) |
| **1. LITERATŪROS APŽVALGA** | | **8,200** | |
| 1.1 Emocijų atpažinimas iš teksto | B | 1,800 | emotion models (Ekman, Plutchik), applications, difficulties (sarcasm, short texts, negation) |
| 1.2 Teksto vaizdavimo metodai | A | 1,800 | one-hot, TF-IDF, word2vec, GloVe, learned embeddings |
| 1.3 Neuroniniai tinklai teksto klasifikavimui | A + B | 3,000 | A: MLP, CNN, Transformer. B: RNN, LSTM, GRU, attention |
| 1.4 Panašių tyrimų apžvalga | A + B | 1,000 + table | table: author, data, models, result; ~5 papers each |
| 1.5 Apžvalgos apibendrinimas | drafter A | 600 | what is missing, what this work does |
| **2. METODIKA** | | **17,400** | |
| 2.1 Tyrimo eiga, protokolas ir programos algoritmas | B | 1,800 | pipeline diagram, fixed settings, seeds, one change at a time |
| 2.2 Duomenų rinkiniai ir jų kokybė | A | 2,200 | emotion, GoEmotions, LT sentences; quality issues |
| 2.3 Teksto paruošimas ir kodavimas | A | 1,200 | tokenisation, padding, multi-hot, embedding |
| 2.4.1–2.4.4 A's models | A | 4 × 900 | logreg + MLP, Embedding, Conv1D, Transformer |
| 2.4.5–2.4.9 B's models | B | 5 × 900 | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention |
| 2.4.10 Modelių suvestinė | drafter B | 400 + table | pros and cons, parameters |
| 2.5 Bazinis lygis ir konkurentai | A | 900 | majority, keyword rule, DistilBERT/BERT |
| 2.6 Mokymo parametrai ir hiperparametrų parinkimas | B | 900 | optimiser (8 compared in S9, how the main one is chosen), loss, early stopping, T0 procedure |
| 2.7 Vertinimo kriterijai ir statistinis palyginimas | B | 1,500 | accuracy, precision, recall, F-measure, macro-F1, McNemar, all with formulas |
| 2.8 Naudotos priemonės | A | 400 | Python, TensorFlow/Keras, git, hardware |
| **3. PRAKTINĖ DALIS** | | **13,900** + figures | |
| 3.1 Duomenų analizė | A | 1,500 | class and length plots, keyword finding, baselines |
| 3.2 Hiperparametrų parinkimas | drafter B | 800 + table | T0 results, chosen sizes |
| 3.3 Modelių palyginimas | drafter B, A adds | 2,200 | main table (mean ± std), bar chart, McNemar, quality vs cost |
| 3.4 Užpildymo ir kaukės įtaka | B | 1,300 | S2 table, explanation via hidden state |
| 3.5 Klasių disbalanso įtaka | B | 1,300 | S3 per-class F1 before/after |
| 3.6 Optimizatoriaus ir mokymosi greičio įtaka | B | 1,200 | S9a heatmap (optimiser × learning rate), S9 table (mean ± std, Conv1D vs GRU), which optimiser the protocol keeps |
| 3.7 Mokymo imties dydžio įtaka | A | 1,100 | S4 learning curve: "is the data enough" |
| 3.8 Binarinė ir daugiaklasė klasifikacija | A | 900 | S5 table |
| 3.9 Apibendrinimas kitiems duomenims | A | 1,500 | GoEmotions drop, LT vs EN |
| 3.10 Klaidų analizė ir dėmesio vizualizacija | B | 1,500 | confusion matrices, examples, negation, heatmaps |
| 3.11 Emocijų atpažinimo įrankis | A | 600 | demo, screenshot |
| **IŠVADOS** | drafter B, both | 2,500 | one numbered conclusion per task, each with a number |
| LITERATŪROS SĄRAŠAS | A formats | – | ~25–30 sources |
| PRIEDAI | both | – | extra tables, link to the repo |
| **Total** | | **~45,600 (≈ 25.3 pages)** | A ≈ 22,700, B ≈ 22,900 (shared sections counted half each; B drafts more of the shared ones) |

### Draft aim, tasks and contributions (Lithuanian, final wording at step 5.6)

**Tikslas** – palyginti skirtingų architektūrų neuroninių tinklų modelius emocijoms atpažinti iš trumpų tekstų anglų kalba ir įvertinti jų tikslumą, sąnaudas bei gebėjimą apibendrinti.

**Uždaviniai:**
1. Išanalizuoti literatūrą apie emocijų atpažinimą iš teksto ir tam naudojamus neuroninių tinklų modelius. → 1 skyrius
2. Ištirti duomenų rinkinį ir jo kokybę, nustatyti bazinį tikslumo lygį. → 2.2, 3.1
3. Sukurti ir vienodomis sąlygomis apmokyti dešimt skirtingų architektūrų modelių. → 2.4, 3.2
4. Palyginti modelius pagal kokybės kriterijus, sąnaudas ir statistinį reikšmingumą. → 3.3
5. Ištirti užpildymo kaukės, klasių disbalanso, optimizatoriaus ir mokymosi greičio, mokymo imties dydžio ir klasių skaičiaus įtaką rezultatams. → 3.4–3.8
6. Įvertinti modelių gebėjimą apibendrinti kitame duomenų rinkinyje ir lietuviškiems sakiniams, išanalizuoti klaidas. → 3.9–3.11

**Darbų pasiskirstymas:** Kursinį darbą atliko du studentai. Darbas pasidalintas pagal lyginamų modelių šeimas.
- Teodora Einorytė tyrė nerekurentinius modelius: logistinę regresiją, daugiasluoksnį perceptroną, vektorinio žodžių atvaizdavimo, konvoliucinį (Conv1D) ir Transformer tinklus. Ji parengė duomenų įkėlimo ir paruošimo modulį, atliko duomenų analizę ir mokymo imties dydžio, binarinės klasifikacijos bei apibendrinimo kitiems duomenims tyrimus, sukūrė emocijų atpažinimo įrankį.
- Edgaras Legus tyrė rekurentinius modelius: SimpleRNN, LSTM, GRU, BiLSTM ir BiLSTM su dėmesio mechanizmu. Jis parengė vertinimo, mokymo ir rezultatų palyginimo modulius, atliko optimizatorių, užpildymo kaukės ir klasių disbalanso tyrimus bei klaidų analizę.

Kiekvienas narys aprašė savo modelius teorinėje ir metodinėje dalyse. Įvadą, modelių palyginimą ir išvadas abu nariai parengė kartu. Išsamus pasiskirstymas pateiktas 1 lentelėje.

### Writing rules

- **Every model section uses the same template:** principle → formula or diagram → pros and cons → our architecture (layer table with output shapes and parameter counts).
- **Every results section uses the same template:** what we did (1–2 sentences) → table or figure → what we see (with numbers) → why → which task it answers.
- **Tasks map one-to-one** to sections and to conclusions. Each conclusion contains at least one number.
- **Figures and tables** are numbered, captioned in Lithuanian ("1 pav. …", "1 lentelė. …") and referred to in the text. Sources are given for anything not ours.
- **Numbers come only from the generated tables.** Use a decimal comma in Lithuanian text (0,881; 92,4 %).
- **Every claim from literature is cited.** Every source is checked in Google Scholar before it goes in.
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

### Cross-review checklist (step 6.1)
- Terms match the table above.
- Every figure and table is referenced in the text and has a Lithuanian caption.
- Numbers match the generated tables.
- Each model section follows the template.
- Every literature claim has a citation.
- Lithuanian diacritics are correct, and no English is left in captions or axes.
- The section stays within ±15 % of its character budget.

## 11. Talk (7–10 min, ~13 slides)

| # | Slide | Presents |
|---|---|---|
| 1 | Title, team | A |
| 2 | Problem, aim, research questions | A |
| 3 | Data and its quality (keyword finding) | A |
| 4 | A's models | A |
| 5 | B's models | B |
| 6 | Protocol: fair comparison, 5 seeds, McNemar | B |
| 7 | Main comparison table / chart | B |
| 8 | Quality vs cost | B |
| 9 | Optimisers, masking finding, class weights | B |
| 10 | Training-set size, binary vs 6 classes | A |
| 11 | Generalisation: GoEmotions, Lithuanian sentences | A |
| 12 | Demo (live or screenshot) | A |
| 13 | Conclusions | B |

Each person must also be able to answer questions about the other's part at a basic level.

## 12. Starter literature (check each in Google Scholar before citing)

**Emotions and data**
- Ekman, P. (1992). An argument for basic emotions. *Cognition and Emotion*, 6(3–4), 169–200.
- Plutchik, R. (2001). The nature of emotions. *American Scientist*, 89(4), 344–350.
- Saravia, E. et al. (2018). CARER: Contextualized affect representations for emotion recognition. *EMNLP 2018*, 3687–3697.
- Demszky, D. et al. (2020). GoEmotions: A dataset of fine-grained emotions. *ACL 2020*.
- Mohammad, S. et al. (2018). SemEval-2018 Task 1: Affect in tweets. *SemEval 2018*.
- Acheampong, F. A., Wenyu, C., Nunoo-Mensah, H. (2020). Text-based emotion detection: Advances, challenges, and opportunities. *Engineering Reports*, 2(7).
- Alswaidan, N., Menai, M. E. B. (2020). A survey of state-of-the-art approaches for emotion recognition in text. *Knowledge and Information Systems*, 62.

**Text representation**
- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space. arXiv:1301.3781.
- Pennington, J., Socher, R., Manning, C. (2014). GloVe: Global vectors for word representation. *EMNLP 2014*.

**Models**
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP 2014*.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP 2014*.
- Chung, J. et al. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv:1412.3555.
- Schuster, M., Paliwal, K. (1997). Bidirectional recurrent neural networks. *IEEE Trans. Signal Processing*, 45(11).
- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Yang, Z. et al. (2016). Hierarchical attention networks for document classification. *NAACL 2016*.
- Zhou, P. et al. (2016). Attention-based bidirectional LSTM networks for relation classification. *ACL 2016*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*.
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.

**Training and evaluation**
- Srivastava, N. et al. (2014). Dropout. *JMLR*, 15.
- Kingma, D., Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML 2013*. (Momentum, Nesterov; S9)
- Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *JMLR*, 12. (Adagrad; S9)
- Tieleman, T., Hinton, G. (2012). Lecture 6.5 – RMSprop. *COURSERA: Neural Networks for Machine Learning*. (RMSprop; S9)
- Dozat, T. (2016). Incorporating Nesterov momentum into Adam. *ICLR 2016 Workshop*. (Nadam; S9)
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. (AdamW; S9)
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2).
- Dietterich, T. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7).

**Textbooks (course list)**
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn and TensorFlow*. O'Reilly.

## 13. Open decisions (settle at step 0.1)

1. **Section numbering.** The lecturer's structure numbers the intro "1." and the conclusions "5."; the current doc leaves the intro unnumbered. Recommendation: follow the lecturer exactly, because the grading percentages are given per numbered part.
2. **Citation style.** Recommendation: IEEE numeric [1], compact for ~30 sources. APA (Author, year) also works.
3. **Talk date.** Recommendation: 11-09 or 11-10.
4. **Who uploads to Moodle.**
