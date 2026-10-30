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

1. **Which architecture recognises emotions in short texts best, and at what cost?** 10 models from 4 families, 5 seeds each, statistical test with correction for multiple comparisons, parameters and time; a fine-tuned DistilBERT run by us as the competitor.
2. **Which training choices matter for text?** Optimiser and learning rate (9 optimisers), padding/masking in recurrent networks, class weights and focal loss for uneven classes, regularisation (Dropout, L2), depth, pretrained word vectors (GloVe), training-set size, 6 vs 2 classes.
3. **Do the models generalise?** Test on a different, human-labelled dataset (GoEmotions) and on Lithuanian: our 2,000 test texts machine-translated into Lithuanian.
4. **What do the models actually learn: keywords or context?** Keyword test (S13: the emotion word removed, or only the emotion word kept), keyword baseline, attention weights, negation, error analysis.

The dataset is easy and the labels follow the emotion word (section 3), so the architectures may end up within a point of each other. If they do, that is a finding, not a failure: questions 3 and 4 then carry the study, and the report says so.

## 2. How this goes beyond lab 1 and lab 2

| | Lab 1 | Lab 2 | Coursework |
|---|---|---|---|
| Task | MNIST | MNIST + IMDB, 2 classes | 6 emotions, uneven classes (3.6 %–33.5 %) |
| Architectures | Dense | 5 (Conv2D, Dense, Conv1D, LSTM, GRU) | 10 in 4 families. New vs lab 2: logistic regression, Transformer, SimpleRNN, BiLSTM, BiLSTM + attention |
| Trained runs | ~30, 1 each | ~21–24, 1 each | ~630: tuning (size and learning rate per model) on validation, then 5 seeds → mean ± std, McNemar test with Holm correction |
| Metrics | accuracy, loss | accuracy, loss, time | per-class precision/recall/F1, macro-F1, confusion matrix, train vs validation gap, parameters, train and inference time |
| Context | – | – | baselines (majority 34.8 %, keyword rule 73.8 %) and competitor: DistilBERT fine-tuned by us, same metrics and test sets |
| Data | ready | ready | own data-quality analysis (keyword labels, conflicts, leakage) |
| Generalisation | – | – | GoEmotions (6,535 texts); Lithuanian: the 2,000 test texts machine-translated |
| Extra studies | depth, width, batch, epochs, activations, 3 optimisers × 4 rates (MNIST, Dense) | pooling, one-hot vs Embedding, LSTM/GRU units, layers, activation, optimiser (1 run each) | 9 optimisers × 5 rates on text models (5 seeds), masking, class weights vs focal loss, Dropout vs L2, depth, GloVe frozen vs fine-tuned, training-set size, binary vs 6 classes |
| Interpretation | – | – | keyword test (emotion word removed / kept alone), attention heatmaps, error examples, negation subset |
| Product | – | – | `predict.py` + a small web app: sentence in → emotion probabilities and the words the model looked at |

## 3. Data

### dair-ai/emotion (main)
- Saravia et al., CARER, EMNLP 2018, DOI 10.18653/v1/D18-1404. English tweets, labels 0–5: sadness, joy, love, anger, fear, surprise → **liūdesys, džiaugsmas, meilė, pyktis, baimė, nuostaba**.
- Official split: train 16,000 / validation 2,000 / test 2,000.
- Classes (train): joy 33.5 %, sadness 29.2 %, anger 13.5 %, fear 12.1 %, love 8.2 %, surprise 3.6 %.
- Texts: lowercase, no punctuation, 19.2 words on average (median 17, 95th percentile 41, 99th percentile 52, max 66). 15,212 distinct words; the top 10,000 cover 98.3 % of word occurrences.

**Quality issues (these go in the report, not under the rug):**
- 97.3 % of texts contain *feel/feeling/felt*. The labels were assigned from the emotion word, so "i didnt feel humiliated" is *sadness*.
- A one-line rule ("word after *feel*" → most common label in train) gets **73.8 %** on test (macro-F1 0.672). The word after *feel* skips intensifiers ("so", "very", "slightly" …); when it is a pronoun or conjunction ("i feel like i …") the text has no emotion word, which is the case for about a quarter of the texts (`data.emotion_word`, lists checked on validation).
- 51 texts appear more than once with **different** labels; 11 test texts also appear in train (5 validation texts likewise).
- **Decision:** keep the official split unchanged so our numbers are comparable with published results; report the issues and their size.

### GoEmotions (generalisation test only, never trained on)
- Demszky et al., ACL 2020. Reddit comments, human-labelled, 28 labels.
- We take the single-label texts whose label is one of our 6 emotions: **6,535 texts** (love 1,760, anger 1,265, joy 1,052, sadness 1,003, surprise 902, fear 553).
- Preprocess like our data (lowercase, strip punctuation).

### Lithuanian test set (generalisation to another language, 2,000 texts)
Used only for testing; the models are trained on English only.
- Source: [SkyWater21/lt_twitter_emotions](https://huggingface.co/datasets/SkyWater21/lt_twitter_emotions) (MIT), the whole CARER corpus, i.e. dair-ai/emotion "unsplit", 416,809 texts, machine-translated into Lithuanian with Google Translate. It keeps the English text and the same 6 labels.
- Our test split is part of that corpus, so we keep the rows whose English text and label match our 2,000 test texts → `data/lt_test.csv` (`test_index`, `text_lt`, `text_en`, `label`). All 2,000 were found.
- **Why it matters:** the same 2,000 texts with the same labels in both languages. The difference between English and Lithuanian accuracy is then the effect of the language alone, measured on enough texts for a meaningful number (±2 points).
- **Limits, stated in the report:** machine translation, not natural Lithuanian ("translationese"); a translation can lose or change the emotion word.
- Expected result: the Lithuanian versions fail, because 92.1 % of their words are unknown to the English vocabulary (`dataset_overview.csv`); the English originals work. The model knows only its training language.
- Examples for the report and the talk are taken from this set; the web app takes any Lithuanian sentence typed in live.
- **Own sentences dropped (2026-10-30):** the plan first had 60 hand-written Lithuanian sentences. With 60 texts any accuracy has a ±12-point uncertainty, and they would show the same failure as the 2,000 translated texts, so this set replaces them.

### Binary version (study S5)
Positive = joy, love, surprise; negative = sadness, anger, fear (45.2 % / 54.8 % in train). Surprise is debatable; say so in the report.

## 4. Fixed protocol (`config.py`), the same for every run

| Setting | Value |
|---|---|
| Tokenisation | `TextVectorization`, vocabulary 10,000, adapted on **train only** |
| Sequence length | 50 tokens (98.7 % of train texts fit whole; `length_coverage.csv`) |
| Embedding size | 64, learned (S12: 100-d learned or GloVe) |
| Masking | `Embedding(mask_zero=True)` for sequence models (except where S2 changes it). Exceptions: Conv1D (fixed windows, no use for a mask) and the Transformer, which was better without the mask on validation (step 2.1); listed in `config.NO_MASK_MODELS` |
| Optimiser | Adam, learning rate 0.001 (`config.OPTIMIZER`). One optimiser for every run; S9 checks this choice before T0 and may replace it (step 2.3) |
| Learning rate | per model, tuned in T0b around the protocol rate (× 0.3 / × 1 / × 3), in `config.MODEL_LEARNING_RATES`; until then the protocol rate |
| Loss | sparse categorical cross-entropy, softmax output (S3: focal loss) |
| Regularisation | none beyond EarlyStopping, except the MLP's own Dropout 0.5 (S10 adds Dropout or L2) |
| Depth | 1 Conv1D / recurrent layer (S11: 2) |
| Batch | 64 |
| Epochs | up to 30, `EarlyStopping(val_loss, patience=3, restore_best_weights=True)` |
| Seeds | 1, 2, 3, 4, 5 (`keras.utils.set_random_seed`) |
| Model choice | by **validation** macro-F1; the test set is used only for final numbers |

**Machines:** every study runs entirely on one machine. S1 is where training times are compared, so it runs on B's machine.

## 5. Models (10) and comparison points

| # | Model | Family | Owner | Tuned (3 values) |
|---|---|---|---|---|
| 0 | Logistic regression (multi-hot → softmax) | feedforward | A | – (baseline) |
| 1 | MLP (multi-hot → Dense → Dropout 0.5 → softmax) | feedforward | A | hidden units 32/64/128 |
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
- **Keyword rule** (word after *feel* → label): 73.8 % on test.
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
- **The Transformer goes the other way (step 2.1, full data):** with the padding mask (attention only to real words, average over real words) validation 85.9 % and 88.6 % (seeds 1, 2), test 86.5 % / 0.808 and 88.0 %; without it validation 90.8 %, test 90.0 % / 0.851 (seed 1). A likely reason: without the mask the padding positions give attention a place to put weight when no word matters (an "attention sink"), and their learned position vectors carry the text length. So the protocol runs the Transformer without the mask, and S2 measures it with the mask (5 seeds).
- **Time:** all models together train in ~3 min on CPU.

## 6. Experiments

Studies are listed in the order they run. "Side models" = Conv1D and GRU, the best non-recurrent and the best recurrent model of the pilot run; side studies use them so that each finding is checked on both families at a manageable cost. Where a study's baseline is the protocol itself, it is **not run again**: `compare.py` takes it from S1.

| ID | What changes (everything else fixed) | Baseline | Runs | Runs on | Owner | Report |
|---|---|---|---|---|---|---|
| S9a | Optimiser grid: 9 optimisers × learning rates 1.0 / 0.1 / 0.01 / 0.001 / 0.0001, side models, seed 1; each optimiser's best rate by validation macro-F1 | – | 9 × 5 × 2 = 90 | B | B | 4.2 |
| S9 | Optimisers, each at its best S9a rate, side models, 5 seeds | – | 9 × 2 × 5 = 90 | B | B | 4.2 |
| T0a | Tuning, size: one capacity setting × 3 values per model, protocol learning rate, seeds 1–2, pick by validation macro-F1 | – | 9 × 3 × 2 = 54 | B | each picks for own models | 4.3 |
| T0b | Tuning, learning rate: protocol rate × 0.3 / × 3 at the chosen size (× 1 is T0a), all 10 models, seeds 1–2 | × 1 = T0a | 10 × 2 × 2 = 40 | B | each picks for own models | 4.3 |
| **S1** | **Main comparison:** 10 tuned models. Also saves predictions for test, GoEmotions, the LT test set and the two keyword-test versions of test (S13) | – | 10 × 5 = 50 | B | both | 4.4 |
| S8 | Cost: parameters, s/epoch, training time, inference time per 1,000 texts → quality vs cost chart | – | from S1 | – | B | 4.4 |
| S14 | Competitor: DistilBERT (`distil_bert_base_en_uncased`, 66M parameters) fine-tuned on the same train split, 3 epochs, learning rate 2e-5; same predictions as S1 | – | 3 seeds | B (same machine as S1, for the cost chart) | A | 4.4, 4.8, 4.9 |
| S2 | Padding/masking: post padding without mask / pre padding without mask, for SimpleRNN, LSTM, GRU, BiLSTM; and the Transformer **with** the mask (its protocol is without, step 2.1) | protocol = S1 | 4 × 2 × 5 + 5 = 45 | B | B | 4.5 |
| S3 | Uneven classes: class weights (inverse frequency) for all 10 models; focal loss (γ = 2) for the side models | neither = S1 | 10 × 5 + 2 × 5 = 60 | B | B | 4.6 |
| S10 | Regularisation: Dropout 0.2 / Dropout 0.5 before the output layer / L2 1e-4 on the weights, side models | none = S1 | 3 × 2 × 5 = 30 | B | B | 4.7 + appendix |
| S11 | Depth: 2 stacked Conv1D / recurrent layers instead of 1, for Conv1D, LSTM, GRU | 1 layer = S1 | 3 × 5 = 15 | B | B | 4.7 + appendix |
| S12 | Word vectors, all 100-dimensional: learned / GloVe frozen / GloVe fine-tuned, side models; also saves GoEmotions, LT test and keyword-test predictions | learned 64-d = S1 | 3 × 2 × 5 = 30 | A | A | 4.7 + appendix |
| S4 | Training-set size: 1k, 2k, 4k, 8k, 16k (stratified) for logreg, the best non-recurrent and the best recurrent model of S1 | – | 5 × 3 × 5 = 75 | A | A | 4.7 + appendix |
| S5 | 2 classes vs 6 classes, all 10 models | 6 classes = S1 | 10 × 5 = 50 | A | A | 4.7 + appendix |
| S13 | **Keywords or context?** Test texts with the emotion word (the word after *feel/feeling/felt*) replaced by an unknown word, and with only that word kept; reads the S1, S12 and S14 predictions, no retraining | original test | 0 | – | A | 4.8 |
| S6 | Generalisation: reads the S1, S12 and S14 predictions for GoEmotions, and the LT test set (2,000, same texts as test), no retraining | – | 0 | – | A | 4.9 |
| S7 | Errors: confusion matrices, most-confused pairs, misclassified examples, negation subset, attention heatmaps | – | 0 | – | B | 4.10 |

About 630 training runs, roughly 5–8 hours of CPU time for our models (side-model runs take ~20–60 s each) plus several hours for the 3 DistilBERT runs.

**Core and side studies.** S1 (with S8, S14), S9, S2, S3, S13, S6 and S7 answer the research questions directly; they get their own report sections and are discussed in depth. S10, S11, S12, S4 and S5 are side studies: one shared report section (4.7) with a summary table and 2–3 sentences each, full tables and figures in the appendices, which do not count towards the 22–26 pages.

**S9, optimisers (runs first, before T0):**
- **Optimisers (9):** all eight from lecture 4 (SGD, SGD + Momentum 0.9, Nesterov, Adagrad, Adadelta, RMSprop, Adam, Nadam; lecture 5 compares six of them in a chart), plus AdamW (Adam with decoupled weight decay, standard for Transformers). Keras defaults otherwise.
- **Why a learning-rate grid:** lab 1 (task 6.3) showed that SGD needs a much larger learning rate than Adam or RMSprop, and Adadelta is normally used with 1.0. Comparing all of them at 0.001 would be unfair, so every optimiser gets the same 5-value grid (S9a) and its own best rate before the 5-seed comparison (S9).
- **Why the side models:** recurrent networks are where plain SGD is expected to struggle (gradients pass through 50 time steps), so the two families may react differently.
- **Sizes:** the middle T0 value (Conv1D 128 filters, GRU 64 units), because T0 runs after S9.
- **Decision rule for the main protocol (👤 both):** the protocol keeps one optimiser for every model, so the main comparison changes only the architecture. After S9, find the optimiser (at its best rate) with the highest mean validation macro-F1 over the 10 runs (2 models × 5 seeds). If Adam, at its own best S9a rate, is within one standard deviation of that mean (the standard deviation of the best optimiser's 10 runs), keep Adam and set `config.LEARNING_RATE` to Adam's best rate. Otherwise switch `config.OPTIMIZER` and `config.LEARNING_RATE` to the best optimiser. Either way this is done before T0, so T0 and all later studies use the final choice. The test set is not used for this decision.

**T0b, learning rate per model:** S9 picks one optimiser for everything, but a single learning rate tuned on Conv1D and GRU could be unfair to models that train differently (the Transformer and SimpleRNN especially), and a lecturer may ask exactly that. T0b tries the protocol rate × 0.3 and × 3 for every model at its T0a size; × 1 is the T0a run. Sizes first, then rates: one setting at a time, as everywhere else. The optimiser stays the same for all models.

**S10–S12, side studies (run after S1, with the T0 sizes and learning rates):**
- **S10 regularisation (lecture 6):** Dropout drops each unit with probability p during training (lecture 6, §3); L2 adds λ·Σw² to the loss (§1.1). Compare train vs validation loss (`train_loss`, `val_loss`) to show whether they reduce overfitting; EarlyStopping stays on in every arm, so the question is what they add to it.
- **S11 depth (lab 2, task 2.3):** two stacked layers (`return_sequences=True` on the first recurrent layer). Shows whether a deeper network pays off on 19-word texts.
- **S12 pretrained word vectors (lecture 6, §6.2):** GloVe 6B 100-d (Pennington et al., 2014). The learned arm is also 100-d, so only the origin of the vectors changes. Questions: do pretrained vectors help with 16,000 training texts, and do they help on GoEmotions (another domain)? The GloVe file (~330 MB) is downloaded by `download_glove.py`, kept out of git.
- **Focal loss (lecture 4, §2.1):** FL = −(1 − p_t)^γ · log p_t with γ = 2: well-classified texts contribute less, so the rare classes weigh more without explicit weights.

**S13, keywords or context? (the most direct answer to research question 4):**
- **Two modified copies of the test set**, built in `data.keyword_variants`. The emotion word (`data.emotion_word`) is the first word after *feel / feels / feeling / felt*, skipping intensifiers and small words ("so", "very", "slightly" …), the same word the keyword rule (73.8 %) uses. 75.1 % of test texts have one; the rest (no *feel* word, or a clause after it such as "i feel like i …") stay unchanged and are reported separately.
  - *Without the keyword*: that word becomes a word outside the vocabulary, so it is encoded as [UNK]. "i feel so **humiliated** today" → "i feel so **[UNK]** today".
  - *Only the keyword*: everything else is dropped. "i feel so humiliated today" → "humiliated".
- **What we measure**, per model: accuracy and macro-F1 on both copies vs the original test set, i.e. the drop without the keyword and how much the keyword alone explains. A model that uses context loses less without the keyword; a keyword matcher keeps almost everything with the keyword alone. Plotted for all models, the keyword rule and DistilBERT in one chart.
- **Link to attention:** for the BiLSTM + attention model, the share of attention weight that falls on the keyword (S7 heatmaps show examples; S13 gives the average).
- **No retraining:** S1 (and S12, S14) save predictions for both copies, like for GoEmotions.

**S14, competitor (DistilBERT fine-tuned by us):**
- KerasHub `DistilBertClassifier` from the `distil_bert_base_en_uncased` preset (Sanh et al., 2019), fine-tuned on our train split with our validation set and EarlyStopping; texts cut at 64 word pieces; Adam 2e-5, batch 32, up to 3 epochs (the usual BERT fine-tuning settings), seeds 1–3.
- Uses its own tokenizer, so it skips our `TextVectorization`; everything after the predictions (metrics, McNemar, cost, GoEmotions, LT test, keyword test) is shared with our models.
- Downloads the pretrained weights (~250 MB) once; kept out of git. Answers "how far are our 0.7M-parameter models from a 66M-parameter pretrained transformer, and at what cost?"

**Robustness check, duplicates:** 11 test texts also appear in train (section 3). The main table also gives macro-F1 on the test set without them (`data.test_in_train`); if it barely changes, the leak does not matter, and the report says so with the number.

**What stays fixed, and why (one sentence each in report 3.1 / 3.7):**
- **Hold-out validation, not k-fold:** the dataset has an official split; lecture 6 (§5) notes that k-fold is rare for neural networks because of training cost, and 5 seeds give the spread instead.
- **Batch size 64:** a common middle value (lecture 4, §3.7: too small is noisy, too large is slow); lab 1 (task 6.1) already studied batch size, so it is not repeated.
- **Activations:** ReLU in Dense/Conv1D hidden layers (best in lab 1, task 6.2), tanh inside recurrent cells (the Keras and lecture 8 default), softmax output.
- **Binary task (S5) uses a 2-unit softmax:** with 2 classes it is mathematically equivalent to 1 sigmoid unit + binary cross-entropy, and keeps one output type for all studies.
- **EarlyStopping on val_loss, model choice on validation macro-F1:** the loss is smooth and moves every epoch, so it is a stable stopping signal; macro-F1 is what we report, so it decides between settings.
- **Pooling:** GlobalMaxPooling after Conv1D (Kim, 2014); lab 2 (task 1.2) already compared max and average pooling.
- **Transformer position embedding:** learned, as in the Keras example; sine/cosine encoding (lecture 9, §3.9) gives similar results for short texts.
- **Sequence length 50:** covers 98.7 % of train texts (`length_coverage.csv`). It is also above the IQR outlier bound of the word counts (lecture 1, §8.4): Q1 = 11, Q3 = 25, Q3 + 1.5·IQR = 46, so only outlier-length texts (2.3 % are longer than 46 words) get cut.

**McNemar test:** best model vs every other model (and vs DistilBERT), per seed. Implemented ourselves in numpy: χ² = (|b − c| − 1)² / (b + c); p from the χ² distribution with 1 degree of freedom, using `math.erfc`.
- **Holm–Bonferroni correction** (`evaluation.holm`, Holm, 1979): the best model is compared with 10 others, so some p < 0.05 would appear by chance. Sort the m p-values, compare the i-th smallest with 0.05 / (m − i + 1), stop at the first that fails. Report in how many of the 5 seeds each difference stays significant after the correction.

**Not in the plan (only if both agree later):**
- A CNN + BiLSTM hybrid.

## 7. Code

```
neuroniniai-tinklai-kd-2026/
  README.md, PLAN.md, requirements.txt, .gitignore
  config.py            shared settings (section 4), class names in English and Lithuanian; learning rates from
                       S9a, sizes and learning rates from T0 are filled in here
  download_data.py     A: one-off script that created data/*.csv from Hugging Face
  download_lt_test.py  B: one-off script: downloads the Lithuanian translation of the CARER corpus and keeps
                       our 2,000 test texts → data/lt_test.csv (the 56 MB download itself is not kept)
  download_glove.py    A: one-off download of GloVe 6B 100-d into data/glove/ (git-ignored, ~330 MB) for S12
  data.py              A: loading, vectorising, pre/post padding, binary version, subsets, GoEmotions, LT, GloVe matrix,
                       keyword-test versions of the test set (S13), test texts that also appear in train
  data_analysis.py     A: data analysis figures + quality checks + baselines (report 4.1)
  models_a.py          A: logreg, MLP, Embedding, Conv1D, Transformer
  models_b.py          B: SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention
  evaluation.py        B: metrics in numpy, confusion matrix, McNemar, Holm correction, plots
  train.py             B: one run → one CSV row + curve + matrix + predictions (+ saved model for S1)
  run_study.py         B: runs a whole study: python run_study.py S1
  competitor.py        A: S14, fine-tunes DistilBERT (KerasHub) and writes the same results row and predictions
  compare.py           report tables and figures, one function per study
                       (B: S9, T0, S1 + S14 + Holm + duplicates check, S8, S2, S3, S10, S11, S7; A: S12, S4, S5)
  generalization.py    A: S6 and S13 tables and figures (S1, S12 and S14 predictions)
  predict.py           A: command-line demo
  app.py               A: small web app (Gradio): sentence in → bar chart of emotion probabilities and the
                       sentence with words shaded by attention weight (best BiLSTM + attention model)
  data/                emotion_{train,validation,test}.csv, goemotions_{train,validation,test}.csv, lt_test.csv
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
load_lt_test(task="6") -> (x_lt, x_en, y, test_index)   # the 2,000 test texts translated (data/lt_test.csv), cleaned
                                                #   like the rest; test_index = row in emotion_test.csv, to pair LT and EN
glove_matrix(vectorizer, dim=100) -> np.array   # (VOCAB_SIZE, 100); row i = GloVe vector of word i, zeros if missing (S12)
keyword_variants(texts) -> dict                 # S13: "without" (keyword → out-of-vocabulary word), "only" (keyword alone),
                                                #      "keyword" (the word or ""), "has_keyword" (bool per text)
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
```

```bash
python train.py --model lstm --seed 1 --study S1 [--size 64] [--no-mask | --mask] [--padding pre]
                [--class-weight] [--loss focal] [--fraction 0.25] [--task 2] [--optimizer sgd --lr 0.1]
                [--dropout 0.5] [--l2 0.0001] [--layers 2] [--embedding glove-frozen] [--dev]
python run_study.py S2           # all runs of one study (S9a, S9, T0a, T0b, S1–S5, S10–S12)
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
- test: `test_acc`, `test_loss`, `test_macro_f1`, `f1_sadness` … `f1_surprise` (binary runs: `f1_negative`, `f1_positive`)

Predictions for every text are saved per run in `predictions/` (test; in S1, S12 and S14 also GoEmotions, the LT test set and the two keyword-test copies of test), which S6, S7, S13 and McNemar need. `lr` holds the learning rate actually used, so per-model rates from T0b show in every row.

New packages: `keras-hub` (S14) and `gradio` (app); both go into `requirements.txt` with exact versions.

## 8. Who does what (summary)

| | A — Teodora | B — Edgaras |
|---|---|---|
| Models (5 each) | logreg, MLP, Embedding, Conv1D, Transformer | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention |
| Shared code | `data.py`, `data_analysis.py`, `generalization.py`, `competitor.py`, `predict.py`, `app.py` | `evaluation.py`, `train.py`, `run_study.py`, most of `compare.py` |
| Studies | data analysis, S13 keyword test, S14 DistilBERT (run on B's machine), S12 GloVe, S4, S5, S6, demo app | S9 optimisers, running T0a/T0b and S1, S2, S3 (+ focal loss), S10 regularisation, S11 depth, S7, S8, Holm and duplicates check |
| Report text | ~22,300 characters (section 10) | ~22,500 characters + drafts of 4.3, 4.4, 3.5.11, conclusions (section 10) |
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

### Phase 2 — Models
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 2.1 | `models_a.py`: 5 models; Conv1D also takes the side-study arguments (section 7) | A (done by B) | 1.3 | each trains via `train.py --dev`, Conv1D also with `--dropout`, `--l2`, `--layers 2`, `--embedding learned100` | ✅ `python models_a.py` checks parameter counts against the formulas (logreg 60,006, MLP 640,454, Embedding 644,550, Conv1D 681,862, Transformer 677,062, the same for 1/2/4 heads); `--dev` runs and saved-model loading pass, Conv1D also with the side-study options. Full-data check (seed 1), test accuracy / macro-F1: MLP 88.9 % / 0.837, Embedding 86.3 % / 0.805, Conv1D 91.7 % / 0.871, Transformer without mask 90.0 % / 0.851 vs with mask 86.5 % / 0.808 (seed 2: 88.0 %). So the Transformer runs without the mask (`config.NO_MASK_MODELS`, section 4; `train.py --mask` turns it on, and the variant name shows "mask" only when it differs from the model's default); S2 has a Transformer-with-mask arm (45 runs) |
| 2.2 | `models_b.py`: 5 models; the attention model can output word weights; GRU takes the side-study arguments, LSTM takes `layers` | B | 1.3 | each trains via `train.py --dev`, GRU also with the side-study options | ✅ all 5 take every side-study argument; `python models_b.py` checks parameter counts against the lecture 8 formulas; full-data check (seed 1): GRU 92.4 % / 0.881, BiLSTM + attention 92.2 % / 0.874; `attention_weights()` works on the saved model (padding gets weight 0) |
| 2.3 | S9a + S9 optimisers on Conv1D and GRU; best rates → `config.S9_LEARNING_RATES`; main optimiser by the rule in section 6 (👤 both) | B | Conv1D from 2.1, GRU from 2.2 | 180 S9a/S9 rows, `config.OPTIMIZER` and `LEARNING_RATE` confirmed or changed | 🔄 branch `step-2.3`, not committed. **S9a done** (90 runs, 70 min). Best rate by mean validation macro-F1 of Conv1D and GRU: SGD 1.0, Momentum 0.1, Nesterov 0.1, Adagrad 0.1, Adadelta 1.0, RMSprop 0.01, Adam 0.001, Nadam 0.01, AdamW 0.001. At those rates (seed 1): Nesterov 0.903, Momentum 0.902, AdamW 0.897, Adam 0.897, Nadam 0.897, RMSprop 0.896, Adadelta 0.893, SGD 0.892, Adagrad 0.890; Conv1D is best with Adam (0.906), GRU with Nesterov (0.916 vs Adam 0.888). Notes for 4.2: SGD's best rate is the top of the grid; 35 of 90 runs (rates too small) were still improving at the 30-epoch limit, so the grid finds the rate that learns within the protocol's 30 epochs. Fixed on the way: `best_epoch` now comes from `EarlyStopping` (np.argmin pointed at the first NaN epoch when the loss diverged). Rates written to `config.S9_LEARNING_RATES`. **S9 done** (90 runs, 28 min). Mean validation macro-F1 over 10 runs (± sd): Nesterov 0.902 ± 0.009, Momentum 0.900, Adam 0.898 ± 0.008, AdamW 0.898, RMSprop 0.898, Nadam 0.898, Adagrad 0.896, Adadelta 0.896, SGD 0.616 ± 0.377. **Rule: Adam is within one sd of Nesterov (0.898 ≥ 0.902 − 0.009 = 0.893), so Adam 0.001 stays** (the values already in `config.py`). Notes for 4.2: no optimiser is best for both models (Conv1D: Adam/AdamW 0.904; GRU: Nesterov 0.909, RMSprop and Nadam 0.908, Adam 0.893); SGD at 1.0 diverged on Conv1D in 4 of 5 seeds (worked only with seed 1, the S9a seed), so 1.0 is at the edge of stability; seed 1 gives slightly different numbers in S9 than in S9a (CPU computations are not exactly repeatable). Left: confirm the decision (👤 both). Proposal to decide with it: GRU lags with Adam 0.001 (0.893 vs Nesterov 0.909), but with Adam 0.01 it reached 0.904 in S9a, so the gap is mostly the learning rate; adding × 10 to `config.T0B_FACTORS` (+20 T0b runs) would let every model reach 0.01 with Adam |
| 2.4 | T0 tuning: T0a sizes, then T0b learning rates; B runs both for all models; each owner picks own sizes and rates → `config.py` | run B, pick each | 2.1, 2.2, 2.3, 1.7 | T0a and T0b rows in CSV, `MODEL_SIZES` and `MODEL_LEARNING_RATES` in `config.py` |  |

### Phase 3 — Experiments
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 3.1 | S1 main comparison (saves predictions incl. keyword-test copies and LT test + best model) | B | 2.4, 1.6, 1.8 | 50 S1 rows, predictions committed |  |
| 3.2 | S2 masking, S3 class weights + focal loss | B | 3.1 | rows in CSV |  |
| 3.3 | S4 training-set size, S5 binary | A | 3.1 (S4 needs the best models of S1) | rows in CSV |  |
| 3.4 | S10 regularisation, S11 depth | B | 2.4 | rows in CSV |  |
| 3.5 | S12 GloVe: `download_glove.py`, `data.glove_matrix`, then `run_study.py S12` | A | 2.4 | 30 S12 rows, predictions committed |  |
| 3.6 | S14 competitor: `competitor.py` (KerasHub DistilBERT, same results row and predictions as `train.py`), tested on a small subset first; the 3 full runs on B's machine | A writes, B runs | 1.6, 1.7 | 3 S14 rows, predictions committed |  |

### Phase 4 — Analysis outputs
| # | Step | Owner | Needs | Done when | Status |
|---|---|---|---|---|---|
| 4.1 | `compare.py`: S9 (heatmap optimiser × rate + table), T0a/T0b, S1 with S14 and baselines, duplicates check, S8, S2, S3, S10 (train vs val gap), S11 tables and figures; McNemar table with Holm; summary table for 4.7 | B | 2.3, 3.1, 3.2, 3.4, 3.6 | files in `tables/`, `figures/` |  |
| 4.2 | `compare.py` S12, S4, S5 functions (appendix tables + rows for the 4.7 summary); `generalization.py` S6 and S13 (S1, S12 and S14 predictions) | A | 3.1, 3.3, 3.5, 3.6 | files in `tables/`, `figures/` |  |
| 4.3 | S7: confusion matrices, most-confused pairs, misclassified examples, negation subset, attention heatmaps | B | 3.1 | figures + example table |  |
| 4.4 | `predict.py` and `app.py` (Gradio: probabilities + words shaded by attention), screenshot for 4.11 and the talk | A | 3.1 | works on any sentence, LT and EN |  |

### Phase 5 — Writing (details in section 10)
| # | Step | Owner | Needs | Status |
|---|---|---|---|---|
| 5.1 | Literature review sections; systematic search in Web of Science Advanced search (query, years, hits, how many selected) for 2.4 | each own; search both | sources (start now) |  |
| 5.2 | Model theory sections | each own | – (start now) |  |
| 5.3 | Methodology: data, protocol, metrics, training | each own | 2.4 (final architectures) |  |
| 5.4 | Results sections | each own | phase 4 |  |
| 5.5 | Shared sections: main comparison, summaries | drafter, then the other adds | 4.1 |  |
| 5.6 | Product structure (3.2): diagram of the modules and of `app.py`; existing tools table (3.6) | A | 2.1 |  |
| 5.7 | Intro and conclusions (written last) | A drafts intro, B drafts conclusions, both edit | 5.1–5.6 |  |

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

Character counts are targets with spaces. "Drafter" writes the first version; the other person adds their own part and edits. Chapters are numbered exactly as in the lecturer's structure (1. Įvadas … 5. Išvados), whose grading weights are given per numbered chapter; page ranges come from the same table.

| Section | Owner | Characters | Content |
|---|---|---|---|
| **1. ĮVADAS** (2 p., 5 %) | drafter A, both | 3,400 | relevance, problem, object, aim, tasks, methods, structure, contributions (drafts below) |
| **2. LITERATŪROS APŽVALGA** (3–5 p., 20 %) | | **7,800** | |
| 2.1 Emocijų atpažinimas iš teksto | B | 1,600 | emotion models (Ekman, Plutchik), applications, difficulties (sarcasm, short texts, negation) |
| 2.2 Teksto vaizdavimo metodai | A | 1,600 | one-hot, TF-IDF, word2vec, GloVe, learned embeddings |
| 2.3 Neuroniniai tinklai teksto klasifikavimui | A + B | 2,600 | A: MLP, CNN, Transformer. B: RNN, LSTM, GRU, attention |
| 2.4 Panašių tyrimų apžvalga | A + B | 1,400 + table | **systematic search**: Web of Science Advanced search query, years, hits, how many selected and why (lecturer's recommendation); table: author, data, models, result; ~5 papers each |
| 2.5 Apžvalgos apibendrinimas | drafter A | 600 | what is missing, what this work does |
| **3. METODIKA** (9–10 p., 30–35 %) | | **17,900** | |
| 3.1 Tyrimo eiga, protokolas ir programos algoritmas | B | 1,400 | pipeline diagram, fixed settings, seeds, one change at a time; why hold-out validation, batch 64, stopping on val_loss vs choosing on macro-F1 (section 6) |
| 3.2 Kuriamo produkto struktūra | A | 700 + diagram | modules (`data` → `train` → `evaluation` → `compare`) and the `predict.py` tool: sentence in → cleaning → vectorizer → saved model → probabilities out |
| 3.3 Duomenų rinkiniai ir jų kokybė | A | 1,800 | emotion, GoEmotions, LT test from translation; quality issues |
| 3.4 Teksto paruošimas ir kodavimas | A | 1,100 | tokenisation, sequence length 50 (coverage + IQR rule), padding, multi-hot, embedding, GloVe |
| 3.5.1–3.5.5 A's models | A | 5 × 700 | logreg, MLP, Embedding, Conv1D, Transformer |
| 3.5.6–3.5.10 B's models | B | 5 × 700 | SimpleRNN, LSTM, GRU, BiLSTM, BiLSTM + attention |
| 3.5.11 Modelių suvestinė | drafter B | 400 + table | pros and cons, parameters |
| 3.6 Bazinis lygis, konkurentai ir esami įrankiai | A | 1,400 + table | majority, keyword rule; DistilBERT (published numbers and our own fine-tuning setup, S14); **existing tools** (comparable services): NRC EmoLex lexicon, Hugging Face emotion models, cloud NLP APIs (most give sentiment, not 6 emotions); which support Lithuanian |
| 3.7 Mokymo parametrai, optimizatoriai ir reguliarizacija | B | 2,100 | loss (cross-entropy, focal), class-weight formula, update rules of the 9 optimisers (compact table), Dropout and L2 formulas, early stopping, activations, T0a/T0b procedure (size, then learning rate per model), how the main optimiser is chosen |
| 3.8 Vertinimo kriterijai ir statistinis palyginimas | B | 1,700 | accuracy, precision, recall, F-measure, macro-F1, McNemar with Holm correction, all with formulas; cost measures (parameters, s/epoch, ms per 1,000 texts); generalisation drop (test − GoEmotions); keyword-test drop; train − validation gap; duplicates check |
| 3.9 Naudotos priemonės | A | 300 | Python, TensorFlow/Keras, KerasHub, Gradio, git, hardware |
| **4. PRAKTINĖ DALIS** (6–9 p., 35–40 %) | | **13,300** + figures | core studies get their own sections and depth; side studies share 4.7, full results in the appendices |
| 4.1 Duomenų analizė | A | 1,300 | class and length plots, keyword finding, baselines |
| 4.2 Optimizatoriaus ir mokymosi greičio parinkimas | B | 1,100 | S9a heatmap (optimiser × rate), S9 table (mean ± std, Conv1D vs GRU), which optimiser the protocol keeps |
| 4.3 Hiperparametrų parinkimas | drafter B | 700 + table | T0a sizes, T0b learning rates per model; full tables in appendix |
| 4.4 Modelių palyginimas | drafter B, A adds | 2,400 | main table (mean ± std) with DistilBERT (S14) and the baselines, bar chart, McNemar with Holm, duplicates check, quality vs cost (S8) |
| 4.5 Užpildymo ir kaukės įtaka | B | 1,000 | S2 table, explanation via hidden state; the Transformer as the opposite case (the mask hurts it, see section 5) |
| 4.6 Klasių disbalanso įtaka | B | 1,000 | S3: per-class F1 without / with class weights / with focal loss |
| 4.7 Kitų mokymo pasirinkimų įtaka | B (S10, S11) + A (S12, S4, S5) | 1,500 + summary table | one table: each side study, its best setting and the change vs S1 (mean ± std); 2–3 sentences each on regularisation, depth, GloVe, training-set size, binary vs 6 classes; figures and full tables in the appendices |
| 4.8 Raktiniai žodžiai ar kontekstas? | A | 1,400 | S13: accuracy and macro-F1 without the keyword / with only the keyword, per model, vs the keyword rule and DistilBERT; attention share on the keyword |
| 4.9 Apibendrinimas kitiems duomenims | A | 1,100 | GoEmotions drop (our models, GloVe, DistilBERT); LT vs EN on the same 2,000 test texts, with examples |
| 4.10 Klaidų analizė ir dėmesio vizualizacija | B | 1,200 | confusion matrices, examples, negation, heatmaps |
| 4.11 Emocijų atpažinimo įrankis | A | 600 | the web app (screenshot), how it is built (3.2), what it shows |
| **5. IŠVADOS** (1–2 p., 5 %) | drafter B, both | 2,400 | one numbered conclusion per task, each with a number |
| LITERATŪROS SĄRAŠAS | A formats | – | ~35–40 sources |
| PRIEDAI (not counted in the 22–26 pages) | both | – | 1: T0a/T0b full tables (B); 2: S10 regularisation (B); 3: S11 depth (B); 4: S12 GloVe (A); 5: S4 learning curve (A); 6: S5 binary (A); 7: per-class F1 of every model in S1 (B); 8: link to the repo |
| **Total** | | **~44,800 (≈ 24.9 pages)** | A ≈ 22,300, B ≈ 22,500 (shared sections counted half each; B drafts more of the shared ones) |

The new studies (S10–S14, focal loss) and the structure items (3.2, existing tools, search method, longer 3.7) were paid for by shorter model sections (900 → 700 each) and by moving the side studies' details to the appendices, so the total stays below the 46,800 maximum with ~2,000 to spare.

### Draft aim, tasks and contributions (Lithuanian, final wording at step 5.7)

**Tikslas** – palyginti skirtingų architektūrų neuroninių tinklų modelius emocijoms atpažinti iš trumpų tekstų anglų kalba ir įvertinti jų tikslumą, sąnaudas bei gebėjimą apibendrinti.

**Uždaviniai:**
1. Išanalizuoti literatūrą apie emocijų atpažinimą iš teksto ir tam naudojamus neuroninių tinklų modelius. → 2 skyrius
2. Ištirti duomenų rinkinį ir jo kokybę, nustatyti bazinį tikslumo lygį. → 3.3, 4.1
3. Sukurti ir vienodomis sąlygomis apmokyti dešimt skirtingų architektūrų modelių, parinkti optimizatorių ir hiperparametrus. → 3.5, 4.2, 4.3
4. Palyginti modelius tarpusavyje ir su iš anksto apmokytu DistilBERT pagal kokybės kriterijus, sąnaudas ir statistinį reikšmingumą. → 4.4
5. Ištirti užpildymo kaukės, klasių disbalanso, reguliarizacijos, tinklo gylio, iš anksto apmokytų žodžių vektorių, mokymo imties dydžio ir klasių skaičiaus įtaką rezultatams. → 4.5–4.7
6. Nustatyti, ar modeliai remiasi emocijos žodžiu, ar kontekstu, įvertinti jų gebėjimą apibendrinti kitame duomenų rinkinyje ir lietuviškiems sakiniams, išanalizuoti klaidas ir sukurti emocijų atpažinimo įrankį. → 4.8–4.11

**Darbų pasiskirstymas:** Kursinį darbą atliko du studentai. Darbas pasidalintas pagal lyginamų modelių šeimas.
- Teodora Einorytė tyrė nerekurentinius modelius: logistinę regresiją, daugiasluoksnį perceptroną, vektorinio žodžių atvaizdavimo, konvoliucinį (Conv1D) ir Transformer tinklus. Ji parengė duomenų įkėlimo ir paruošimo modulį, atliko duomenų analizę, raktinių žodžių testą, palygino modelius su DistilBERT, atliko iš anksto apmokytų žodžių vektorių, mokymo imties dydžio, binarinės klasifikacijos bei apibendrinimo kitiems duomenims tyrimus, sukūrė emocijų atpažinimo įrankį.
- Edgaras Legus tyrė rekurentinius modelius: SimpleRNN, LSTM, GRU, BiLSTM ir BiLSTM su dėmesio mechanizmu. Jis parengė vertinimo, mokymo ir rezultatų palyginimo modulius, atliko optimizatorių, užpildymo kaukės, klasių disbalanso, reguliarizacijos ir tinklo gylio tyrimus bei klaidų analizę.

Kiekvienas narys aprašė savo modelius teorinėje ir metodinėje dalyse. Įvadą, modelių palyginimą ir išvadas abu nariai parengė kartu. Išsamus pasiskirstymas pateiktas 1 lentelėje.

### Writing rules

- **Every model section uses the same template:** principle → formula or diagram → pros and cons → our architecture (layer table with output shapes and parameter counts).
- **Every results section uses the same template:** what we did (1–2 sentences) → table or figure → what we see (with numbers) → why → which task it answers.
- **Depth over breadth:** a core section explains *why* a result happens (hidden state for masking, keyword labels for S13, domain shift for GoEmotions), not only what the numbers are. Side studies in 4.7 get 2–3 sentences each; everything else goes to the appendices and is referred to ("3 priedas").
- **Differences are claimed only when they hold:** "better" means the gap is larger than the standard deviation over seeds and survives McNemar with Holm correction; otherwise write "no significant difference".
- **Tasks map one-to-one** to sections and to conclusions. Each conclusion contains at least one number.
- **Figures and tables** are numbered, captioned in Lithuanian ("1 pav. …", "1 lentelė. …") and referred to in the text. Sources are given for anything not ours.
- **Numbers come only from the generated tables.** Use a decimal comma in Lithuanian text (0,881; 92,4 %).
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

### Cross-review checklist (step 6.1)
- Terms match the table above.
- Every figure and table is referenced in the text and has a Lithuanian caption.
- Numbers match the generated tables.
- Each model section follows the template.
- Every literature claim has a citation.
- Lithuanian diacritics are correct, and no English is left in captions or axes.
- The section stays within ±15 % of its character budget.

## 11. Talk (7–10 min, ~14 slides)

| # | Slide | Presents |
|---|---|---|
| 1 | Title, team | A |
| 2 | Problem, aim, research questions | A |
| 3 | Data and its quality (keyword finding) | A |
| 4 | A's models | A |
| 5 | B's models | B |
| 6 | Protocol: fair comparison, 5 seeds, McNemar; optimiser chosen by S9 | B |
| 7 | Main comparison table / chart, with DistilBERT and the baselines; what is significant after Holm | B |
| 8 | Quality vs cost (DistilBERT: 66M parameters vs our ~0.7M) | B |
| 9 | Training choices: masking finding, class weights vs focal loss, and the 4.7 summary table | B |
| 10 | **Keywords or context?** The keyword test: accuracy without the emotion word / with only it | A |
| 11 | Generalisation: GoEmotions; Lithuanian (the 2,000 test texts translated) | A |
| 12 | Attention heatmaps and typical errors | B |
| 13 | Demo: the web app, live or screenshot | A |
| 14 | Conclusions | B |

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
- Mohammad, S., Turney, P. (2013). Crowdsourcing a word–emotion association lexicon. *Computational Intelligence*, 29(3). (NRC EmoLex; existing tools in 3.6)
- SkyWater21. *lt_twitter_emotions* (Lithuanian machine translation of the CARER corpus), Hugging Face dataset, MIT licence. https://huggingface.co/datasets/SkyWater21/lt_twitter_emotions (LT test set; cite as a dataset with the access date)

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
- Srivastava, N. et al. (2014). Dropout. *JMLR*, 15. (S10)
- Krogh, A., Hertz, J. (1992). A simple weight decay can improve generalization. *NIPS 1991*. (L2; S10)
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*. (focal loss; S3)
- Zeiler, M. (2012). ADADELTA: An adaptive learning rate method. arXiv:1212.5701. (Adadelta; S9)
- Kingma, D., Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML 2013*. (Momentum, Nesterov; S9)
- Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *JMLR*, 12. (Adagrad; S9)
- Tieleman, T., Hinton, G. (2012). Lecture 6.5 – RMSprop. *COURSERA: Neural Networks for Machine Learning*. (RMSprop; S9)
- Dozat, T. (2016). Incorporating Nesterov momentum into Adam. *ICLR 2016 Workshop*. (Nadam; S9)
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. (AdamW; S9)
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2).
- Dietterich, T. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. (Holm correction)

**Shortcuts and label artifacts (for S13 and the data-quality findings)**
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673.
- Gururangan, S. et al. (2018). Annotation artifacts in natural language inference data. *NAACL 2018*.

**Textbooks (course list)**
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn and TensorFlow*. O'Reilly.

## 13. Open decisions (settle at step 0.1)

1. **Section numbering.** ✅ Decided 2026-10-30: follow the lecturer exactly (1. Įvadas, 2. Literatūros apžvalga, 3. Metodika, 4. Praktinė dalis, 5. Išvados); section 10 and the Google Doc use it.
2. **Citation style.** Recommendation: IEEE numeric [1], compact for ~30 sources. APA (Author, year) also works.
3. **Talk date.** Recommendation: 11-09 or 11-10.
4. **Who uploads to Moodle.**
