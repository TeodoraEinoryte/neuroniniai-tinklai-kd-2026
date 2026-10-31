"""Run all training runs of one study (PLAN.md, section 6).

Run: python run_study.py S1 [--dev]
Runs already in results.csv are skipped, so an interrupted study can simply be started again. A run counts as
done only if its optimiser and learning rate also match, so a corrected rate in config.py is run again.
Each study changes one thing; everything else stays as in config.py. Where a study's baseline is
the protocol itself (e.g. the RNNs "with mask" in S2, "dropout 0" in S10), it is not run again: compare.py reads it from S1.
"""
import argparse
import os

import pandas as pd

import config
import train

T0_SEEDS = [1, 2]
RNN_MODELS = ["simplernn", "lstm", "gru", "bilstm"]                  # S2: models with a hidden state
S2_SEARCH_RATES = [0.0003, 0.001, 0.003]          # S2: Adam rates tried for the Transformer with the mask (seeds 1-2)
S4_FRACTIONS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]                    # 1k, 2k, 4k, 8k, 16k train texts
# S4, S5: logreg + the best non-recurrent and the best recurrent model of S1 on the selection set: Conv1D (0.8905);
# GRU (0.8973) and BiLSTM + attention (0.8974) tie, GRU is simpler, 2.4 times faster and the side studies' model
S4_MODELS = ["logreg", "conv1d", "gru"]
S5_MODELS = ["logreg", "conv1d", "gru"]
SIDE_MODELS = ["conv1d", "gru"]                  # S9, S10, S12, focal loss: best non-recurrent and recurrent of the pilot
S16_MODELS = ["conv1d", "gru", "bilstm_attention"]                   # context-only reference
S17_MODELS = ["logreg", "conv1d", "gru", "bilstm_attention"]         # trained on GoEmotions
S9_LEARNING_RATES = [1.0, 0.1, 0.01, 0.001, 0.0001]   # the same grid for every optimiser (S9a); 1.0 is Adadelta's usual rate
STUDIES = ["S9a", "S9", "T0a", "T0b", "S1", "S2", "S3", "S4", "S5", "S10", "S11", "S12", "S15", "S16", "S17"]
TUNED_STUDIES = STUDIES[4:]                      # everything from S1 on uses the T0 choices


def study_runs(study, dev=False):
    """All runs of a study as a list of keyword arguments for train.run. dev: skip the T0 check (quick tests only)."""
    models = list(train.MODELS)
    seeds = config.SEEDS
    if study in TUNED_STUDIES and not dev:
        check_tuning(models)
    if study == "S9a":
        # every optimiser x 5 learning rates x 2 models, seed 1: finds each optimiser's best learning rate
        return [dict(model=m, seed=1, optimizer=o, lr=lr)
                for m in SIDE_MODELS for o in train.OPTIMIZERS for lr in S9_LEARNING_RATES]
    if study == "S9":
        # every optimiser at its best learning rate from S9a (config.S9_LEARNING_RATES) x 2 models x 5 seeds
        missing = [o for o in train.OPTIMIZERS if o not in config.S9_LEARNING_RATES]
        if missing:
            raise ValueError(f"Fill config.S9_LEARNING_RATES from the S9a results first; missing: {missing}")
        return [dict(model=m, seed=s, optimizer=o, lr=config.S9_LEARNING_RATES[o])
                for m in SIDE_MODELS for o in train.OPTIMIZERS for s in seeds]
    if study == "T0a":
        # every model x its 3 sizes (logreg: its only setting, the "x 1" of its T0b) x seeds 1-2, Adam 0.001
        # set explicitly, so T0a stays the same after T0b fills MODEL_OPTIMIZERS; chosen on the selection set.
        # Edge rule: config.T0A_EDGE_SIZES adds one more size for models whose best was at the end of the grid
        return [dict(model=m, seed=s, size=size, optimizer=config.OPTIMIZER, lr=config.LEARNING_RATE)
                for m in models for size in (train.MODELS[m][2] or [None]) + config.T0A_EDGE_SIZES.get(m, [])
                for s in T0_SEEDS]
    if study == "T0b":
        # every model at its T0a size x the optimiser-rate pairs of config.T0B_GRID x seeds 1-2 (Adam 0.001 is T0a).
        # Edge rule: config.T0B_EDGE_RATES adds one more pair for models whose best rate was at the end of a list
        missing = [m for m in models if train.MODELS[m][2] and m not in config.MODEL_SIZES]
        if missing:
            raise ValueError(f"Fill config.MODEL_SIZES from the T0a results first; missing: {missing}")
        grid = [(o, lr) for o, rates in config.T0B_GRID.items() for lr in rates]
        return [dict(model=m, seed=s, size=config.MODEL_SIZES.get(m), optimizer=o, lr=lr)
                for m in models for o, lr in grid + config.T0B_EDGE_RATES.get(m, []) for s in T0_SEEDS]
    if study == "S1":
        return [dict(model=m, seed=s) for m in models for s in seeds]
    if study == "S2":
        # RNNs: post padding without mask / pre padding without mask; "with mask" is S1
        variants = [dict(mask=False), dict(mask=False, padding="pre")]
        runs = [dict(model=m, seed=s, **v) for m in RNN_MODELS for v in variants for s in seeds]
        # Transformer the other way round: its protocol is without the mask (config.NO_MASK_MODELS). The masked
        # version first gets its own short Adam rate search, then 5 seeds at the chosen rate; and the 2 x 2 check:
        # the mask only in the attention, or only in the average
        runs += [dict(model="transformer", seed=s, mask=True, optimizer="adam", lr=lr)
                 for lr in S2_SEARCH_RATES for s in T0_SEEDS]
        if config.S2_MASKED_TRANSFORMER_LR is not None:
            runs += [dict(model="transformer", seed=s, mask=True, optimizer="adam", lr=config.S2_MASKED_TRANSFORMER_LR)
                     for s in seeds if not (s in T0_SEEDS and config.S2_MASKED_TRANSFORMER_LR in S2_SEARCH_RATES)]
        runs += [dict(model="transformer", seed=s, mask=m) for m in ["attention", "pooling"] for s in seeds]
        return runs
    if study == "S3":
        # class weights for every model + focal loss for the two side models; "neither" is S1
        return ([dict(model=m, seed=s, class_weight=True) for m in models for s in seeds]
                + [dict(model=m, seed=s, loss="focal") for m in SIDE_MODELS for s in seeds])
    if study == "S4":
        return [dict(model=m, seed=s, fraction=f) for m in S4_MODELS for f in S4_FRACTIONS for s in seeds]
    if study == "S5":
        # binary models; the 6-class S1 predictions collapsed to positive/negative are the free first step
        return [dict(model=m, seed=s, task="2") for m in S5_MODELS for s in seeds]
    if study == "S10":
        # dropout 0.2 / 0.5 and L2 1e-4 on all weights; "no regularisation" is S1
        variants = [dict(dropout=0.2), dict(dropout=0.5), dict(l2=1e-4)]
        return [dict(model=m, seed=s, **v) for m in SIDE_MODELS for v in variants for s in seeds]
    if study == "S11":
        # 2 layers; "1 layer" is S1
        return [dict(model=m, seed=s, layers=2) for m in ["conv1d", "lstm", "gru"] for s in seeds]
    if study == "S12":
        # learned 100-d vectors vs GloVe 100-d frozen vs GloVe 100-d fine-tuned vs GloVe frozen with the vocabulary
        # extended by the 40,000 most frequent GloVe words (words unseen in train get vectors too)
        variants = ["learned100", "glove-frozen", "glove-tuned", "glove-extended"]
        return [dict(model=m, seed=s, embedding=e) for m in SIDE_MODELS for e in variants for s in seeds]
    if study == "S15":
        # all 10 models trained on the Lithuanian translation, with their English T0 settings; English is S1
        return [dict(model=m, seed=s, lang="lt") for m in models for s in seeds]
    if study == "S16":
        # context-only reference: trained without the emotion word
        return [dict(model=m, seed=s, without_keyword=True) for m in S16_MODELS for s in seeds]
    if study == "S17":
        # trained on GoEmotions, tested on GoEmotions and CARER; CARER -> GoEmotions is S1
        return [dict(model=m, seed=s, corpus="goemotions") for m in S17_MODELS for s in seeds]
    raise ValueError(f"Unknown study: {study}")


def check_tuning(models):
    """S1 and later studies use the T0 choices, so they refuse to start until those are in config.py."""
    missing = ([f"MODEL_SIZES[{m}]" for m in models if train.MODELS[m][2] and m not in config.MODEL_SIZES]
               + [f"MODEL_OPTIMIZERS[{m}]" for m in models if m not in config.MODEL_OPTIMIZERS]
               + [f"MODEL_LEARNING_RATES[{m}]" for m in models if m not in config.MODEL_LEARNING_RATES])
    if missing:
        raise ValueError(f"Fill in the T0 choices in config.py first (step 2.4); missing: {', '.join(missing)}")


def run_key(study, model, variant, seed, optimizer, lr):
    """What makes a run unique; the learning rate as text, so 0.0003 from config and from the CSV compare equal."""
    return (study, model, variant, int(seed), optimizer, f"{float(lr):g}")


def done_runs(path):
    """run_key of every run already in results.csv."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return {run_key(*row) for row in zip(df["study"], df["model"], df["variant"], df["seed"], df["optimizer"], df["lr"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all runs of one study.")
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--dev", action="store_true", help="quick check of every run: 2,000 texts, 3 epochs, results/dev/")
    args = parser.parse_args()

    runs = [r for r in study_runs(args.study, args.dev) if r["model"] in train.MODELS]   # models not written yet are skipped
    out = os.path.join(config.RESULTS_DIR, "dev") if args.dev else config.RESULTS_DIR
    done = done_runs(os.path.join(out, "results.csv"))
    print(f"Tyrimas {args.study}: {len(runs)} paleidimų")

    for i, kwargs in enumerate(runs, 1):
        options = {k: v for k, v in kwargs.items() if k not in ("model", "seed")}
        o, variant = train.settings(kwargs["model"], **options)   # the same names and values train.run will write
        if run_key(args.study, kwargs["model"], variant, kwargs["seed"], o["optimizer"], o["lr"]) in done:
            print(f"[{i}/{len(runs)}] jau atlikta: {kwargs['model']} {variant} sėkla {kwargs['seed']}")
            continue
        print(f"[{i}/{len(runs)}]", end="")
        train.run(kwargs["model"], kwargs["seed"], args.study, args.dev, **options)
