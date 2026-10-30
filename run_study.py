"""Run all training runs of one study (PLAN.md, section 6).

Run: python run_study.py S1 [--dev]
Runs already in results.csv are skipped, so an interrupted study can simply be started again.
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
S4_FRACTIONS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]                    # 1k, 2k, 4k, 8k, 16k train texts
S4_MODELS = ["logreg"]       # + the best non-recurrent and the best recurrent model, added after S1 (PLAN.md, S4)
SIDE_MODELS = ["conv1d", "gru"]                  # S9, S10, S12, focal loss: best non-recurrent and recurrent of the pilot
S9_LEARNING_RATES = [1.0, 0.1, 0.01, 0.001, 0.0001]   # the same grid for every optimiser (S9a); 1.0 is Adadelta's usual rate
STUDIES = ["S9a", "S9", "T0a", "T0b", "S1", "S2", "S3", "S4", "S5", "S10", "S11", "S12"]


def study_runs(study):
    """All runs of a study as a list of keyword arguments for train.run."""
    models = list(train.MODELS)
    seeds = config.SEEDS
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
        # every tuned model x its 3 sizes x seeds 1-2, protocol learning rate; picked by validation macro-F1 (step 2.4)
        return [dict(model=m, seed=s, size=size)
                for m in models for size in train.MODELS[m][2] for s in T0_SEEDS]
    if study == "T0b":
        # every model at its T0a size x learning rates LEARNING_RATE x 0.3 and x 3 (x 1 is the T0a run) x seeds 1-2
        missing = [m for m in models if train.MODELS[m][2] and m not in config.MODEL_SIZES]
        if missing:
            raise ValueError(f"Fill config.MODEL_SIZES from the T0a results first; missing: {missing}")
        rates = [round(config.LEARNING_RATE * f, 10) for f in config.T0B_FACTORS]
        return [dict(model=m, seed=s, size=config.MODEL_SIZES.get(m), lr=lr)
                for m in models for lr in rates for s in T0_SEEDS]
    if study == "S1":
        return [dict(model=m, seed=s) for m in models for s in seeds]
    if study == "S2":
        # RNNs: post padding without mask / pre padding without mask; "with mask" is S1.
        # Transformer the other way round: its protocol is without the mask (config.NO_MASK_MODELS), so S2 adds "with mask"
        variants = [dict(mask=False), dict(mask=False, padding="pre")]
        return ([dict(model=m, seed=s, **v) for m in RNN_MODELS for v in variants for s in seeds]
                + [dict(model="transformer", seed=s, mask=True) for s in seeds])
    if study == "S3":
        # class weights for every model + focal loss for the two side models; "neither" is S1
        return ([dict(model=m, seed=s, class_weight=True) for m in models for s in seeds]
                + [dict(model=m, seed=s, loss="focal") for m in SIDE_MODELS for s in seeds])
    if study == "S4":
        return [dict(model=m, seed=s, fraction=f) for m in S4_MODELS for f in S4_FRACTIONS for s in seeds]
    if study == "S5":
        return [dict(model=m, seed=s, task="2") for m in models for s in seeds]
    if study == "S10":
        # dropout 0.2 / 0.5 and L2 1e-4; "no regularisation" is S1
        variants = [dict(dropout=0.2), dict(dropout=0.5), dict(l2=1e-4)]
        return [dict(model=m, seed=s, **v) for m in SIDE_MODELS for v in variants for s in seeds]
    if study == "S11":
        # 2 layers; "1 layer" is S1
        return [dict(model=m, seed=s, layers=2) for m in ["conv1d", "lstm", "gru"] for s in seeds]
    if study == "S12":
        # learned 100-d vectors vs GloVe 100-d frozen vs GloVe 100-d fine-tuned
        variants = ["learned100", "glove-frozen", "glove-tuned"]
        return [dict(model=m, seed=s, embedding=e) for m in SIDE_MODELS for e in variants for s in seeds]
    raise ValueError(f"Unknown study: {study}")


def done_runs(path):
    """(study, model, variant, seed) of every run already in results.csv."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["study"], df["model"], df["variant"], df["seed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all runs of one study.")
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--dev", action="store_true", help="quick check of every run: 2,000 texts, 3 epochs, results/dev/")
    args = parser.parse_args()

    runs = [r for r in study_runs(args.study) if r["model"] in train.MODELS]   # models not written yet are skipped
    out = os.path.join(config.RESULTS_DIR, "dev") if args.dev else config.RESULTS_DIR
    done = done_runs(os.path.join(out, "results.csv"))
    print(f"Tyrimas {args.study}: {len(runs)} paleidimų")

    for i, kwargs in enumerate(runs, 1):
        options = {k: v for k, v in kwargs.items() if k not in ("model", "seed")}
        variant = train.settings(kwargs["model"], **options)[1]   # the same variant name train.run will write
        if (args.study, kwargs["model"], variant, kwargs["seed"]) in done:
            print(f"[{i}/{len(runs)}] jau atlikta: {kwargs['model']} {variant} sėkla {kwargs['seed']}")
            continue
        print(f"[{i}/{len(runs)}]", end="")
        train.run(kwargs["model"], kwargs["seed"], args.study, args.dev, **options)
