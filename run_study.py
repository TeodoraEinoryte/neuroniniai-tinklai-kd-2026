"""Run all training runs of one study (PLAN.md, section 6).

Run: python run_study.py S1 [--dev]
Runs already in results.csv are skipped, so an interrupted study can simply be started again.
Each study changes one thing; everything else stays as in config.py.
"""
import argparse
import os

import pandas as pd

import config
import train

T0_SEEDS = [1, 2]
RNN_MODELS = ["simplernn", "lstm", "gru", "bilstm"]                  # S2: models with a hidden state
S4_FRACTIONS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]                    # 1k, 2k, 4k, 8k, 16k train texts
S4_MODELS = ["logreg"]       # + the best model of each family, added after S1 (PLAN.md, S4)
S9_MODELS = ["conv1d", "gru"]                    # best non-recurrent and best recurrent model of the pilot run
S9_LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]   # the same grid for every optimiser (S9a)
STUDIES = ["S9a", "S9", "T0", "S1", "S2", "S3", "S4", "S5"]


def study_runs(study):
    """All runs of a study as a list of keyword arguments for train.run."""
    models = list(train.MODELS)
    if study == "S9a":
        # every optimiser x 4 learning rates x 2 models, seed 1: finds each optimiser's best learning rate
        return [dict(model=m, seed=1, optimizer=o, lr=lr)
                for m in S9_MODELS for o in train.OPTIMIZERS for lr in S9_LEARNING_RATES]
    if study == "S9":
        # every optimiser at its best learning rate from S9a (config.S9_LEARNING_RATES) x 2 models x 5 seeds
        missing = [o for o in train.OPTIMIZERS if o not in config.S9_LEARNING_RATES]
        if missing:
            raise ValueError(f"Fill config.S9_LEARNING_RATES from the S9a results first; missing: {missing}")
        return [dict(model=m, seed=s, optimizer=o, lr=config.S9_LEARNING_RATES[o])
                for m in S9_MODELS for o in train.OPTIMIZERS for s in config.SEEDS]
    if study == "T0":
        # every tuned model x its 3 sizes x seeds 1-2; picked by validation macro-F1 (step 2.4)
        return [dict(model=m, seed=s, size=size)
                for m in models for size in train.MODELS[m][2] for s in T0_SEEDS]
    if study == "S1":
        return [dict(model=m, seed=s) for m in models for s in config.SEEDS]
    if study == "S2":
        # post padding without mask / pre padding without mask / with mask (post)
        variants = [dict(mask=False), dict(mask=False, padding="pre"), dict(mask=True)]
        return [dict(model=m, seed=s, **v) for m in RNN_MODELS if m in train.MODELS
                for v in variants for s in config.SEEDS]
    if study == "S3":
        return [dict(model=m, seed=s, class_weight=True) for m in models for s in config.SEEDS]
    if study == "S4":
        return [dict(model=m, seed=s, fraction=f) for m in S4_MODELS for f in S4_FRACTIONS for s in config.SEEDS]
    if study == "S5":
        return [dict(model=m, seed=s, task="2") for m in models for s in config.SEEDS]
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
        settings = {k: v for k, v in kwargs.items() if k != "seed"}
        variant = train.settings(**settings)[3]      # the same variant name train.run will write
        if (args.study, kwargs["model"], variant, kwargs["seed"]) in done:
            print(f"[{i}/{len(runs)}] jau atlikta: {kwargs['model']} {variant} sėkla {kwargs['seed']}")
            continue
        print(f"[{i}/{len(runs)}]", end="")
        train.run(study=args.study, dev=args.dev, **kwargs)
