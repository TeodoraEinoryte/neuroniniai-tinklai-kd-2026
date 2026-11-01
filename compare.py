"""Comparison tables and figures for report 4.2–4.7 and appendices 1–3 and 7 (PLAN.md, section 6; step 4.1).

Run: python compare.py [S9] [T0] [S1] [S2] [S3] [S10] [S11] [S12] [S4] [S5] [S47]   (no argument = all)
    S9   optimisers: the S9a heatmap (optimiser × learning rate) and the S9 table, report 4.2
    T0   tuning: chosen size, optimiser and rate per model (report 4.3), full T0a / T0b tables and budget (appendix 1)
    S1   main comparison with DistilBERT (S14) and the two baselines, McNemar, report 4.4; per-class F1 (appendix 7)
    S2   padding and the padding mask, accuracy by text length, the Transformer's mask arms, report 4.5
    S3   class weights and focal loss, per class, report 4.6
    S10  regularisation (appendix 2); S11 depth (appendix 3)
    S12  word vectors, GloVe (appendix 4); S4 train-set size with the keyword-rule curve (appendix 5);
    S5   two classes: S1 collapsed and the binary models (appendix 6)                    (S12, S4, S5: A)
    S47  the 4.7 summary: each side study's best setting against S1
Nothing is trained: everything comes from results.csv and the saved predictions. Tables go to results/tables/
(Lithuanian headers, decimal comma, "mean ± sd" over the seeds), figures to results/figures/. All parts take ~8 min,
most of it the bootstraps (S1 alone ~4 min: its groups need several rounds).

Statistics (PLAN.md section 6): a difference between two models is a paired bootstrap on the extended test
(50,000 texts, evaluation.bootstrap_diff: 1,000 draws, seeds averaged), Holm-corrected over the comparisons of one
table; McNemar per seed on the official test. The wording rule is in verdict().
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import evaluation
from data import keyword_groups, load_carer_extra, load_emotion
from data_analysis import keyword_rule
from generalization import FIGURES_DIR, TABLES_DIR, comma_axis, interval, mean_sd, runs, save

RESULTS = pd.read_csv(os.path.join(config.RESULTS_DIR, "results.csv"))
MODELS = [m for m in config.MODEL_NAMES_LT if m not in ("distilbert", "distilbert_multi")]   # our 10, in order
SIDE_MODELS = ["conv1d", "gru"]
RNN_MODELS = ["simplernn", "lstm", "gru", "bilstm"]
OPTIMIZERS_LT = {"sgd": "SGD", "momentum": "SGD + Momentum", "nesterov": "SGD + Nesterov", "adagrad": "Adagrad",
                 "adadelta": "Adadelta", "rmsprop": "RMSprop", "adam": "Adam", "nadam": "Nadam", "adamw": "AdamW"}
EQUAL_MARGIN = 0.01        # "practically equal": the whole 95 % interval within ±1 macro-F1 point
REFERENCE_LT = "etalonas"
DIFF_LT = "skirtumas išplėstinėje imtyje [95 % PI]"


# ---------- shared helpers ----------

def rows(study, model=None, variant=None, seeds=None):
    """The results.csv rows of a study (optionally one model, one variant, some seeds)."""
    d = RESULTS[RESULTS["study"] == study]
    if model is not None:
        d = d[d["model"] == model]
    if variant is not None:
        d = d[d["variant"] == variant]
    if seeds is not None:
        d = d[d["seed"].isin(seeds)]
    return d


def predictions(study, model, variant=None, key="p_xtest", seeds=None):
    """One prediction array per seed (the model's first variant unless one is given), [] if the run is missing."""
    for (m, v), seed_runs in runs(study).items():
        if m == model and (variant is None or v == variant):
            return [z[key] for s, z in seed_runs if seeds is None or s in seeds]
    return []


def s1_base(model):
    """S1 as the baseline of a side study: only seeds 1–5, the seeds the side studies ran with (S1 has 3 more
    for its top group, config.S1_EXTRA_SEEDS), so both sides of a comparison have the same seeds."""
    return predictions("S1", model, seeds=config.SEEDS)


def rate(lr):
    """0.001 -> "0,001": a learning rate without an exponent, decimal comma."""
    return np.format_float_positional(lr, trim="-").replace(".", ",")


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def p_text(p):
    """The bootstrap's p with 1,000 draws cannot be smaller than 0.002, so 0 is written as "< 0,002"."""
    return "< 0,002" if p == 0 else number(p)


def verdict(low, high, significant):
    """The wording rule of PLAN.md section 6, for the difference model − reference and its 95 % interval:
    "practically equal" when the whole interval lies within ±0.01 (even if the difference is real, it is
    negligible: then it says so); "better" / "worse" when the interval excludes 0 and survives Holm;
    otherwise "no clear difference"."""
    real = significant and (low > 0 or high < 0)
    direction = "geresnis" if low > 0 else "blogesnis"
    if -EQUAL_MARGIN <= low and high <= EQUAL_MARGIN:
        return f"praktiškai lygus, nors reikšmingai {direction}" if real else "praktiškai lygus"
    return direction if real else "aiškaus skirtumo nėra"


def against(comparisons):
    """Bootstrap + Holm for one table. comparisons: {name: (preds, reference preds)} on the extended test.
    Returns {name: (diff, low, high, p, Holm-significant)}."""
    y = y_xtest()
    tests = {name: evaluation.bootstrap_diff(y, preds, reference) for name, (preds, reference) in comparisons.items()}
    significant = evaluation.holm([t[3] for t in tests.values()])
    return {name: (*t, s) for (name, t), s in zip(tests.items(), significant)}


def y_xtest():
    return runs("S1")[next(iter(runs("S1")))][0][1]["y_xtest"]


def diff_columns(row, test):
    """The difference, its p and the verdict as table columns."""
    diff, low, high, p, significant = test
    row[DIFF_LT] = interval(diff, low, high)
    row["p"] = p_text(p)
    row["išvada"] = verdict(low, high, significant)
    return row


def macro_f1s(y, preds):
    return [evaluation.metrics(y, p, len(config.CLASSES_EN))["macro_f1"] for p in preds]


# ---------- S9: optimisers (report 4.2) ----------

def s9():
    grid = rows("S9a")
    optimizers = list(OPTIMIZERS_LT)
    rates = sorted(grid["lr"].unique(), reverse=True)                # 1 ... 0.0001
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    table = {}
    for ax, model in zip(axes, SIDE_MODELS):
        values = (grid[grid["model"] == model].pivot_table(index="optimizer", columns="lr", values="val_macro_f1")
                  .reindex(index=optimizers, columns=rates))
        image = ax.imshow(values.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i, j in np.ndindex(values.shape):
            v = values.iloc[i, j]
            ax.text(j, i, "–" if np.isnan(v) else number(v, 2), ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
        ax.set_xticks(range(len(rates)), [rate(r) for r in rates], fontsize=9)
        ax.set_yticks(range(len(optimizers)), [OPTIMIZERS_LT[o] for o in optimizers] if model == SIDE_MODELS[0] else [])
        ax.set_xlabel("Mokymosi greitis η")
        ax.set_title(config.MODEL_NAMES_LT[model])
        for r in rates:
            table[f"{config.MODEL_NAMES_LT[model]}, η = {rate(r)}"] = [number(v) for v in values[r]]
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85, label="Validavimo makro F-matas (1 sėkla)")
    comma_axis(colorbar.ax.yaxis)
    path = os.path.join(FIGURES_DIR, "s9a_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)
    save(pd.DataFrame(table, index=[OPTIMIZERS_LT[o] for o in optimizers]), "s9a_grid.csv",
         "optimizatorius (validavimo makro F-matas, 1 sėkla)")

    # S9: each optimiser at its best S9a rate, 5 seeds; validation (where S9 chose) vs test (where it did not hold)
    s9_rows = rows("S9")
    table = {}
    for o in optimizers:
        r = s9_rows[s9_rows["optimizer"] == o]
        assert r["lr"].nunique() == 1, o
        row = {"mokymosi greitis (geriausias S9a)": rate(r["lr"].iloc[0])}
        for model in SIDE_MODELS:
            m = r[r["model"] == model]
            row[f"{config.MODEL_NAMES_LT[model]}: validavimo makro F-matas"] = mean_sd(m["val_macro_f1"])
            row[f"{config.MODEL_NAMES_LT[model]}: testavimo makro F-matas"] = mean_sd(m["test_macro_f1"])
        table[OPTIMIZERS_LT[o]] = row
    table = pd.DataFrame(table).T
    print(table.to_string())
    save(table, "s9_optimizers.csv", "optimizatorius (5 sėklos)")


# ---------- T0: tuning (report 4.3, appendix 1) ----------

def t0_setting(model, size, optimizer, lr):
    """Mean selection-set macro-F1 (seeds 1–2) of one T0 setting: Adam 0.001 is the T0a run, the rest T0b."""
    study = "T0a" if (optimizer, lr) == ("adam", 0.001) else "T0b"
    r = rows(study, model)
    r = r[(r["optimizer"] == optimizer) & np.isclose(r["lr"], lr)]
    if size is not None:
        r = r[r["variant"].str.startswith(f"size{size}-")]
    return r["sel_macro_f1"].mean() if len(r) else np.nan


def t0():
    choices, sizes, pairs, budget = {}, {}, {}, {}
    pair_list = sorted({(r.optimizer, r.lr) for r in rows("T0b").itertuples()} | {("adam", 0.001)})
    for model in MODELS:
        name = config.MODEL_NAMES_LT[model]
        size = config.MODEL_SIZES.get(model)
        optimizer, lr = config.MODEL_OPTIMIZERS[model], config.MODEL_LEARNING_RATES[model]
        chosen, default = t0_setting(model, size, optimizer, lr), t0_setting(model, size, "adam", 0.001)
        choices[name] = {"dydis": size or "–", "optimizatorius": OPTIMIZERS_LT[optimizer],
                         "mokymosi greitis": rate(lr), "parinkimo imtis: pasirinktas": number(chosen),
                         "parinkimo imtis: Adam 0,001": number(default), "skirtumas": number(chosen - default)}
        a = rows("T0a", model).groupby("variant")["sel_macro_f1"].mean()
        sizes[name] = {int(v.split("-")[0][4:]): number(f) for v, f in a.items() if v.startswith("size")}
        sizes[name]["pasirinktas dydis"] = size or "–"
        pairs[name] = {f"{OPTIMIZERS_LT[o]} {rate(r)}": number(t0_setting(model, size, o, r)) for o, r in pair_list}
        pairs[name]["pasirinkta"] = f"{OPTIMIZERS_LT[optimizer]} {rate(lr)}"
        a_runs, b_runs = rows("T0a", model), rows("T0b", model)
        budget[name] = {"T0a paleidimų": len(a_runs), "T0b paleidimų": len(b_runs),
                        "T0a mokymo laikas, min": round(a_runs["train_time_s"].sum() / 60, 1),
                        "T0b mokymo laikas, min": round(b_runs["train_time_s"].sum() / 60, 1)}
    choices = pd.DataFrame(choices).T
    print(choices.to_string())
    save(choices, "t0_choices.csv", "modelis (parinkimo imtis, makro F-matas, 1–2 sėklos)")
    sizes = pd.DataFrame(sizes).T.drop(index=config.MODEL_NAMES_LT["logreg"])     # logreg has no size to tune
    sizes = sizes[sorted(c for c in sizes.columns if isinstance(c, int)) + ["pasirinktas dydis"]].fillna("–")
    save(sizes, "t0a_sizes.csv", "modelis (dydis; parinkimo imtis, makro F-matas, 1–2 sėklos, Adam 0,001)")
    pairs = pd.DataFrame(pairs).T.replace("nan", "–")
    save(pairs, "t0b_rates.csv", "modelis (parinkimo imtis, makro F-matas, 1–2 sėklos)")
    budget = pd.DataFrame(budget).T
    budget.loc["Iš viso"] = budget.sum()
    budget = budget.astype({"T0a paleidimų": int, "T0b paleidimų": int})
    print(budget.to_string())
    save(budget.map(lambda v: str(v).replace(".", ",")), "t0_budget.csv", "modelis")


# ---------- S1: main comparison (report 4.4, appendix 7) ----------

def groups(candidates, sel, first_tests, reference):
    """Groups of models that cannot be told apart (PLAN.md section 6). Round 1: the reference and every model whose
    difference from it is not significant after Holm (first_tests, the main table's own tests). Then the best
    remaining model by selection-set macro-F1 becomes the next reference, and so on."""
    group, left, k = {}, list(candidates), 1
    while left:
        if k == 1:
            ref, tests = reference, first_tests
        else:
            ref = max(left, key=lambda n: sel[n])
            tests = against({n: (candidates[n], candidates[ref]) for n in left if n != ref})
        members = [ref] + [n for n in left if n != ref and not tests[n][4]]
        for n in members:
            group[n] = k
        left = [n for n in left if n not in members]
        k += 1
    return group


def mcnemar_table(seed_runs, reference):
    """McNemar per seed on the official test against the reference with the same seed; Holm over the models within
    each seed. A difference counts if it is significant in at least 4 of 5 seeds (all 3 for DistilBERT)."""
    ref = dict(seed_runs[reference])
    tests = {}                                             # seed -> [(model, b, c, p)]
    for name, model_runs in seed_runs.items():
        if name == reference:
            continue
        for seed, z in model_runs:
            y, p, p_ref = z["y_test"], z["p_test"], ref[seed]["p_test"]
            b = int(np.sum((p == y) & (p_ref != y)))       # this model right, the reference wrong
            c = int(np.sum((p != y) & (p_ref == y)))       # the other way round
            tests.setdefault(seed, []).append((name, b, c, evaluation.mcnemar(y, p, p_ref)[1]))
    table, hits = {}, {}
    for seed, seed_tests in tests.items():
        for (name, b, c, p), significant in zip(seed_tests, evaluation.holm([t[3] for t in seed_tests])):
            table.setdefault(name, {})[f"{seed} sėkla: b / c; p"] = f"{b} / {c}; {'< 0,001' if p < 0.001 else number(p)}" + ("*" if significant else "")
            hits.setdefault(name, []).append(significant)
    for name, h in hits.items():                           # 4 of 5 seeds = 80 %; DistilBERT has 3 seeds
        table[name]["reikšminga po Holm"] = f"{sum(h)} iš {len(h)}"
        table[name]["išvada"] = "skiriasi" if sum(h) >= 0.8 * len(h) else "nesiskiria"
    columns = [f"{seed} sėkla: b / c; p" for seed in sorted(tests)] + ["reikšminga po Holm", "išvada"]
    return pd.DataFrame(table).T[columns].fillna("–")


def s1():
    d = load_emotion()
    x_xtest, y_text = load_carer_extra("xtest")
    y_x, y_t = y_xtest(), d["y_test"]
    assert (y_text == y_x).all()
    seed_runs = {config.MODEL_NAMES_LT[m]: r for (m, _), r in runs("S1").items()}
    seed_runs.update({config.MODEL_NAMES_LT[m]: r for (m, _), r in runs("S14").items()})
    trained = {name: [z["p_xtest"] for _, z in r] for name, r in seed_runs.items()}
    xtest_mean = {name: np.mean(macro_f1s(y_x, p)) for name, p in trained.items()}
    sel, setting = {}, {}
    for study in ["S1", "S14"]:
        for model, r in rows(study).groupby("model", sort=False):
            name = config.MODEL_NAMES_LT[model]
            sel[name] = r["sel_macro_f1"].mean()
            setting[name] = f"{OPTIMIZERS_LT[r['optimizer'].iloc[0]]}, {rate(r['lr'].iloc[0])}"
    ours = [config.MODEL_NAMES_LT[m] for m in MODELS]
    reference = max(ours, key=lambda n: sel[n])      # the best of our models on the selection set
    print(f"Etalonas (geriausias parinkimo imtyje): {reference}, {number(sel[reference], 4)}")

    rule = keyword_rule(d["x_train"], d["y_train"])
    majority = np.bincount(d["y_train"]).argmax()
    rule_lt, majority_lt = "Raktinio žodžio taisyklė", f"Dažniausia klasė („{config.CLASSES_LT[majority]}“)"
    baselines = {rule_lt: (rule(d["x_test"]), rule(x_xtest)),
                 majority_lt: (np.full(len(y_t), majority), np.full(len(y_x), majority))}

    comparisons = {n: (p, trained[reference]) for n, p in trained.items() if n != reference}
    comparisons.update({n: ([px], trained[reference]) for n, (_, px) in baselines.items()})
    tests = against(comparisons)
    group = groups(trained, sel, tests, reference)

    table = {}
    for name in trained:
        test_preds = [z["p_test"] for _, z in seed_runs[name]]
        row = {"optimizatorius, mokymosi greitis": setting[name], "sėklų": len(test_preds),
               "parinkimo imtis": number(sel[name]),
               "testavimo imtis: bendras tikslumas, %": mean_sd([(p == y_t).mean() for p in test_preds], 100, 1),
               "testavimo imtis: makro F-matas": mean_sd(macro_f1s(y_t, test_preds)),
               "išplėstinė imtis: makro F-matas": mean_sd(macro_f1s(y_x, trained[name]))}
        if name == reference:
            row.update({DIFF_LT: REFERENCE_LT, "p": "–", "išvada": REFERENCE_LT})
        else:
            diff_columns(row, tests[name])
        row["grupė"] = group[name]
        table[name] = row
    for name, (pt, px) in baselines.items():
        row = {"optimizatorius, mokymosi greitis": "–", "sėklų": "–", "parinkimo imtis": "–",
               "testavimo imtis: bendras tikslumas, %": number((pt == y_t).mean() * 100, 1),
               "testavimo imtis: makro F-matas": number(macro_f1s(y_t, [pt])[0]),
               "išplėstinė imtis: makro F-matas": number(macro_f1s(y_x, [px])[0])}
        table[name] = diff_columns(row, tests[name]) | {"grupė": "–"}
    table = pd.DataFrame(table).T
    table = table.loc[sorted(trained, key=lambda n: (group[n], -xtest_mean[n])) + list(baselines)]
    print(table.to_string())
    save(table, "s1_main.csv", "modelis")
    top = [xtest_mean[n] for n in trained if group[n] == 1]
    rule = ("" if max(top) - min(top) > 0.005 else " (≤ 0,005: 6 skyriaus taisyklė jau pritaikyta, config.S1_EXTRA_SEEDS)"
            if config.S1_EXTRA_SEEDS else " (≤ 0,005: 6 skyriaus taisyklė siūlo +3 sėklas)")
    print(f"1 grupė: {', '.join(n for n in trained if group[n] == 1)}; išplėstinės imties vidurkiai skiriasi iki "
          f"{number(max(top) - min(top), 4)}{rule}")

    mcnemar = mcnemar_table(seed_runs, reference)
    print(mcnemar.to_string())
    save(mcnemar, "s1_mcnemar.csv", f"modelis (palyginti su {reference}; testavimo imtis)")

    shares = np.bincount(d["y_train"]) / len(d["y_train"])
    columns = [f"{c} ({number(s * 100, 1)} %)" for c, s in zip(config.CLASSES_LT, shares)]
    per_class = {}
    for name, preds in list(trained.items()) + [(rule_lt, [baselines[rule_lt][1]])]:
        f1 = np.array([evaluation.metrics(y_x, p, len(config.CLASSES_EN))["f1"] for p in preds])
        per_class[name] = {c: mean_sd(f1[:, i]) for i, c in enumerate(columns)}
        per_class[name]["makro F-matas"] = mean_sd(f1.mean(axis=1))
    save(pd.DataFrame(per_class).T, "s1_per_class.csv", "modelis (F-matas išplėstinėje imtyje; klasės dalis mokymo imtyje)")


# ---------- S2: padding and the padding mask (report 4.5) ----------

MASK_ARMS_LT = {"S1": "su kauke (S1)", "nomask": "be kaukės, užpildymas po teksto",
                "nomask-pre": "be kaukės, užpildymas prieš tekstą"}
LENGTH_BINS = [(1, 10), (11, 20), (21, 35), (36, 1000)]


def s2_arms(model):
    """{arm: predictions per seed} of one RNN: S1 (post padding with the mask) and the two S2 arms without it."""
    variant = rows("S1", model)["variant"].iloc[0]
    return {"S1": s1_base(model), "nomask": predictions("S2", model, f"{variant}-nomask"),
            "nomask-pre": predictions("S2", model, f"{variant}-nomask-pre")}


def s2():
    y = y_xtest()
    arms = {model: s2_arms(model) for model in RNN_MODELS}
    tests = against({(model, arm): (a[arm], a["S1"]) for model, a in arms.items() for arm in ["nomask", "nomask-pre"]})
    table = {}
    for model, a in arms.items():
        row = {f"{MASK_ARMS_LT[arm]}: makro F-matas": mean_sd(macro_f1s(y, a[arm])) for arm in MASK_ARMS_LT}
        for arm in ["nomask", "nomask-pre"]:
            diff, low, high, p, significant = tests[(model, arm)]
            row[f"{MASK_ARMS_LT[arm]} − S1 [95 % PI]"] = interval(diff, low, high)
            row[f"{MASK_ARMS_LT[arm]}: išvada"] = verdict(low, high, significant)
        table[config.MODEL_NAMES_LT[model]] = row
    table = pd.DataFrame(table).T
    print(table.to_string())
    save(table, "s2_masking.csv", "modelis (išplėstinė imtis)")

    # accuracy by text length on the extended test
    x_xtest, _ = load_carer_extra("xtest")
    words = np.array([len(t.split()) for t in x_xtest])
    bins = [(words >= lo) & (words <= hi) for lo, hi in LENGTH_BINS]
    labels = [f"{lo}–{hi}" if hi < 1000 else f"> {lo - 1}" for lo, hi in LENGTH_BINS]
    labels = [f"{label}\n(n = {m.sum()})" for label, m in zip(labels, bins)]
    fig, axes = plt.subplots(1, len(RNN_MODELS), figsize=(13, 3.8), sharey=True, constrained_layout=True)
    length_table = {}
    for ax, (model, a) in zip(axes, arms.items()):
        for arm, preds in a.items():
            accuracy = [np.mean([(p[m] == y[m]).mean() for p in preds]) * 100 for m in bins]
            ax.plot(range(len(bins)), accuracy, marker="o", label=MASK_ARMS_LT[arm])
            length_table[f"{config.MODEL_NAMES_LT[model]}: {MASK_ARMS_LT[arm]}"] = [number(v, 1) for v in accuracy]
        ax.set_xticks(range(len(bins)), labels, fontsize=8)
        ax.set_title(config.MODEL_NAMES_LT[model])
        ax.set_xlabel("Teksto ilgis, žodžiai")
        ax.grid(alpha=0.3)
        comma_axis(ax.yaxis)
    axes[0].set_ylabel("Bendras tikslumas, %")
    axes[0].legend(fontsize=8, loc="lower left")
    path = os.path.join(FIGURES_DIR, "s2_length.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)
    save(pd.DataFrame(length_table, index=[f"{lo}–{hi}" if hi < 1000 else f"> {lo - 1}" for lo, hi in LENGTH_BINS]).T,
         "s2_length.csv", "modelis ir užpildymas (bendras tikslumas išplėstinėje imtyje, %, pagal teksto ilgį žodžiais)")

    # the Transformer the other way round: its protocol is without the mask
    lr = config.S2_MASKED_TRANSFORMER_LR
    arms = {"be kaukės (S1)": ("S1", "size128", config.MODEL_LEARNING_RATES["transformer"]),
            "su kauke": ("S2", f"size128-mask-adam-lr{lr:g}", lr),
            "su kauke, Adam 0,001": ("S2", "size128-mask-adam-lr0.001", 0.001),
            "tik dėmesio kaukė": ("S2", "size128-attentionmask", config.MODEL_LEARNING_RATES["transformer"]),
            "tik vidurkinimo kaukė": ("S2", "size128-poolingmask", config.MODEL_LEARNING_RATES["transformer"])}
    preds = {arm: predictions(study, "transformer", variant) for arm, (study, variant, _) in arms.items()}
    tests = against({arm: (p, preds["be kaukės (S1)"]) for arm, p in preds.items() if arm != "be kaukės (S1)"})
    table = {}
    for arm, (study, variant, arm_lr) in arms.items():
        r = rows(study, "transformer", variant)
        row = {"mokymosi greitis (Adam)": rate(arm_lr), "sėklų": len(r),
               "parinkimo imtis": mean_sd(r["sel_macro_f1"]), "testavimo imtis": mean_sd(r["test_macro_f1"]),
               "išplėstinė imtis": mean_sd(macro_f1s(y, preds[arm]))}
        table[arm] = (row | {DIFF_LT: REFERENCE_LT, "p": "–", "išvada": REFERENCE_LT} if study == "S1"
                      else diff_columns(row, tests[arm]))
    table = pd.DataFrame(table).T
    print(table.to_string())
    save(table, "s2_transformer.csv", "transformeris (makro F-matas)")
    search = rows("S2", "transformer")
    search = search[search["variant"].str.contains("-mask-adam-")]
    search = search[search["seed"].isin([1, 2])].groupby("lr")
    search = pd.DataFrame({"parinkimo imtis, makro F-matas (1–2 sėklos)": search["sel_macro_f1"].mean().map(number),
                           "geriausia epocha": search["best_epoch"].mean().map(lambda v: number(v, 1))})
    search.index = [rate(lr) for lr in search.index]
    save(search, "s2_transformer_rates.csv", "mokymosi greitis (Adam), transformeris su kauke")


# ---------- S3: uneven classes (report 4.6) ----------

def s3():
    y = y_xtest()
    weights = {m: (predictions("S3", m, rows("S1", m)["variant"].iloc[0] + "-cw"), s1_base(m)) for m in MODELS}
    focal = {m: (predictions("S3", m, rows("S1", m)["variant"].iloc[0] + "-focal"), s1_base(m))
             for m in SIDE_MODELS}
    tests = against({**{("cw", m): v for m, v in weights.items()}, **{("focal", m): v for m, v in focal.items()}})
    table = {}
    for m in MODELS:
        row = {"S1": mean_sd(macro_f1s(y, weights[m][1])), "su klasių svoriais": mean_sd(macro_f1s(y, weights[m][0]))}
        diff, low, high, _, significant = tests[("cw", m)]
        row["klasių svoriai − S1 [95 % PI]"] = interval(diff, low, high)
        row["klasių svoriai: išvada"] = verdict(low, high, significant)
        if m in focal:
            diff, low, high, _, significant = tests[("focal", m)]
            row["su focal loss"] = mean_sd(macro_f1s(y, focal[m][0]))
            row["focal loss − S1 [95 % PI]"] = interval(diff, low, high)
            row["focal loss: išvada"] = verdict(low, high, significant)
        table[config.MODEL_NAMES_LT[m]] = row
    table = pd.DataFrame(table).T.fillna("–")
    print(table.to_string())
    save(table, "s3_class_weights.csv", "modelis (makro F-matas išplėstinėje imtyje)")

    # per-class F1: the mean over the 10 models, and the two side models with each loss
    def f1(preds):
        return np.mean([evaluation.metrics(y, p, len(config.CLASSES_EN))["f1"] for p in preds], axis=0)

    d = load_emotion()
    shares = np.bincount(d["y_train"]) / len(d["y_train"])
    columns = {"10 modelių vidurkis: S1": np.mean([f1(weights[m][1]) for m in MODELS], axis=0),
               "10 modelių vidurkis: su klasių svoriais": np.mean([f1(weights[m][0]) for m in MODELS], axis=0)}
    for m in SIDE_MODELS:
        name = config.MODEL_NAMES_LT[m]
        columns.update({f"{name}: S1": f1(weights[m][1]), f"{name}: su klasių svoriais": f1(weights[m][0]),
                        f"{name}: su focal loss": f1(focal[m][0])})
    per_class = pd.DataFrame({c: [number(v) for v in f] + [number(f.mean())] for c, f in columns.items()},
                             index=[f"{c} ({number(s * 100, 1)} %)" for c, s in zip(config.CLASSES_LT, shares)]
                             + ["makro F-matas"])
    print(per_class.to_string())
    save(per_class, "s3_per_class.csv", "klasė (dalis mokymo imtyje; F-matas išplėstinėje imtyje)")


# ---------- S10, S11: regularisation and depth (appendices 2 and 3) ----------

S10_ARMS_LT = {"do0.2": "Dropout 0,2", "do0.5": "Dropout 0,5", "l20.0001": "L2 0,0001"}


def side_arms(study, models, arms_lt):
    """{(model, arm label): (rows, predictions)} for S1 and every arm of a side study."""
    found = {}
    for m in models:
        variant = rows("S1", m)["variant"].iloc[0]
        found[(m, "S1")] = (rows("S1", m, seeds=config.SEEDS), s1_base(m))
        for suffix, label in arms_lt.items():
            found[(m, label)] = (rows(study, m, f"{variant}-{suffix}"), predictions(study, m, f"{variant}-{suffix}"))
    return found


def side_table(study, models, arms_lt, extra):
    """One appendix table: every arm against S1 of the same model, Holm over the table."""
    y = y_xtest()
    arms = side_arms(study, models, arms_lt)
    tests = against({k: (p, arms[(k[0], "S1")][1]) for k, (_, p) in arms.items() if k[1] != "S1"})
    table = {}
    for (m, label), (r, p) in arms.items():
        row = {"parinkimo imtis": mean_sd(r["sel_macro_f1"]), "išplėstinė imtis": mean_sd(macro_f1s(y, p))}
        row = row | {DIFF_LT: "S1", "p": "–", "išvada": "S1"} if label == "S1" else diff_columns(row, tests[(m, label)])
        table[(config.MODEL_NAMES_LT[m], label)] = row | extra(r)
    table = pd.DataFrame(table).T
    table.index = [f"{m}: {label}" for m, label in table.index]
    print(table.to_string())
    return table, tests


def s10():
    def extra(r):
        return {"mokymo tikslumas, %": mean_sd(r["train_acc"], 100, 1),
                "validavimo tikslumas, %": mean_sd(r["val_acc"], 100, 1),
                "tarpas, proc. punktai": mean_sd(r["train_acc"] - r["val_acc"], 100, 1),
                "geriausia epocha": number(r["best_epoch"].mean(), 1),
                "mokymo laikas, s": number(r["train_time_s"].mean(), 0)}

    table, _ = side_table("S10", SIDE_MODELS, S10_ARMS_LT, extra)
    save(table, "s10_regularisation.csv", "modelis: reguliarizacija (makro F-matas; 5 sėklos)")


def s11():
    def extra(r):
        return {"parametrų": f"{int(r['params'].iloc[0]):,}".replace(",", " "),
                "tarpas, proc. punktai": mean_sd(r["train_acc"] - r["val_acc"], 100, 1),
                "mokymo laikas, s": number(r["train_time_s"].mean(), 0)}

    table, _ = side_table("S11", ["conv1d", "lstm", "gru"], {"layers2": "2 sluoksniai"}, extra)
    table = table.rename(index=lambda i: i.replace(": S1", ": 1 sluoksnis (S1)"))
    save(table, "s11_depth.csv", "modelis: sluoksnių (makro F-matas; 5 sėklos)")


# ---------- S12, S4, S5: word vectors, train-set size, two classes (appendices 4–6; A) ----------

S12_ARMS_LT = {"learned100": "išmokti, 100-mačiai", "glove-frozen": "GloVe, nekeičiami",
               "glove-tuned": "GloVe, derinami", "glove-extended": "GloVe, nekeičiami, išplėstas žodynas"}
S12_LT = "S12 žodžių vektoriai"
S5_LT = "S5 dvi klasės"


def f1s(y, preds, n_classes):
    """macro_f1s() for any number of classes (S5 has 2)."""
    return [evaluation.metrics(y, p, n_classes)["macro_f1"] for p in preds]


def s12():
    """S12, pretrained word vectors (appendix 4): every arm against S1 (learned 64-d vectors) on the extended test,
    and the two places where pretrained vectors could help most: all 6,535 GoEmotions texts (another domain; 7.9 %
    of its word occurrences are outside the train vocabulary), with its own paired bootstrap and Holm, and the
    test texts whose emotion word never is one in train (keyword group "unseen", accuracy). Coverage of GloVe:
    `python data_analysis.py glove`."""
    def extra(r):
        return {"geriausia epocha": number(r["best_epoch"].mean(), 1)}

    table, _ = side_table("S12", SIDE_MODELS, S12_ARMS_LT, extra)
    d = load_emotion()
    unseen = keyword_groups(d["x_test"], d["x_train"], d["y_train"]) == "unseen"
    y_go = runs("S1")[next(iter(runs("S1")))][0][1]["y_go"]
    go, test = {}, {}                             # (model, arm label) -> one prediction array per seed
    for m in SIDE_MODELS:
        variant = rows("S1", m)["variant"].iloc[0]
        go[(m, "S1")] = predictions("S1", m, key="p_go", seeds=config.SEEDS)
        test[(m, "S1")] = predictions("S1", m, key="p_test", seeds=config.SEEDS)
        for suffix, label in S12_ARMS_LT.items():
            go[(m, label)] = predictions("S12", m, f"{variant}-{suffix}", key="p_go")
            test[(m, label)] = predictions("S12", m, f"{variant}-{suffix}", key="p_test")
    arms = [k for k in go if k[1] != "S1"]
    tests = [evaluation.bootstrap_diff(y_go, go[k], go[(k[0], "S1")]) for k in arms]
    significant = dict(zip(arms, evaluation.holm([t[3] for t in tests])))
    tests = dict(zip(arms, tests))
    columns = {"GoEmotions (6 535)": {}, "GoEmotions: skirtumas [95 % PI]": {}, "GoEmotions: išvada": {},
               f"nematytas emocijos žodis (n = {unseen.sum()}), %": {}}
    for (m, label) in go:
        name = f"{config.MODEL_NAMES_LT[m]}: {label}"
        values = [columns[c] for c in columns]
        values[0][name] = mean_sd(macro_f1s(y_go, go[(m, label)]))
        if label == "S1":
            values[1][name], values[2][name] = "S1", "S1"
        else:
            diff, low, high, _ = tests[(m, label)]
            values[1][name] = interval(diff, low, high)
            values[2][name] = verdict(low, high, significant[(m, label)])
        values[3][name] = mean_sd([(p[unseen] == d["y_test"][unseen]).mean() for p in test[(m, label)]], 100, 1)
    for column, values in columns.items():
        table[column] = pd.Series(values)
    table = table.rename(index=lambda i: i.replace(": S1", ": S1 (išmokti, 64-mačiai)"))
    print(table.to_string())
    save(table, "s12_word_vectors.csv", "modelis: žodžių vektoriai (makro F-matas; 5 sėklos)")


def _fraction(variant):
    """Share of train in an S4 variant name: "size128-frac0.25" -> 0.25, "size128" -> 1.0."""
    return float(variant.split("frac")[1]) if "frac" in variant else 1.0


def s4():
    """S4, train-set size (appendix 5): test macro-F1 of logreg, Conv1D and GRU trained on stratified subsets of
    1k–16k train texts (the same subset for every model of one seed), and the keyword rule rebuilt from each subset.
    S4 keeps no extended-test predictions (train.NO_XTEST_STUDIES), so it is read on the official test set."""
    d = load_emotion()
    s4_rows = rows("S4")
    models = [m for m in MODELS if m in set(s4_rows["model"])]
    fractions = sorted({_fraction(v) for v in s4_rows["variant"]})
    table, curves = {}, {}
    for f in fractions:
        n = len(load_emotion(fraction=f, seed=1)["y_train"])
        row = {}
        for m in models:
            r = s4_rows[(s4_rows["model"] == m) & (s4_rows["variant"].map(_fraction) == f)]
            row[config.MODEL_NAMES_LT[m]] = mean_sd(r["test_macro_f1"])
            curves.setdefault(config.MODEL_NAMES_LT[m], []).append((n, r["test_macro_f1"].mean(), r["test_macro_f1"].std()))
        rule = []
        for seed in config.SEEDS:
            subset = load_emotion(fraction=f, seed=seed)
            rule.append(macro_f1s(d["y_test"], [keyword_rule(subset["x_train"], subset["y_train"])(d["x_test"])])[0])
        row["Raktinio žodžio taisyklė"] = mean_sd(rule)
        curves.setdefault("Raktinio žodžio taisyklė", []).append((n, np.mean(rule), np.std(rule, ddof=1)))
        table[f"{n:,}".replace(",", " ")] = row
    table = pd.DataFrame(table).T
    print(table.to_string())
    save(table, "s4_train_size.csv", "mokymo tekstų (testavimo imties makro F-matas; 5 sėklos)")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, points in curves.items():
        n, mean, sd = map(np.array, zip(*points))
        style = {"linestyle": "--", "color": "gray"} if name.startswith("Raktinio") else {"marker": "o"}
        ax.errorbar(n, mean, yerr=sd, capsize=3, label=name, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [f"{v:,}".replace(",", " ") for v in n])
    ax.minorticks_off()
    ax.set_xlabel("Mokymo tekstų skaičius")
    ax.set_ylabel("Makro F-matas (testavimo imtis)")
    ax.set_ylim(0, 1)
    comma_axis(ax.yaxis)
    ax.set_title("Mokymo imties dydžio įtaka (vidurkis ± st. nuokrypis, 5 sėklos)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "s4_learning_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)


def to_binary(labels):
    """Six emotions -> 1 = positive (joy, love, surprise; config.POSITIVE_CLASSES), 0 = negative."""
    return np.isin(labels, config.POSITIVE_CLASSES).astype(int)


def s5_comparisons():
    """The S5 models trained on two classes against the six-class S1 model collapsed to two classes, on the
    extended test: [(model, collapsed S1 predictions, trained binary predictions)], seeds 1–5 on both sides."""
    found = []
    for m in MODELS:
        trained = predictions("S5", m)
        if trained:
            found.append((m, [to_binary(p) for p in s1_base(m)], trained))
    return found


def s5():
    """S5, two classes instead of six (appendix 6). Positive = joy, love, surprise; negative = sadness, anger, fear.
    First the free step: the six-class S1 predictions of every model collapsed to positive / negative, and the share
    of its six-class errors that stay on the right side. Then the binary models trained for logreg, Conv1D and GRU
    against their own collapsed S1 predictions on the extended test (binary macro-F1, paired bootstrap, Holm).
    Surprise counts as positive, which is debatable (PLAN.md section 3)."""
    y_test = runs("S1")[next(iter(runs("S1")))][0][1]["y_test"]
    yb_test, yb_x = to_binary(y_test), to_binary(y_xtest())
    comparisons = s5_comparisons()
    tests = [evaluation.bootstrap_diff(yb_x, trained, collapsed, n_classes=2) for _, collapsed, trained in comparisons]
    significant = evaluation.holm([t[3] for t in tests])
    tested = {m: (*t, s) for (m, _, _), t, s in zip(comparisons, tests, significant)}
    table = {}
    for m in MODELS:
        six = s1_base(m)
        test_preds = predictions("S1", m, key="p_test", seeds=config.SEEDS)
        errors = [p != y_test for p in test_preds]
        row = {"sutraukta: testavimo bendras tikslumas, %": mean_sd([(to_binary(p) == yb_test).mean() for p in test_preds], 100, 1),
               "sutraukta: išplėstinė imtis, makro F-matas": mean_sd(f1s(yb_x, [to_binary(p) for p in six], 2)),
               "6 klasių klaidos toje pačioje pusėje, %": mean_sd([(to_binary(p[e]) == to_binary(y_test[e])).mean()
                                                                   for p, e in zip(test_preds, errors)], 100, 1)}
        if m in tested:
            trained = predictions("S5", m)
            r = rows("S5", m)
            row["mokyta 2 klasėms: testavimo bendras tikslumas, %"] = mean_sd(r["test_acc"], 100, 1)
            row["mokyta 2 klasėms: išplėstinė imtis, makro F-matas"] = mean_sd(f1s(yb_x, trained, 2))
            diff, low, high, p, s = tested[m]
            row["mokyta − sutraukta [95 % PI]"] = interval(diff, low, high)
            row["išvada"] = verdict(low, high, s)
        table[config.MODEL_NAMES_LT[m]] = row
    table = pd.DataFrame(table).T.fillna("–")
    print(table.to_string())
    save(table, "s5_binary.csv", "modelis (2 klasės: teigiama / neigiama; 5 sėklos)")


# ---------- 4.7: summary of the side studies ----------

def summary_rows(study, models, arms_lt, study_lt):
    """4.7 rows of one side study: per model, its best arm by selection-set macro-F1 against S1.
    Returns [(row, (S1 predictions, arm predictions))]; s47() runs the tests, so Holm covers the whole table."""
    arms = side_arms(study, models, arms_lt)
    found = []
    for m in models:
        best = max(arms_lt.values(), key=lambda label: arms[(m, label)][0]["sel_macro_f1"].mean())
        found.append(({"tyrimas": study_lt, "modelis": config.MODEL_NAMES_LT[m], "geriausias nustatymas": best},
                      (arms[(m, "S1")][1], arms[(m, best)][1])))
    return found


def s47():
    """4.7 summary: the best setting of each side study against S1 on the extended test, Holm over the whole table.
    S5 is compared in two classes: the binary model against S1 collapsed to two classes (binary macro-F1).
    S4 is not here: it kept no extended-test predictions; its curve is in s4_train_size.csv (appendix 5)."""
    y = y_xtest()
    found = [(row, y, s1_preds, arm, 6) for row, (s1_preds, arm) in
             summary_rows("S10", SIDE_MODELS, S10_ARMS_LT, "S10 reguliarizacija")
             + summary_rows("S11", ["conv1d", "lstm", "gru"], {"layers2": "2 sluoksniai"}, "S11 gylis")
             + summary_rows("S12", SIDE_MODELS, S12_ARMS_LT, S12_LT)]
    found += [({"tyrimas": S5_LT, "modelis": config.MODEL_NAMES_LT[m],
                "geriausias nustatymas": "mokyta 2 klasėms (S1 sutrauktas į 2 klases)"}, to_binary(y), collapsed, trained, 2)
              for m, collapsed, trained in s5_comparisons()]
    tests = [evaluation.bootstrap_diff(yy, arm, s1_preds, n_classes=k) for _, yy, s1_preds, arm, k in found]
    significant = evaluation.holm([t[3] for t in tests])
    table = []
    for (row, yy, s1_preds, arm, k), t, s in zip(found, tests, significant):
        row = row | {"S1": mean_sd(f1s(yy, s1_preds, k)), "nustatymas": mean_sd(f1s(yy, arm, k))}
        table.append(diff_columns(row, (*t, s)))
    table = pd.DataFrame(table).set_index("tyrimas")
    print(table.to_string())
    save(table, "s47_summary.csv", "tyrimas (makro F-matas išplėstinėje imtyje; geriausias pagal parinkimo imtį)")


PARTS = {"S9": s9, "T0": t0, "S1": s1, "S2": s2, "S3": s3, "S10": s10, "S11": s11, "S12": s12, "S4": s4, "S5": s5,
         "S47": s47}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
