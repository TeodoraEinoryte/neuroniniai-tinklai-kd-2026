"""Generalisation and what the models learn: tables and figures for report 4.8 and 4.9.

Run: python generalization.py [S13] [S6]   (no argument = both)
    S13  keywords or context (with the context-only models of S16), report 4.8
    S6   another corpus and another language (S17 GoEmotions, S15 Lithuanian, S14b, S12 on GoEmotions), report 4.9
Nothing is trained: the tables come from results.csv and the saved predictions; the negation templates and the
attention checks also load the saved S1 models (results/models/, kept locally, not in git). Tables go to
results/tables/ (Lithuanian headers, decimal comma, "mean ± sd" over the seeds), figures to results/figures/.
Studies not run yet are simply left out of the tables.
"""
import functools
import os
import sys
from collections import Counter

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from tensorflow.keras.layers import TextVectorization

import config
import evaluation
import models_b                                   # registers AttentionPooling, so the attention model can be loaded
from data import (emotion_word, encode, keyword_groups, keyword_variants, load_emotion, load_goemotions,
                  load_goemotions_splits, negation_templates, vectorize)
from data_analysis import keyword_rule
from train import MODELS

TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(config.RESULTS_DIR, "predictions")
MODELS_DIR = os.path.join(config.RESULTS_DIR, "models")
N_CLASSES = 6
GROUPS_LT = {"pure": "vienareikšmis žodis", "ambiguous": "daugiareikšmis žodis", "none": "be emocijos žodžio",
             "unseen": "mokymo imtyje nematytas žodis"}
COPIES = {"original": ("p_test", "originalus tekstas"), "without": ("p_kw_without", "be emocijos žodžio"),
          "only": ("p_kw_only", "tik emocijos žodis"), "i_feel": ("p_kw_i_feel", "„i feel“ + emocijos žodis"),
          "control": ("p_kw_control", "kitas žodis pakeistas [UNK]")}
RULE_LT = "Raktinio žodžio taisyklė"


# ---------- shared helpers (also for the 4.9 parts) ----------

@functools.lru_cache(maxsize=None)
def runs(study):
    """{(model, variant): [(seed, predictions), ...]} for every run of a study in results.csv, in model order.
    Read once per study (the tables use the same runs several times); do not change the returned dict."""
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "results.csv"))
    order = list(config.MODEL_NAMES_LT)
    found = {}
    for row in sorted(df[df["study"] == study].itertuples(), key=lambda r: (order.index(r.model), r.variant, r.seed)):
        path = os.path.join(PREDICTIONS_DIR, f"{study}_{row.model}_{row.variant}_seed{row.seed}.npz")
        found.setdefault((row.model, row.variant), []).append((row.seed, np.load(path)))
    return found


def saved_seeds(study, model, variant, seed_runs):
    """The seeds of seed_runs whose saved model is on this machine (results/models/ is not in git: S1 seeds 6–8
    were trained on B's machine). Prints the missing ones, so a table never leaves them out silently."""
    present = [s for s, _ in seed_runs
               if os.path.exists(os.path.join(MODELS_DIR, f"{study}_{model}_{variant}_seed{s}.keras"))]
    missing = [s for s, _ in seed_runs if s not in present]
    if missing:
        print(f"Nėra išsaugoto modelio: {study} {model} sėklos {missing}; jos praleistos")
    return present


def five_seeds(seed_runs):
    """Only seeds 1–5 (config.SEEDS): S1 has 3 more for its top group (config.S1_EXTRA_SEEDS), but the studies it
    is compared with here (S12, S15, S17) have seeds 1–5, so both sides of a comparison get the same seeds."""
    return [(s, z) for s, z in seed_runs if s in config.SEEDS]


def load_saved(study, model, variant, seed):
    """A saved model (train.py saves the S1 and S15 ones) and its vectorizer, rebuilt from the saved vocabulary."""
    run_id = f"{study}_{model}_{variant}_seed{seed}"
    with open(os.path.join(MODELS_DIR, f"{run_id}_vocab.txt"), encoding="utf-8") as f:
        vocabulary = f.read().split("\n")
    assert vocabulary[:2] == ["", "[UNK]"], run_id
    vectorizer = TextVectorization(vocabulary=vocabulary[2:], ragged=True)   # Keras adds "" and "[UNK]" itself
    return keras.models.load_model(os.path.join(MODELS_DIR, f"{run_id}.keras")), vectorizer


def predict_texts(net, vectorizer, model, texts):
    """Class probabilities of a saved model for any cleaned texts."""
    return net.predict(encode(vectorizer, texts, MODELS[model][1]), batch_size=config.BATCH_SIZE, verbose=0)


def macro_f1(y, p):
    return evaluation.metrics(y, p, N_CLASSES)["macro_f1"]


def mean_sd(values, scale=1.0, digits=3):
    """"0,878 ± 0,006": mean and standard deviation over the seeds, decimal comma as in the report.
    A single value (one seed, or a rule) is written without ±."""
    v = np.asarray(values, dtype=float) * scale
    text = f"{v.mean():.{digits}f}" if len(v) == 1 else f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"
    return text.replace(".", ",")


def save(table, filename, index_label):
    path = os.path.join(TABLES_DIR, filename)
    table.to_csv(path, index_label=index_label, lineterminator="\n")
    print("Išsaugota:", path)


def comma_axis(axis):
    """Decimal comma on a matplotlib axis (ax.xaxis or ax.yaxis), as in the report text."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


# ---------- S13 (with S16): keywords or context? Report 4.8 ----------

def keyword_group_table(d):
    """Accuracy per keyword group of the test texts (data.keyword_groups; PLAN.md section 3) for the S1 models,
    DistilBERT (S14), the keyword rule and the majority class. Where the emotion word is pure every model is
    almost always right, so the architectures can differ only on the ambiguous and no-word groups."""
    y = d["y_test"]
    groups = keyword_groups(d["x_test"], d["x_train"], d["y_train"])
    masks = {"visi tekstai": np.ones(len(y), dtype=bool), **{lt: groups == g for g, lt in GROUPS_LT.items()}}

    def row(predictions):                         # one prediction array per seed
        r = {f"{name} (n = {m.sum()}), %": mean_sd([(p[m] == y[m]).mean() for p in predictions], 100, 1)
             for name, m in masks.items()}
        r["makro F-matas (visi tekstai)"] = mean_sd([macro_f1(y, p) for p in predictions])
        return r

    rows = {}
    for study in ["S1", "S14"]:
        for (model, _), seed_runs in runs(study).items():
            rows[config.MODEL_NAMES_LT[model]] = row([z["p_test"] for _, z in seed_runs])
    rows[RULE_LT] = row([keyword_rule(d["x_train"], d["y_train"])(d["x_test"])])
    majority = np.bincount(d["y_train"]).argmax()
    rows[f"Dažniausia klasė („{config.CLASSES_LT[majority]}“)"] = row([np.full(len(y), majority)])
    return pd.DataFrame(rows).T


GROUP_FIGURE_MODELS = ["logreg", "conv1d", "gru", "bilstm_attention", "distilbert"]   # a readable subset


def plot_keyword_groups(d, path):
    """Accuracy per keyword group for the keyword rule and a few models (all their seeds): where the emotion word is
    pure every model is right; the models differ, and beat the rule, only where it is ambiguous or missing."""
    y = d["y_test"]
    groups = keyword_groups(d["x_test"], d["x_train"], d["y_train"])
    series = {RULE_LT: [keyword_rule(d["x_train"], d["y_train"])(d["x_test"])]}
    for study in ["S1", "S14"]:
        for (model, _), seed_runs in runs(study).items():
            if model in GROUP_FIGURE_MODELS:
                series[config.MODEL_NAMES_LT[model]] = [z["p_test"] for _, z in seed_runs]
    x = np.arange(len(GROUPS_LT))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, predictions) in enumerate(series.items()):
        accuracy = [np.mean([(p[groups == g] == y[groups == g]).mean() for p in predictions]) * 100 for g in GROUPS_LT]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, accuracy, width=width, label=name,
               color="gray" if name == RULE_LT else None)
    ax.set_xticks(x, [f"{lt}\n(n = {(groups == g).sum()})" for g, lt in GROUPS_LT.items()])
    ax.set_ylabel("Bendras tikslumas, %")
    ax.set_ylim(0, 100)
    comma_axis(ax.yaxis)
    ax.set_title("Tikslumas pagal emocijos žodį (testavimo imtis)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)


def keyword_copies_table(d):
    """Macro-F1 on the keyword test texts for the original text and its copies (data.keyword_variants): S1 models,
    DistilBERT (S14), the context-only models (S16, trained without the emotion word) and the keyword rule.

    All rows use the same 1,459 texts: those whose emotion word is in the S1 vocabulary (S1 kw_has, equal in every
    S1 run). The rule finds the word after "feel", so its "only the word" value is scored on "i feel <word>"; on the
    "without" copy it falls back to the majority class.
    Returns the table and the means (rows x copies) for the figure.
    """
    s1 = runs("S1")
    mask = next(iter(s1.values()))[0][1]["kw_has"]
    y = d["y_test"][mask]
    rows, means = {}, {}

    def add(name, predictions):                   # predictions: {copy: [one array per seed]}
        f1 = {c: [macro_f1(y, p[mask]) for p in predictions[c]] for c in COPIES}
        rows[name] = {COPIES[c][1]: mean_sd(v) for c, v in f1.items()}
        means[name] = {c: np.mean(v) for c, v in f1.items()}

    for study, suffix in [("S1", ""), ("S14", ""), ("S16", " (tik kontekstas)")]:
        for (model, _), seed_runs in runs(study).items():
            add(config.MODEL_NAMES_LT[model] + suffix, {c: [z[key] for _, z in seed_runs] for c, (key, _) in COPIES.items()})
    rule = keyword_rule(d["x_train"], d["y_train"])
    kw = keyword_variants(d["x_test"])
    add(RULE_LT, {"original": [rule(d["x_test"])], "without": [rule(kw["without"])], "only": [rule(kw["i_feel"])],
                  "i_feel": [rule(kw["i_feel"])], "control": [rule(kw["control"])]})
    table = pd.DataFrame(rows).T
    table["kritimas be emocijos žodžio"] = [f"{m['original'] - m['without']:.3f}".replace(".", ",") for m in means.values()]
    return table, pd.DataFrame(means).T, int(mask.sum())


def plot_keyword_copies(means, n_texts, path):
    """Macro-F1 of every model on the original keyword texts, without the emotion word and with only the word."""
    shown = [("original", "originalus tekstas", "tab:blue"), ("without", "be emocijos žodžio", "tab:red"),
             ("only", "tik emocijos žodis", "tab:green")]
    names = list(means.index)[::-1]               # first model at the top
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.2))
    for i, (copy, label, color) in enumerate(shown):
        ax.barh(y + (1 - i) * 0.27, means.loc[names, copy], height=0.27, color=color, label=label)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Makro F-matas")
    comma_axis(ax.xaxis)
    ax.set_title(f"Raktinio žodžio testas ({n_texts:,} tekstų su emocijos žodžiu)".replace(",", " "))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)


def negation_table(d):
    """Negation templates (CheckList, Ribeiro et al., 2020; data.negation_templates): the 10 most frequent pure
    emotion words of each class in "i feel X", "i dont feel X" and "i never feel X". There is no gold label for
    a negated sentence, so the table shows how often the prediction differs from the one for "i feel X" and how much
    the probability of X's class drops. If DistilBERT ignores negation too, the labels taught the shortcut.
    S1 models are loaded from results/models/; DistilBERT's probabilities come from its S14 predictions."""
    templates = negation_templates(d["x_train"], d["y_train"])
    labels = templates["label"].to_numpy()[0::3]  # three rows per word: i feel / i dont feel / i never feel
    texts = templates["text"].to_numpy()

    def scores(prob):                             # prob: (3 x words, 6) in the order of the templates
        prob = prob.reshape(-1, 3, N_CLASSES)
        pred = prob.argmax(axis=2)
        own = prob[np.arange(len(labels)), :, labels]          # probability of X's class, (words, 3)
        return {"„i feel X“ priskirta X klasei, %": (pred[:, 0] == labels).mean() * 100,
                "„i dont feel X“: prognozė pasikeitė, %": (pred[:, 1] != pred[:, 0]).mean() * 100,
                "„i never feel X“: prognozė pasikeitė, %": (pred[:, 2] != pred[:, 0]).mean() * 100,
                "X klasės tikimybė: i feel X": own[:, 0].mean(),
                "X klasės tikimybė: i dont feel X": own[:, 1].mean(),
                "X klasės tikimybė: i never feel X": own[:, 2].mean()}

    rows = {}

    def add(name, per_seed):                      # per_seed: list of score dicts
        rows[name] = {k: mean_sd([s[k] for s in per_seed], 1, 1 if k.endswith("%") else 3) for k in per_seed[0]}

    for (model, variant), seed_runs in runs("S1").items():
        per_seed = []
        for seed in saved_seeds("S1", model, variant, seed_runs):
            net, vectorizer = load_saved("S1", model, variant, seed)
            per_seed.append(scores(predict_texts(net, vectorizer, model, texts)))
            keras.backend.clear_session()
        add(config.MODEL_NAMES_LT[model], per_seed)
    for (model, _), seed_runs in runs("S14").items():
        add(config.MODEL_NAMES_LT[model], [scores(z["prob_neg"]) for _, z in seed_runs])
    print(f"Neigimo šablonai: {len(labels)} žodžių × 3 šablonai")
    return pd.DataFrame(rows).T


def attention_table(d, seed=0):
    """BiLSTM + attention (S1, all seeds): how much of the attention the emotion word gets, and an erasure check.

    Attention share: the weight a_t on the emotion word (keyword texts with the word among the first 50 words),
    against 1 / (number of words) if every word got the same weight, and how often it is the most-attended word.
    Erasure (does the attention point at what the model uses? Jain & Wallace, 2019): delete the most-attended word,
    or a random other word, from every test text with at least 2 words and count how often the prediction changes.
    """
    rng = np.random.default_rng(seed)
    texts = d["x_test"]
    words = [t.split() for t in texts]
    s1 = runs("S1")
    variant, seed_runs = next((v, r) for (m, v), r in s1.items() if m == "bilstm_attention")
    keyword_mask = seed_runs[0][1]["kw_has"]
    positions = [emotion_word(t) for t in texts]
    on_keyword = np.array([keyword_mask[k] and positions[k] is not None and positions[k] < config.SEQUENCE_LENGTH
                           for k in range(len(texts))])
    erasable = np.array([len(w) >= 2 for w in words])

    def delete(k, j):
        return " ".join(words[k][:j] + words[k][j + 1:])

    per_seed = []
    for run_seed in saved_seeds("S1", "bilstm_attention", variant, seed_runs):
        net, vectorizer = load_saved("S1", "bilstm_attention", variant, run_seed)
        x = encode(vectorizer, texts, "sequence")
        a = models_b.attention_weights(net, x)                                   # (n, 50); padding gets 0
        pred = net.predict(x, batch_size=config.BATCH_SIZE, verbose=0).argmax(axis=1)
        idx = np.where(on_keyword)[0]
        share = a[idx, [positions[k] for k in idx]]
        equal = np.array([1 / min(len(words[k]), config.SEQUENCE_LENGTH) for k in idx])
        top = a.argmax(axis=1)
        idx_e = np.where(erasable)[0]
        other = [rng.choice([j for j in range(min(len(words[k]), config.SEQUENCE_LENGTH)) if j != top[k]]) for k in idx_e]
        p_top = predict_texts(net, vectorizer, "bilstm_attention", np.array([delete(k, top[k]) for k in idx_e])).argmax(1)
        p_other = predict_texts(net, vectorizer, "bilstm_attention",
                                np.array([delete(k, j) for k, j in zip(idx_e, other)])).argmax(1)
        per_seed.append({
            "dėmesio dalis emocijos žodžiui": share.mean(),
            "ta pati dalis, jei visi žodžiai gautų vienodai": equal.mean(),
            "emocijos žodis gauna daugiausia dėmesio, %": (top[idx] == [positions[k] for k in idx]).mean() * 100,
            "pašalinus labiausiai stebimą žodį, prognozė pasikeičia, %": (p_top != pred[idx_e]).mean() * 100,
            "pašalinus atsitiktinį kitą žodį, prognozė pasikeičia, %": (p_other != pred[idx_e]).mean() * 100,
        })
        keras.backend.clear_session()
    column = f"BiLSTM + dėmesys ({len(per_seed)} sėklos; {on_keyword.sum()} / {erasable.sum()} tekstų)"
    table = pd.DataFrame({column: {k: mean_sd([s[k] for s in per_seed], 1, 1 if k.endswith("%") else 3)
                                   for k in per_seed[0]}})
    return table


def s13():
    d = load_emotion()
    groups = keyword_group_table(d)
    print(groups.to_string())
    save(groups, "s13_keyword_groups.csv", "modelis (bendras tikslumas, %)")
    plot_keyword_groups(d, os.path.join(FIGURES_DIR, "s13_keyword_groups.png"))
    copies, means, n_texts = keyword_copies_table(d)
    print(copies.to_string())
    save(copies, "s13_keyword_copies.csv", f"modelis (makro F-matas, {n_texts} tekstų)")
    plot_keyword_copies(means, n_texts, os.path.join(FIGURES_DIR, "s13_keyword_copies.png"))
    negation = negation_table(d)
    print(negation.to_string())
    save(negation, "s13_negation.csv", "modelis")
    attention = attention_table(d)
    print(attention.to_string())
    save(attention, "s13_attention.csv", "rodiklis")


# ---------- S6 (with S15, S17, S12, S14, S14b): another corpus and another language. Report 4.9 ----------

LOVE = config.CLASSES_EN.index("love")
EMBEDDINGS_LT = {"learned100": "išmokti 100-mačiai vektoriai", "glove-frozen": "GloVe, nekeičiami",
                 "glove-tuned": "GloVe, derinami", "glove-extended": "GloVe, nekeičiami, išplėstas žodynas"}


def first_runs(study, model):
    """The runs of one model in a study (its first variant), or [] if the study has none yet."""
    return next((seed_runs for (m, _), seed_runs in runs(study).items() if m == model), [])


def interval(diff, low, high):
    """"-0,228 [-0,233; -0,223]": a difference with its 95 % confidence interval, decimal comma."""
    return f"{diff:.3f} [{low:.3f}; {high:.3f}]".replace(".", ",")


def lt_numbers(table, digits=1):
    """Numbers of a plain table written with a decimal comma (whole numbers stay whole)."""
    return table.map(lambda v: str(int(v)) if float(v).is_integer() else f"{v:.{digits}f}".replace(".", ","))


def lithuanian_table():
    """S15 vs S1: every model trained in Lithuanian and in English on the same texts and labels (macro-F1, mean ±
    sd over the seeds). The difference LT − EN on the extended test comes with its 95 % paired-bootstrap interval
    (evaluation.bootstrap_diff: texts resampled, seeds averaged). The last column feeds the Lithuanian test texts to
    the English model, a sanity check only: 92 % of their words are unknown to it. DistilBERT rows: English (S14)
    and multilingual trained in English only, tested in Lithuanian without any Lithuanian training (S14b)."""
    cols = ["testavimo imtis, anglų k.", "testavimo imtis, lietuvių k.", "išplėstinė imtis, anglų k.",
            "išplėstinė imtis, lietuvių k.", "skirtumas LT − EN išplėstinėje imtyje [95 % PI]",
            "angliškas modelis su lietuviškais tekstais"]
    rows = {}
    for (model, _), en in ((k, five_seeds(r)) for k, r in runs("S1").items()):
        lt = first_runs("S15", model)
        y_x = en[0][1]["y_xtest"]
        assert (lt[0][1]["y_xtest"] == y_x).all()          # the same 50,000 texts, translated
        diff = evaluation.bootstrap_diff(y_x, [z["p_xtest"] for _, z in lt], [z["p_xtest"] for _, z in en])
        rows[config.MODEL_NAMES_LT[model]] = dict(zip(cols, [
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in en]),
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in lt]),
            mean_sd([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in en]),
            mean_sd([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in lt]),
            interval(*diff[:3]),
            mean_sd([macro_f1(z["y_lt_test"], z["p_lt_test"]) for _, z in en])]))
        print("S15:", model, interval(*diff[:3]))
    for (model, _), en in runs("S14").items():
        rows[config.MODEL_NAMES_LT[model]] = dict(zip(cols, [
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in en]), "–",
            mean_sd([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in en]), "–", "–",
            mean_sd([macro_f1(z["y_lt_test"], z["p_lt_test"]) for _, z in en])]))
    for (model, _), multi in runs("S14b").items():
        y_x = multi[0][1]["y_xtest"]
        assert (multi[0][1]["y_lt_xtest"] == y_x).all()
        diff = evaluation.bootstrap_diff(y_x, [z["p_lt_xtest"] for _, z in multi], [z["p_xtest"] for _, z in multi])
        rows[config.MODEL_NAMES_LT[model] + " (mokytas tik angliškai)"] = dict(zip(cols, [
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in multi]),
            mean_sd([macro_f1(z["y_lt_test"], z["p_lt_test"]) for _, z in multi]),
            mean_sd([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in multi]),
            mean_sd([macro_f1(z["y_lt_xtest"], z["p_lt_xtest"]) for _, z in multi]),
            interval(*diff[:3]), "–"]))
    return pd.DataFrame(rows).T


def lithuanian_groups_table(d):
    """S15 vs S1 per keyword group (groups from the English original texts, data.keyword_groups): accuracy, %.
    Shows where Lithuanian loses: the emotion word can be translated into several word forms, or lost."""
    y = d["y_test"]
    groups = keyword_groups(d["x_test"], d["x_train"], d["y_train"])
    rows = {}
    for (model, _), en in ((k, five_seeds(r)) for k, r in runs("S1").items()):
        lt = first_runs("S15", model)
        row = {}
        for g, name in GROUPS_LT.items():
            m = groups == g
            row[f"{name} (n = {m.sum()}): EN, %"] = mean_sd([(z["p_test"][m] == y[m]).mean() for _, z in en], 100, 1)
            row[f"{name} (n = {m.sum()}): LT, %"] = mean_sd([(z["p_test"][m] == y[m]).mean() for _, z in lt], 100, 1)
        rows[config.MODEL_NAMES_LT[model]] = row
    return pd.DataFrame(rows).T


def vocabulary_table():
    """The same train texts in English and Lithuanian (PLAN.md section 3): Lithuanian has about twice as many word
    forms, so the vocabulary of VOCAB_SIZE covers fewer word occurrences and more validation words are unknown."""
    rows = {}
    for lang, name in [("en", "anglų k."), ("lt", "lietuvių k.")]:
        d = load_emotion(lang=lang)
        counts = Counter(w for text in d["x_train"] for w in text.split())
        vocabulary = set(vectorize(d)["vectorizer"].get_vocabulary())
        val_words = [w for text in d["x_val"] for w in text.split()]
        rows[name] = {
            "skirtingų žodžių formų mokymo imtyje": len(counts),
            "iš jų pasitaiko tik kartą, %": sum(n == 1 for n in counts.values()) / len(counts) * 100,
            f"žodynas ({config.VOCAB_SIZE:,} žodžių)".replace(",", " ") + " apima mokymo imties žodžių, %":
                sum(n for w, n in counts.items() if w in vocabulary) / sum(counts.values()) * 100,
            "validavimo imties žodžių už žodyno ribų, %": sum(w not in vocabulary for w in val_words) / len(val_words) * 100,
            "žodžių tekste (vidurkis)": np.mean([len(text.split()) for text in d["x_train"]]),
        }
    return lt_numbers(pd.DataFrame(rows))


def goemotions_test_part():
    """Where the 638 GoEmotions test texts are within load_goemotions() (its splits come in the order train,
    validation, test), so CARER-trained models are scored on the same texts as the GoEmotions-trained ones."""
    splits = load_goemotions_splits()
    start = len(splits["y_train"]) + len(splits["y_val"])
    return slice(start, start + len(splits["y_test"])), splits["y_test"]


def cross_corpus_table():
    """S17 with S1, macro-F1: train corpus (CARER = S1, GoEmotions = S17) × test corpus (the CARER test set and the
    638-text GoEmotions test split), for the four S17 models: a 2 × 2 table as in Bostan & Klinger (2018). The
    last column scores the CARER models on all 6,535 GoEmotions texts (S6). Last row: the most common class of the
    training corpus, predicted for every text."""
    part, y_go_test = goemotions_test_part()
    cols = ["CARER → CARER", "GoEmotions → GoEmotions", "CARER → GoEmotions", "GoEmotions → CARER",
            "CARER → visi GoEmotions (6 535)"]
    rows = {}
    for (model, _), go in runs("S17").items():
        carer = five_seeds(first_runs("S1", model))
        assert (carer[0][1]["y_go"][part] == y_go_test).all() and (go[0][1]["y_test"] == y_go_test).all()
        rows[config.MODEL_NAMES_LT[model]] = dict(zip(cols, [
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in carer]),
            mean_sd([macro_f1(z["y_test"], z["p_test"]) for _, z in go]),
            mean_sd([macro_f1(z["y_go"][part], z["p_go"][part]) for _, z in carer]),
            mean_sd([macro_f1(z["y_carer"], z["p_carer"]) for _, z in go]),
            mean_sd([macro_f1(z["y_go"], z["p_go"]) for _, z in carer])]))
    d, go_splits, (_, y_go_all) = load_emotion(), load_goemotions_splits(), load_goemotions()
    carer_major, go_major = np.bincount(d["y_train"]).argmax(), np.bincount(go_splits["y_train"]).argmax()

    def constant(y, c):
        return f"{macro_f1(y, np.full(len(y), c)):.3f}".replace(".", ",")

    rows[f"Dažniausia klasė („{config.CLASSES_LT[carer_major]}“ / „{config.CLASSES_LT[go_major]}“)"] = dict(zip(cols, [
        constant(d["y_test"], carer_major), constant(y_go_test, go_major), constant(y_go_test, carer_major),
        constant(d["y_test"], go_major), constant(y_go_all, carer_major)]))
    return pd.DataFrame(rows).T


def goemotions_per_class_table():
    """S6: the CARER-trained models (S1, S12, S14) on all 6,535 GoEmotions texts: recall per class, macro-F1, and
    macro-F1 over the other 5 classes on the texts that are not love. Love is the most common GoEmotions class
    (26.9 %) and means something else there: 1,228 of its 1,760 texts contain the word "love", which CARER labels
    joy as often as love."""
    y = load_goemotions()[1]
    no_love = y != LOVE
    others = [c for c in range(N_CLASSES) if c != LOVE]

    def scores(p):
        m = evaluation.metrics(y, p, N_CLASSES)
        r = {f"pilnumas: {config.CLASSES_LT[c]}, %": m["recall"][c] * 100 for c in range(N_CLASSES)}
        r["makro F-matas"] = m["macro_f1"]
        r["makro F-matas be meilės (5 klasės)"] = evaluation.metrics(y[no_love], p[no_love], N_CLASSES)["f1"][others].mean()
        return r

    rows = {}

    def add(name, per_seed):
        rows[name] = {k: mean_sd([s[k] for s in per_seed], 1, 1 if k.endswith("%") else 3) for k in per_seed[0]}

    for study in ["S1", "S14"]:
        for (model, _), seed_runs in ((k, five_seeds(r) if study == "S1" else r) for k, r in runs(study).items()):
            add(config.MODEL_NAMES_LT[model], [scores(z["p_go"]) for _, z in seed_runs])
    for (model, variant), seed_runs in runs("S12").items():
        embedding = variant.split("-", 1)[1]                 # "size128-glove-frozen" -> "glove-frozen"
        add(f"{config.MODEL_NAMES_LT[model]}, {EMBEDDINGS_LT[embedding]}", [scores(z["p_go"]) for _, z in seed_runs])
    add(f"Dažniausia klasė („{config.CLASSES_LT[LOVE]}“)", [scores(np.full(len(y), LOVE))])
    return pd.DataFrame(rows).T


def plot_goemotions_confusion(path):
    """Confusion matrix on all GoEmotions texts of the reference model (best S1 model on the selection set), seed 1."""
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "results.csv"))
    best = df[df["study"] == "S1"].groupby("model")["sel_macro_f1"].mean().idxmax()
    seed, z = first_runs("S1", best)[0]
    m = evaluation.confusion_matrix(z["y_go"], z["p_go"], N_CLASSES)
    evaluation.plot_confusion_matrix(m, config.CLASSES_LT, f"{config.MODEL_NAMES_LT[best]} (mokytas CARER), "
                                                           f"GoEmotions, sėkla {seed}", path)
    print("Išsaugota:", path)


def plot_lithuanian(path):
    """Macro-F1 on the extended test of every model trained in English (S1, seeds 1–5) and in Lithuanian (S15) on
    the same texts, and multilingual DistilBERT trained in English only, tested in both languages (S14b)."""
    names, english, lithuanian = [], [], []
    for (model, _), en in runs("S1").items():
        lt = first_runs("S15", model)
        names.append(config.MODEL_NAMES_LT[model])
        english.append(np.mean([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in five_seeds(en)]))
        lithuanian.append(np.mean([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in lt]))
    for (model, _), multi in runs("S14b").items():
        names.append(config.MODEL_NAMES_LT[model] + "\n(be lietuviško mokymo)")
        english.append(np.mean([macro_f1(z["y_xtest"], z["p_xtest"]) for _, z in multi]))
        lithuanian.append(np.mean([macro_f1(z["y_lt_xtest"], z["p_lt_xtest"]) for _, z in multi]))
    y = np.arange(len(names))[::-1]                  # first model at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.3))
    ax.barh(y + 0.2, english, height=0.4, label="anglų k.", color="tab:blue")
    ax.barh(y - 0.2, lithuanian, height=0.4, label="lietuvių k. (mūsų modeliai mokyti lietuviškai)", color="tab:orange")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Makro F-matas (išplėstinė testavimo imtis, tie patys tekstai)")
    comma_axis(ax.xaxis)
    ax.set_title("Tie patys tekstai anglų ir lietuvių kalba")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Išsaugota:", path)


def s6():
    d = load_emotion()
    lithuanian = lithuanian_table()
    print(lithuanian.to_string())
    save(lithuanian, "s15_lithuanian.csv", "modelis (makro F-matas)")
    plot_lithuanian(os.path.join(FIGURES_DIR, "s15_en_vs_lt.png"))
    groups = lithuanian_groups_table(d)
    print(groups.to_string())
    save(groups, "s15_keyword_groups.csv", "modelis (bendras tikslumas, %)")
    vocabulary = vocabulary_table()
    print(vocabulary.to_string())
    save(vocabulary, "s15_vocabulary.csv", "rodiklis")
    cross = cross_corpus_table()
    print(cross.to_string())
    save(cross, "s17_cross_corpus.csv", "modelis (makro F-matas; mokyta → testuota)")
    per_class = goemotions_per_class_table()
    print(per_class.to_string())
    save(per_class, "s6_goemotions_per_class.csv", "modelis (mokytas CARER)")
    plot_goemotions_confusion(os.path.join(FIGURES_DIR, "s6_goemotions_confusion.png"))


PARTS = {"S13": s13, "S6": s6}

if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
