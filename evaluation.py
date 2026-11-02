import math

import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(y, p, n_classes):
    """Confusion matrix: rows = true class, columns = predicted class.
    m[i, j] = how many texts of class i the model predicted as class j; the diagonal is correct.
    """
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (y, p), 1)                       # +1 in cell (true, predicted) for every text
    return m


def metrics(y, p, n_classes):
    """Accuracy, per-class precision/recall/F1 and macro-F1, computed from the confusion matrix.

    Returns a dict: accuracy, macro_f1 (floats) and precision, recall, f1 (np.array, one value per class).
    For one class: TP = diagonal cell, FP = rest of its column, FN = rest of its row.
    """
    m = confusion_matrix(y, p, n_classes)
    tp = np.diag(m)
    predicted = m.sum(axis=0)                     # TP + FP: texts predicted as this class
    actual = m.sum(axis=1)                        # TP + FN: texts that really are this class

    precision = _divide(tp, predicted)            # tikslumas = TP / (TP + FP)
    recall = _divide(tp, actual)                  # pilnumas = TP / (TP + FN)
    f1 = _divide(2 * precision * recall, precision + recall)   # F-matas = 2PR / (P + R)
    return {
        "accuracy": tp.sum() / m.sum(),           # bendras tikslumas = teisingų / visų
        "macro_f1": f1.mean(),                    # makro F-matas: every class counts equally, however small
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def mcnemar(y, p1, p2):
    """McNemar test: do two models make significantly different errors on the same texts?

    b = texts model 1 got right and model 2 got wrong; c = the other way round.
    chi2 = (|b - c| - 1)^2 / (b + c), with continuity correction; p from chi2 with 1 degree of freedom.
    Returns (chi2, p_value); p_value < 0.05 means the difference is significant.
    """
    right1 = p1 == y
    right2 = p2 == y
    b = int(np.sum(right1 & ~right2))
    c = int(np.sum(~right1 & right2))
    if b + c == 0:                                # both models err on exactly the same texts
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(chi2 / 2))      # P(X >= chi2) for chi2 with 1 degree of freedom
    return chi2, p_value


def bootstrap_diff(y, preds1, preds2, metric="macro_f1", n_classes=6, n=1000, seed=0):
    """Paired bootstrap: is model 1 better than model 2 on these texts, beyond the luck of which texts were drawn?

    preds1, preds2: lists of prediction arrays, one per seed, for the same texts y. A model's score is its metric
    ("macro_f1" or "accuracy") averaged over its seeds. Draw len(y) texts with replacement, n times; each time both
    models are scored on the same drawn texts and the difference score1 - score2 is kept.
    Returns (diff, ci_low, ci_high, p_value): the difference on all texts, the 2.5 % and 97.5 % points of the n
    differences (the 95 % confidence interval), and p = twice the share of differences on the other side of 0.
    """
    rng = np.random.default_rng(seed)

    def score(preds, texts):
        return np.mean([metrics(y[texts], p[texts], n_classes)[metric] for p in preds])

    everything = np.arange(len(y))
    diff = score(preds1, everything) - score(preds2, everything)
    diffs = []
    for _ in range(n):
        texts = rng.integers(0, len(y), len(y))          # the same drawn texts for both models: "paired"
        diffs.append(score(preds1, texts) - score(preds2, texts))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = min(1.0, 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))
    return diff, ci_low, ci_high, p_value


def holm(p_values, alpha=0.05):
    """Holm–Bonferroni correction for m tests at once (e.g. the best model vs each of 10 others).

    Sort the p-values from smallest to largest; the i-th smallest (i = 1 ... m) must be <= alpha / (m - i + 1).
    Stop at the first one that fails: it and all larger p-values are not significant.
    Returns a bool array in the original order: True = still significant after the correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    significant = np.zeros(m, dtype=bool)
    for i, index in enumerate(np.argsort(p_values)):   # i = 0 for the smallest p-value
        if p_values[index] > alpha / (m - i):
            break
        significant[index] = True
    return significant


def cohen_kappa(a, b, n_categories):
    """Cohen's kappa: how much two labellers agree beyond chance (S7 error audit; Cohen, 1960).

    p_o = share of items both put in the same category; p_e = agreement expected by chance = sum over the
    categories of (share of A's labels in it) x (share of B's labels in it). kappa = (p_o - p_e) / (1 - p_e):
    1 = full agreement, 0 = no more than chance.
    """
    m = confusion_matrix(a, b, n_categories) / len(a)   # rows = A's category, columns = B's
    p_o = np.trace(m)
    p_e = np.sum(m.sum(axis=1) * m.sum(axis=0))
    return (p_o - p_e) / (1 - p_e)


def plot_learning_curves(history, best, title, path):
    """Accuracy and loss per epoch for train and validation; history = keras History.history,
    best = the epoch EarlyStopping went back to (marked with a dashed line). NaN epochs are not drawn."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["accuracy"], label="mokymo")
    ax1.plot(epochs, history["val_accuracy"], label="validavimo")
    ax1.set_title("Bendras tikslumas")
    ax2.plot(epochs, history["loss"], label="mokymo")
    ax2.plot(epochs, history["val_loss"], label="validavimo")
    ax2.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"min. val. nuostoliai ({best} ep.)")
    ax2.set_title("Nuostoliai")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epocha")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(m, class_names, title, path):
    """Confusion matrix as a heatmap with the count in every cell; class_names in Lithuanian."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(m, cmap="Blues")
    for i in range(n):
        for j in range(n):
            color = "white" if m[i, j] > m.max() / 2 else "black"   # readable on dark cells
            ax.text(j, i, m[i, j], ha="center", va="center", color=color)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Prognozuota klasė")
    ax.set_ylabel("Tikroji klasė")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _divide(a, b):
    """a / b element by element, 0 where b = 0 (e.g. precision of a class the model never predicts)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


if __name__ == "__main__":
    # Small example computed by hand: 10 texts, 3 classes
    y  = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    p1 = np.array([0, 0, 1, 2, 1, 1, 0, 2, 2, 1])
    p2 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 1])

    # By hand, rows = true class:     class 0: predicted 0,0,1,2 -> [2, 1, 1]
    #                                 class 1: predicted 1,1,0   -> [1, 2, 0]
    #                                 class 2: predicted 2,2,1   -> [0, 1, 2]
    m = confusion_matrix(y, p1, 3)
    print("Maišaties matrica:\n", m)
    assert (m == [[2, 1, 1], [1, 2, 0], [0, 1, 2]]).all()

    # accuracy = 6/10; precision = 2/3, 2/4, 2/3 (columns); recall = 2/4, 2/3, 2/3 (rows)
    # F1: class 0 = 2 * 2/3 * 1/2 / (2/3 + 1/2) = 4/7, class 1 = 4/7, class 2 = 2/3
    # macro-F1 = (4/7 + 4/7 + 2/3) / 3 = 38/63
    r = metrics(y, p1, 3)
    print("Bendras tikslumas:", r["accuracy"])
    print("Tikslumas:", r["precision"].round(4), "| pilnumas:", r["recall"].round(4), "| F-matas:", r["f1"].round(4))
    print("Makro F-matas:", round(r["macro_f1"], 4))
    assert np.isclose(r["accuracy"], 0.6)
    assert np.allclose(r["precision"], [2 / 3, 1 / 2, 2 / 3])
    assert np.allclose(r["recall"], [1 / 2, 2 / 3, 2 / 3])
    assert np.allclose(r["f1"], [4 / 7, 4 / 7, 2 / 3])
    assert np.isclose(r["macro_f1"], 38 / 63)

    # McNemar, model 1 vs model 2: model 2 is right everywhere except the last text, model 1 is wrong there too.
    # b = 0 (1 right, 2 wrong), c = 3 (texts 3, 4 and 7) -> chi2 = (|0 - 3| - 1)^2 / 3 = 4/3,
    # chi2 table with 1 degree of freedom: p = 0.248
    chi2, p_value = mcnemar(y, p1, p2)
    print(f"McNemar: chi2 = {chi2:.4f}, p = {p_value:.4f}")
    assert np.isclose(chi2, 4 / 3)
    assert abs(p_value - 0.248) < 1e-3
    assert abs(mcnemar(y, p1, p1)[1] - 1.0) < 1e-12       # identical models: no difference
    assert abs(math.erfc(math.sqrt(3.841 / 2)) - 0.05) < 1e-3   # table value: chi2 = 3.841 -> p = 0.05

    # Holm by hand: p = 0.01, 0.04, 0.03, 0.005 (m = 4); all four are < 0.05 on their own.
    # Sorted: 0.005 <= 0.05/4 = 0.0125 yes; 0.01 <= 0.05/3 = 0.0167 yes; 0.03 <= 0.05/2 = 0.025 no -> stop.
    # So only 0.01 and 0.005 stay significant.
    significant = holm([0.01, 0.04, 0.03, 0.005])
    print("Holm:", significant)
    assert significant.tolist() == [True, False, False, True]

    # Paired bootstrap on the same 10 texts, accuracy. By hand: model 2 is right on 9, model 1 on 6, so the
    # difference on all texts is 0.6 - 0.9 = -0.3. Model 1 is never right where model 2 is wrong (b = 0 above),
    # so in every draw model 1 scores at most model 2: the interval lies at or below 0.
    diff, low, high, p_value = bootstrap_diff(y, [p1], [p2], "accuracy", 3)
    print(f"Bootstrap (tikslumas): skirtumas {diff:.2f}, 95 % intervalas [{low:.2f}; {high:.2f}], p = {p_value:.3f}")
    assert np.isclose(diff, -0.3) and high <= 0 and low < diff < high
    # identical models: difference 0 in every draw, p = 1; always right vs always wrong: difference 1, p = 0
    assert bootstrap_diff(y, [p1], [p1], "accuracy", 3)[1:] == (0.0, 0.0, 1.0)
    assert bootstrap_diff(y, [y], [(y + 1) % 3], "accuracy", 3) == (1.0, 1.0, 1.0, 0.0)
    # two seeds per model: the score is the mean over the seeds, (0.6 + 0.9) / 2 - 0.9 = -0.15
    assert np.isclose(bootstrap_diff(y, [p1, p2], [p2, p2], "accuracy", 3)[0], -0.15)

    # Cohen's kappa by hand, y and p1 as two labellers' categories: they agree on 6 of 10, p_o = 0.6.
    # Shares: y = [0.4, 0.3, 0.3], p1 = [0.3, 0.4, 0.3] -> p_e = 0.12 + 0.12 + 0.09 = 0.33,
    # kappa = (0.6 - 0.33) / (1 - 0.33) = 27/67
    kappa = cohen_kappa(y, p1, 3)
    print(f"Cohen kappa: {kappa:.4f}")
    assert np.isclose(kappa, 27 / 67)
    assert np.isclose(cohen_kappa(y, y, 3), 1.0)            # the same labels: full agreement

    # Cross-check with Keras (library used only for checking, as for the labs)
    import keras
    keras_f1 = keras.metrics.F1Score(average=None)
    keras_f1.update_state(np.eye(3)[y], np.eye(3)[p1])
    print("Keras F1Score:", np.array(keras_f1.result()).round(4))
    assert np.allclose(keras_f1.result(), r["f1"], atol=1e-6)

    print("Visi patikrinimai sutampa su skaičiavimu ranka.")
