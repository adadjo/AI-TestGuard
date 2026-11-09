"""Style commun et constructeurs de figures (matplotlib, rendu sans écran)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
COLORS = {
    "logreg": "#1F5FA8",
    "tree": "#D9822B",
    "fraud": "#C0392B",
    "legit": "#5B7C99",
    "ok": "#2E8B57",
    "warn": "#E0A100",
    "bad": "#C0392B",
    "grey": "#7F8C8D",
    "train": "#DCE9F7",
    "val": "#FBE7CF",
    "test": "#F6D5D1",
    "infra": "#E6E8EA",
}
LABELS = {"logreg": "Régression logistique", "tree": "Arbre de décision"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 100,
        }
    )


style()


def save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f} %".replace(".", ",")


# ----------------------------------------------------------------------- architecture
def fig_architecture() -> str:
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.25,rounding_size=1.2",
                fc=fc,
                ec="#34495E",
                lw=0.9,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold" if bold else "normal",
            linespacing=1.25,
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#34495E"
            )
        )

    w, h = 17.5, 12
    top = [
        (2, "1. Chargement\nCSV sécurisé\n(taille, extension,\nformat)", "infra"),
        (22, "2. validate_data\ncontrat : schéma,\ntypes, domaines", "infra"),
        (42, "3. Dédoublonnage\n(lignes identiques)", "infra"),
        (62, "4. Partition\nstratifiée\n70 / 15 / 15", "infra"),
    ]
    for x, text, color in top:
        box(x, 40, w, h, text, COLORS[color])
    for x in (19.5, 39.5, 59.5):
        arrow(x + 0.2, 46, x + 2.3, 46)
    box(82, 40, 16, h, "Jeu de test\n(jamais utilisé\npour régler)", COLORS["test"], bold=True)
    arrow(79.5, 46, 81.8, 46)

    box(
        2,
        20,
        28,
        h + 2,
        "5. ENTRAÎNEMENT (train)\nimputation médiane → log1p(Amount)\n→ mise à l'échelle → modèle\n(tout appris sur train)",
        COLORS["train"],
        bold=False,
    )
    box(
        37,
        20,
        27,
        h + 2,
        "6. RÉGLAGE (validation)\nhyperparamètres (PR-AUC)\net seuil : max rappel\navec FPR ≤ 0,08 %",
        COLORS["val"],
    )
    box(
        71,
        20,
        27,
        h + 2,
        "7. SAUVEGARDE\njoblib + empreinte SHA-256\n8. RECHARGEMENT vérifié\n(avant désérialisation)",
        COLORS["infra"],
    )
    arrow(30.2, 27, 36.8, 27)
    arrow(64.2, 27, 70.8, 27)
    arrow(70, 40, 16, 34.4)  # partition -> train
    ax.text(41, 38.2, "train", fontsize=7, color="#1F5FA8", style="italic")

    box(
        2,
        2,
        44,
        11,
        "9. PRÉDICTION : predict_fraud (contrat de sortie)\nprobabilité ∈ [0, 1] ; décision  p ≥ seuil",
        COLORS["infra"],
    )
    box(
        54,
        2,
        44,
        11,
        "10. ÉVALUATION FINALE (test, une seule fois)\nmétriques + IC, rapport, critères d'acceptation",
        COLORS["test"],
        bold=True,
    )
    arrow(84.5, 20, 30, 13.3)
    arrow(46.2, 7.5, 53.8, 7.5)
    ax.text(
        50,
        54.5,
        "Pipeline AI-TestGuard : chaque étape est testée (unitaire, intégration, robustesse)",
        ha="center",
        fontsize=8.6,
        fontweight="bold",
    )
    return save(fig, "fig_architecture.png")


# ---------------------------------------------------------- évaluation : exigences ML
def fig_accuracy_trap(dummy: dict, model: dict, model_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    counts = [dummy["n_negative"], dummy["n_positive"]]
    axes[0].bar(["Normales", "Fraudes"], counts, color=[COLORS["legit"], COLORS["fraud"]])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Transactions (échelle log)")
    axes[0].set_title("Jeu de test : classes très déséquilibrées")
    for i, c in enumerate(counts):
        axes[0].text(i, c * 1.25, f"{c:,}".replace(",", " "), ha="center", fontsize=8)
    axes[0].set_ylim(1, max(counts) * 8)
    names = ["Accuracy", "Précision", "Rappel", "F1"]
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(keys))
    axes[1].bar(
        x - 0.2,
        [dummy[k] for k in keys],
        0.38,
        color=COLORS["grey"],
        label="Classifieur trivial (« jamais de fraude »)",
    )
    axes[1].bar(
        x + 0.2, [model[k] for k in keys], 0.38, color=COLORS[model_name], label=LABELS[model_name]
    )
    for i, k in enumerate(keys):
        axes[1].text(
            i - 0.2, dummy[k] + 0.02, f"{dummy[k]:.3f}".replace(".", ","), ha="center", fontsize=7
        )
        axes[1].text(
            i + 0.2, model[k] + 0.02, f"{model[k]:.3f}".replace(".", ","), ha="center", fontsize=7
        )
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 1.42)
    axes[1].set_title("L'accuracy masque l'absence de détection")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_accuracy_trap.png")


def fig_roc(curves: dict, ops: dict, aucs: dict, cap: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, zoom in zip(axes, (False, True), strict=True):
        for name, (fpr, tpr) in curves.items():
            ax.plot(
                fpr,
                tpr,
                color=COLORS[name],
                lw=1.6,
                label=f"{LABELS[name]} (AUC = {aucs[name]:.3f})".replace(".", ","),
            )
            ax.scatter(*ops[name], s=38, color=COLORS[name], edgecolor="black", zorder=5)
        if zoom:
            ax.set_xlim(0, 0.005)
            ax.set_ylim(0, 1.0)
            ax.axvline(cap, ls="--", color=COLORS["bad"], lw=1, label="Plafond métier FPR = 0,1 %")
            ax.set_title("Zoom : FPR ≤ 0,5 % (zone opérationnelle)")
            ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f} %".replace(".", ","))
        else:
            ax.plot([0, 1], [0, 1], ls=":", color=COLORS["grey"], label="Hasard")
            ax.set_title("Courbe ROC (jeu de test)")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Rappel (taux de vrais positifs)")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_roc.png")


def fig_pr(curves: dict, ops: dict, aps: dict, prevalence: float) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, (rec, prec) in curves.items():
        ax.plot(
            rec,
            prec,
            color=COLORS[name],
            lw=1.6,
            label=f"{LABELS[name]} (AP = {aps[name]:.3f})".replace(".", ","),
        )
        ax.scatter(*ops[name], s=38, color=COLORS[name], edgecolor="black", zorder=5)
    ax.axhline(
        prevalence,
        ls=":",
        color=COLORS["grey"],
        label=f"Hasard (prévalence {100 * prevalence:.2f} %)".replace(".", ","),
    )
    ax.axvline(0.75, ls="--", color=COLORS["bad"], lw=1, label="Rappel minimal CA01 = 0,75")
    ax.axhline(0.50, ls="--", color=COLORS["warn"], lw=1, label="Précision minimale CA03 = 0,50")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_title("Courbe précision-rappel (jeu de test) — points = seuil figé")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_pr.png")


def fig_confusion(matrices: dict) -> str:
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.9 * len(matrices), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, matrices.items(), strict=True):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        names = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{names[i][j]}\n{cm[i, j]:,}\n({100 * norm[i, j]:.2f} %)".replace(
                        ",", " "
                    ).replace(".", ","),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1], ["Prédit normale", "Prédit fraude"])
        ax.set_yticks([0, 1], ["Réelle normale", "Réelle fraude"], rotation=90, va="center")
        ax.set_title(LABELS[name])
    fig.suptitle(
        "Matrices de confusion (jeu de test, seuil figé sur la validation)", fontsize=9.5, y=1.02
    )
    fig.tight_layout()
    return save(fig, "fig_confusion.png")


def fig_threshold(
    roc_val: tuple, cost_curve: list, target: float, cap: float, chosen_cost: float
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    fpr, tpr = roc_val
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["logreg"], lw=1.6)
    ax.axvline(target, color=COLORS["ok"], ls="--", lw=1.1, label="Cible de réglage 0,08 %")
    ax.axvline(cap, color=COLORS["bad"], ls="--", lw=1.1, label="Plafond métier 0,10 %")
    ax.set_xlim(0, 0.005)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f} %".replace(".", ","))
    ax.set_xlabel("Taux de faux positifs (validation)")
    ax.set_ylabel("Rappel (validation)")
    ax.set_title("Compromis rappel / fausses alertes")
    ax.legend(loc="lower right", fontsize=7)
    ax = axes[1]
    xs = [r["max_fpr"] for r in cost_curve]
    ys = [r["total_cost"] for r in cost_curve]
    ax.plot(xs, ys, color=COLORS["fraud"], lw=1.6)
    k = int(np.argmin(ys))
    ax.scatter(
        xs[k], ys[k], color=COLORS["fraud"], edgecolor="black", zorder=5, label="Minimum économique"
    )
    ax.axvline(target, color=COLORS["ok"], ls="--", lw=1.1)
    ax.axvline(cap, color=COLORS["bad"], ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Budget de FPR (échelle log)")
    ax.set_ylabel("Coût total (validation, €)")
    ax.set_title("Coût = fraude manquée + 5 € / fausse alerte")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_threshold.png")


def fig_scores(p_legit, p_fraud, threshold: float, name: str) -> str:
    def logit(p):
        p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    bins = np.linspace(-14, 14, 57)
    ax.hist(
        logit(p_legit), bins=bins, color=COLORS["legit"], alpha=0.85, label="Transactions normales"
    )
    ax.hist(logit(p_fraud), bins=bins, color=COLORS["fraud"], alpha=0.85, label="Fraudes")
    ax.axvline(
        logit(threshold),
        color="black",
        ls="--",
        lw=1.2,
        label=f"Seuil figé (p = {threshold:.4f})".replace(".", ","),
    )
    ax.set_yscale("log")
    ax.set_xlabel("Score du modèle, échelle logit : log(p / (1 − p))")
    ax.set_ylabel("Transactions (échelle log)")
    ax.set_title(f"Distribution des scores — {LABELS[name]} (test)")
    ax.legend(loc="upper center", fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_scores.png")


def fig_overfit(rows: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    names = list(rows)
    x = np.arange(len(names))
    for i, (part, color) in enumerate(
        (("train", "#9DB9D9"), ("val", "#E8B27A"), ("test", "#D98A82"))
    ):
        vals = [rows[n][part] for n in names]
        ax.bar(
            x + (i - 1) * 0.26,
            vals,
            0.25,
            color=color,
            label={"train": "Entraînement", "val": "Validation", "test": "Test"}[part],
        )
        for xi, v in zip(x + (i - 1) * 0.26, vals, strict=True):
            ax.text(xi, v + 0.012, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax.axhline(0.65, ls="--", color=COLORS["bad"], lw=1, label="Critère CA04 (PR-AUC ≥ 0,65)")
    ax.set_xticks(x, [LABELS[n] for n in names])
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_title("Surapprentissage : entraînement vs validation vs test")
    ax.legend(loc="upper center", fontsize=7, ncol=2)
    fig.tight_layout()
    return save(fig, "fig_overfit.png")


def fig_seeds(data: dict, criteria: dict) -> str:
    metrics = [
        ("recall", "Rappel", criteria["recall"]),
        ("fpr", "FPR (‰)", criteria["fpr"]),
        ("precision", "Précision", criteria["precision"]),
        ("average_precision", "PR-AUC", criteria["average_precision"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 3.2))
    for ax, (key, title, crit) in zip(axes, metrics, strict=True):
        scale = 1000 if key == "fpr" else 1
        for i, name in enumerate(data):
            vals = np.array([r[key] for r in data[name]]) * scale
            ax.boxplot(
                vals,
                positions=[i],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": COLORS[name], "alpha": 0.35},
                medianprops={"color": "black"},
            )
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
            ax.scatter(i + jitter, vals, s=12, color=COLORS[name], zorder=4)
        ax.axhline(crit * scale, ls="--", color=COLORS["bad"], lw=1)
        ax.set_xticks(range(len(data)), ["RL", "AD"])
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        "Sensibilité à la graine : 10 partitions/entraînements indépendants (jeu de test)",
        fontsize=9.5,
        y=1.03,
    )
    fig.tight_layout()
    return save(fig, "fig_seeds.png")
