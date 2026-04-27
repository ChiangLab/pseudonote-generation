#!/usr/bin/env python
"""
Generate publication-quality figures comparing Gemma vs ClinicalBERT KNN analysis.
Run AFTER both KNN scripts have completed.
"""

import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline"
GEMMA_KNN_PATH = f"{BASE_DIR}/outputs/knn_analysis_gemma/knn_model_gemma.joblib"
CBERT_KNN_PATH = f"{BASE_DIR}/outputs/knn_analysis_clinicalbert/knn_model_clinicalbert.joblib"
FIG_DIR = f"{BASE_DIR}/outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GEMMA_COLOR = "#2196F3"
CBERT_COLOR = "#FF5722"
PROG_COLOR = "#E53935"
NONPROG_COLOR = "#43A047"


def load_models():
    print("Loading KNN models...")
    models = {}

    if os.path.exists(GEMMA_KNN_PATH):
        models["Gemma"] = joblib.load(GEMMA_KNN_PATH)
        print(f"  Gemma: {len(models['Gemma']['ids'])} patients")
    else:
        print(f"  WARNING: Gemma model not found at {GEMMA_KNN_PATH}")

    if os.path.exists(CBERT_KNN_PATH):
        models["ClinicalBERT"] = joblib.load(CBERT_KNN_PATH)
        print(f"  ClinicalBERT: {len(models['ClinicalBERT']['ids'])} patients")
    else:
        print(f"  WARNING: ClinicalBERT model not found at {CBERT_KNN_PATH}")

    return models


def knn_predict_progression(knn_obj, k=10, n_sample=None):
    ids = knn_obj["ids"]
    X = knn_obj["X"]
    model = knn_obj["model"]
    meta = knn_obj["meta"]

    n = len(ids) if n_sample is None else min(n_sample, len(ids))
    indices_to_eval = np.random.RandomState(42).choice(len(ids), size=n, replace=False)

    y_true = []
    y_pred = []
    y_scores = []
    avg_distances = []
    stage_concordances = []

    for i in tqdm(indices_to_eval, desc="    Evaluating"):
        dists, nbr_idxs = model.kneighbors(X[i].reshape(1, -1), n_neighbors=k + 1)
        dists = dists[0]
        nbr_idxs = nbr_idxs[0]

        mask = nbr_idxs != i
        dists = dists[mask][:k]
        nbr_idxs = nbr_idxs[mask][:k]

        query_row = meta.iloc[i]
        true_label = int(query_row["progressed"])
        true_stage = query_row["max_stage"]

        nbr_labels = meta.iloc[nbr_idxs]["progressed"].values
        nbr_stages = meta.iloc[nbr_idxs]["max_stage"].values

        pred_label = int(np.round(nbr_labels.mean()))
        prog_score = nbr_labels.mean()
        stage_match = np.mean(nbr_stages == true_stage)

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(prog_score)
        avg_distances.append(dists.mean())
        stage_concordances.append(stage_match)

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_scores": np.array(y_scores),
        "avg_distances": np.array(avg_distances),
        "stage_concordances": np.array(stage_concordances),
    }


def figure1_cohort_overview(models):
    model_name = list(models.keys())[0]
    meta = models[model_name]["meta"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # A: CKD stage distribution
    ax = axes[0]
    stage_counts = meta["max_stage"].value_counts().sort_index()
    colors = [NONPROG_COLOR if s <= 3 else PROG_COLOR for s in stage_counts.index]
    bars = ax.bar(stage_counts.index.astype(str), stage_counts.values, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Max CKD Stage")
    ax.set_ylabel("Number of Patients")
    ax.set_title("A) CKD Stage Distribution")
    for bar, count in zip(bars, stage_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=8)

    # B: Pie chart
    ax = axes[1]
    prog_counts = meta["progressed"].value_counts()
    labels = ["Non-progressor\n(Stage \u2264 3)", "Progressor\n(Stage > 3)"]
    colors_pie = [NONPROG_COLOR, PROG_COLOR]
    ax.pie(
        [prog_counts.get(0, 0), prog_counts.get(1, 0)],
        labels=labels, colors=colors_pie, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title("B) Progression Status")

    # C: Time to progression
    ax = axes[2]
    progressors = meta[meta["progressed"] == 1]
    t4 = progressors["time_to_stage4"].dropna()
    if len(t4) > 0:
        ax.hist(t4 / 365.25, bins=50, color=PROG_COLOR, alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax.axvline(t4.median() / 365.25, color="black", linestyle="--", linewidth=1.5,
                   label=f"Median: {t4.median()/365.25:.1f} yrs")
        ax.set_xlabel("Time to Stage 4 (years)")
        ax.set_ylabel("Number of Patients")
        ax.set_title("C) Time to Progression")
        ax.legend()

    plt.tight_layout()
    path = f"{FIG_DIR}/fig1_cohort_overview.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def figure2_roc_pr(models, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"Gemma": GEMMA_COLOR, "ClinicalBERT": CBERT_COLOR}

    # A: ROC
    ax = axes[0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
        auc = roc_auc_score(res["y_true"], res["y_scores"])
        ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A) ROC Curve \u2014 CKD Progression Prediction")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # B: PR
    ax = axes[1]
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_scores"])
        ap = average_precision_score(res["y_true"], res["y_scores"])
        ax.plot(rec, prec, color=colors[name], linewidth=2, label=f"{name} (AP = {ap:.3f})")
    baseline = results[list(results.keys())[0]]["y_true"].mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B) Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = f"{FIG_DIR}/fig2_roc_pr_comparison.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def figure3_metrics_bar(results):
    metrics_data = {}
    for name, res in results.items():
        y_true, y_pred, y_scores = res["y_true"], res["y_pred"], res["y_scores"]
        metrics_data[name] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "AUROC": roc_auc_score(y_true, y_scores),
            "AUPRC": average_precision_score(y_true, y_scores),
        }

    metric_names = list(list(metrics_data.values())[0].keys())
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Gemma": GEMMA_COLOR, "ClinicalBERT": CBERT_COLOR}

    for i, (name, metrics) in enumerate(metrics_data.items()):
        vals = [metrics[m] for m in metric_names]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors.get(name, f"C{i}"),
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("KNN Classification Performance: Gemma vs ClinicalBERT")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim([0, 1.12])
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    path = f"{FIG_DIR}/fig3_metrics_comparison.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def figure4_confusion(results):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        labels_display = ["Non-progressor", "Progressor"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_display, rotation=45, ha="right")
        ax.set_yticklabels(labels_display)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}")

    plt.suptitle("Confusion Matrices \u2014 KNN CKD Progression Prediction (k=10)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = f"{FIG_DIR}/fig4_confusion_matrices.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def figure5_concordance_distance(models):
    colors = {"Gemma": GEMMA_COLOR, "ClinicalBERT": CBERT_COLOR}
    k_values = [5, 10, 20, 50]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A: Stage concordance vs k
    ax = axes[0]
    for name, knn_obj in models.items():
        concordances = []
        for k in k_values:
            res = knn_predict_progression(knn_obj, k=k, n_sample=2000)
            concordances.append(res["stage_concordances"].mean())
        ax.plot(k_values, concordances, "o-", color=colors[name], linewidth=2,
                markersize=8, label=name)
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Mean Stage Concordance")
    ax.set_title("A) Neighbor Stage Concordance vs k")
    ax.legend()
    ax.set_xticks(k_values)

    # B: Distance distribution
    ax = axes[1]
    for name, knn_obj in models.items():
        res = knn_predict_progression(knn_obj, k=10, n_sample=5000)
        ax.hist(res["avg_distances"], bins=50, alpha=0.5, color=colors[name],
                label=f"{name} (\u03bc={res['avg_distances'].mean():.3f})",
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Cosine Distance to k=10 Neighbors")
    ax.set_ylabel("Number of Patients")
    ax.set_title("B) Neighbor Distance Distribution")
    ax.legend()

    plt.tight_layout()
    path = f"{FIG_DIR}/fig5_concordance_distance.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def figure6_neighbor_example(models):
    shared_patients = set()
    all_ids = [set(m["ids"].values) for m in models.values()]
    if len(all_ids) >= 2:
        shared_patients = all_ids[0].intersection(all_ids[1])
    else:
        shared_patients = all_ids[0]

    model_name = list(models.keys())[0]
    meta = models[model_name]["meta"]
    progressors_in_shared = meta[(meta["PatientID"].isin(shared_patients)) & (meta["progressed"] == 1)]

    if len(progressors_in_shared) == 0:
        print("  No shared progressors found, skipping figure 6")
        return

    query_pid = progressors_in_shared.iloc[0]["PatientID"]

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, knn_obj) in zip(axes, models.items()):
        ids = knn_obj["ids"]
        X = knn_obj["X"]
        model_knn = knn_obj["model"]
        meta_knn = knn_obj["meta"]

        idx = ids.index[ids == query_pid].tolist()[0]
        dists, nbr_idxs = model_knn.kneighbors(X[idx].reshape(1, -1), n_neighbors=6)

        mask = nbr_idxs[0] != idx
        dists = dists[0][mask][:5]
        nbr_idxs = nbr_idxs[0][mask][:5]

        similarities = 1 - dists
        stages = meta_knn.iloc[nbr_idxs]["max_stage"].values
        prog_status = meta_knn.iloc[nbr_idxs]["progressed"].values

        y_pos = np.arange(5)
        bar_colors = [PROG_COLOR if p == 1 else NONPROG_COLOR for p in prog_status]

        ax.barh(y_pos, similarities, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.6)

        for j, (sim, stage) in enumerate(zip(similarities, stages)):
            ax.text(sim + 0.005, j, f"Stage {stage:.0f} | sim={sim:.3f}", va="center", fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Rank {i+1}" for i in range(5)])
        ax.set_xlabel("Cosine Similarity")
        ax.set_title(f"{name}")
        ax.invert_yaxis()
        ax.set_xlim([0, 1.15])

    query_info = meta.loc[meta["PatientID"] == query_pid].iloc[0]
    fig.suptitle(
        f"Top-5 Neighbors for Patient {query_pid} (Stage {query_info['max_stage']:.0f}, Progressor)",
        fontsize=13, y=1.02,
    )

    legend_elements = [
        Patch(facecolor=PROG_COLOR, label="Progressor"),
        Patch(facecolor=NONPROG_COLOR, label="Non-progressor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = f"{FIG_DIR}/fig6_neighbor_example.png"
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


def generate_metrics_table(results):
    rows = []
    for name, res in results.items():
        y_true, y_pred, y_scores = res["y_true"], res["y_pred"], res["y_scores"]
        rows.append({
            "Model": name,
            "Embedding Dim": 1024 if name == "Gemma" else 768,
            "Accuracy": f"{accuracy_score(y_true, y_pred):.3f}",
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
            "Recall": f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
            "F1": f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
            "AUROC": f"{roc_auc_score(y_true, y_scores):.3f}",
            "AUPRC": f"{average_precision_score(y_true, y_scores):.3f}",
        })

    df = pd.DataFrame(rows)

    csv_path = f"{FIG_DIR}/metrics_comparison_table.csv"
    df.to_csv(csv_path, index=False)

    latex_path = f"{FIG_DIR}/metrics_comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(df.to_latex(index=False, escape=False))

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {latex_path}")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING CAPSTONE FIGURES - GEMMA vs CLINICALBERT KNN COMPARISON")
    print("=" * 80)

    models = load_models()

    if len(models) == 0:
        print("\nERROR: No KNN models found. Run the KNN analysis scripts first.")
        exit(1)

    print("\n[1/7] Evaluating KNN prediction performance (k=10)...")
    print("       (sampling 5,000 patients per model for speed)")
    results = {}
    for name, knn_obj in models.items():
        print(f"\n  Evaluating {name}...")
        results[name] = knn_predict_progression(knn_obj, k=10, n_sample=5000)

    print("\n[2/7] Figure 1: Cohort Overview...")
    figure1_cohort_overview(models)

    print("\n[3/7] Figure 2: ROC & PR Curves...")
    figure2_roc_pr(models, results)

    print("\n[4/7] Figure 3: Metrics Comparison Bar Chart...")
    figure3_metrics_bar(results)

    print("\n[5/7] Figure 4: Confusion Matrices...")
    figure4_confusion(results)

    print("\n[6/7] Figure 5: Stage Concordance & Distance Distributions...")
    figure5_concordance_distance(models)

    print("\n[7/7] Figure 6: Neighbor Example Comparison...")
    figure6_neighbor_example(models)

    print("\n[TABLE] Generating metrics comparison table...")
    metrics_df = generate_metrics_table(results)
    print("\n" + metrics_df.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"✓ ALL FIGURES GENERATED!")
    print(f"  Output directory: {FIG_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"    - {f}")
    print("=" * 80)