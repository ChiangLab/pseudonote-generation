#!/usr/bin/env python
"""
Regenerate ONLY fig1_cohort_overview with updated Panel C.
Run from: /opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline/
"""

import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline"
GEMMA_KNN_PATH = f"{BASE_DIR}/outputs/knn_analysis_gemma/knn_model_gemma.joblib"
FIG_DIR = f"{BASE_DIR}/outputs/figures"

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

PROG_COLOR = "#E53935"
NONPROG_COLOR = "#43A047"

print("Loading Gemma KNN model for metadata...")
knn_obj = joblib.load(GEMMA_KNN_PATH)
meta = knn_obj["meta"]
print(f"  Loaded metadata for {len(meta)} patients")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel A: CKD stage distribution (unchanged)
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

# Panel B: Pie chart (unchanged)
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

# Panel C: Time to progression - UPDATED
ax = axes[2]
progressors = meta[meta["progressed"] == 1]
t4 = progressors["time_to_stage4"].dropna()
# Filter to patients with actual progression (time > 0)
t4_positive = t4[t4 > 0]
print(f"  Total progressors: {len(progressors)}")
print(f"  Progressors with time_to_stage4 data: {len(t4)}")
print(f"  Progressors with time_to_stage4 > 0: {len(t4_positive)}")
print(f"  Progressors with time_to_stage4 == 0: {(t4 == 0).sum()}")
if len(t4_positive) > 0:
    t4_years = t4_positive / 365.25
    bins = np.arange(0, 21, 0.5)  # 0 to 20 years in half-year bins
    ax.hist(t4_years, bins=bins, color=PROG_COLOR, alpha=0.7,
            edgecolor="white", linewidth=0.5)
    median_yrs = t4_years.median()
    ax.axvline(median_yrs, color="black", linestyle="--", linewidth=1.5,
               label=f"Median: {median_yrs:.1f} yrs")
    ax.set_xlabel("Time to Stage 4 (years)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("C) Time to Progression")
    ax.set_xlim([0, 20])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.legend()

plt.tight_layout()
path = f"{FIG_DIR}/fig1_cohort_overview.png"
plt.savefig(path)
plt.savefig(path.replace(".png", ".pdf"))
plt.close()
print(f"\nSaved: {path}")
print(f"Saved: {path.replace('.png', '.pdf')}")