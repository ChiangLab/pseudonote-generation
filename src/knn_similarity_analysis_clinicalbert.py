#!/usr/bin/env python
"""
KNN Similarity Analysis - ClinicalBERT Embeddings
Mirrors the Gemma KNN analysis for side-by-side comparison.
ClinicalBERT embeddings are (num_encounters, 768) per patient and need mean pooling.
"""

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
import joblib

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline"
CLINICALBERT_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full_v3_icd_stage_filter"
META_CSV_FULL = f"{CLINICALBERT_DIR}/meta_v3_all.csv"
OUTPUT_DIR = f"{BASE_DIR}/outputs/knn_analysis_clinicalbert"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("KNN SIMILARITY ANALYSIS - CLINICALBERT EMBEDDINGS")
print("=" * 80)

# ============================================================
# [1/5] Load ClinicalBERT embeddings from .npy files
# ============================================================
print("\n[1/5] Loading ClinicalBERT embeddings from .npy files...")

all_items = os.listdir(CLINICALBERT_DIR)
patient_folders = [
    d for d in all_items
    if os.path.isdir(os.path.join(CLINICALBERT_DIR, d))
    and d not in ["checkpoints", "__pycache__"]
]
print(f"      Found {len(patient_folders)} patient folders")

patient_embeddings = {}
skipped = 0
embedding_dim = None

for patient_id in tqdm(patient_folders, desc="      Loading & pooling"):
    npy_file = os.path.join(CLINICALBERT_DIR, patient_id, f"{patient_id}.npy")

    if not os.path.exists(npy_file):
        skipped += 1
        continue

    emb = np.load(npy_file)

    if emb.ndim == 2:
        patient_embeddings[patient_id] = emb.mean(axis=0)
        if embedding_dim is None:
            embedding_dim = emb.shape[1]
    elif emb.ndim == 1:
        patient_embeddings[patient_id] = emb
        if embedding_dim is None:
            embedding_dim = emb.shape[0]
    else:
        skipped += 1
        continue

print(f"      Loaded embeddings for {len(patient_embeddings)} patients")
print(f"      Embedding dimension: {embedding_dim}")
if skipped > 0:
    print(f"      Skipped {skipped} patients (missing/invalid .npy)")

patient_ids_with_embeddings = set(patient_embeddings.keys())

# ============================================================
# [2/5] Load metadata
# ============================================================
print("\n[2/5] Loading metadata for patients with embeddings...")
meta_df = pd.read_csv(META_CSV_FULL, sep="$")
print(f"      Loaded {len(meta_df)} total encounters")
meta_df = meta_df[meta_df["PatientID"].isin(patient_ids_with_embeddings)]
print(f"      Filtered to {len(meta_df)} encounters from {meta_df['PatientID'].nunique()} patients")

# ============================================================
# [3/5] Process metadata
# ============================================================
print("\n[3/5] Processing metadata...")
meta_df["date"] = pd.to_datetime(meta_df["date"])
meta_df["dx_date"] = meta_df["date"].where(meta_df["CKD_stage_numeric"] > 3)
meta_df["stage5_date"] = meta_df["date"].where(meta_df["CKD_stage_numeric"] > 4)

time_df = (
    meta_df.groupby("PatientID", as_index=False)
    .agg({"date": "min", "dx_date": "min", "stage5_date": "min"})
    .rename(columns={"date": "first_date", "dx_date": "stage4_date"})
)
time_df["time_to_stage4"] = (time_df["stage4_date"] - time_df["first_date"]).dt.days
time_df["time_to_stage5"] = (time_df["stage5_date"] - time_df["first_date"]).dt.days

patient_summary = (
    meta_df.groupby("PatientID")
    .agg({"CKD_stage_numeric": "max", "max_stage": "max"})
    .rename(columns={"CKD_stage_numeric": "final_stage"})
)

patient_summary = patient_summary.merge(
    time_df[["PatientID", "time_to_stage4", "time_to_stage5"]],
    left_index=True,
    right_on="PatientID",
    how="left",
).set_index("PatientID")

patient_summary["progressed"] = (patient_summary["max_stage"] > 3).astype(int)

print(f"      Processed {len(patient_summary)} unique patients")
print(f"      Progressors: {patient_summary['progressed'].sum()}")
print(f"      Non-progressors: {(1 - patient_summary['progressed']).sum()}")

# ============================================================
# [4/5] Build embedding matrix
# ============================================================
print("\n[4/5] Building embedding matrix...")

aligned_patients = [pid for pid in patient_summary.index if pid in patient_embeddings]
X = np.vstack([patient_embeddings[pid] for pid in aligned_patients])
patient_summary_aligned = patient_summary.loc[aligned_patients]

print(f"      Aligned {len(aligned_patients)} patients")
print(f"      Final embedding shape: {X.shape}")

data_for_knn = pd.DataFrame(X)
data_for_knn.insert(0, "PatientID", aligned_patients)
data_for_knn = data_for_knn.merge(
    patient_summary_aligned.reset_index()[
        ["PatientID", "max_stage", "progressed", "time_to_stage4", "time_to_stage5"]
    ],
    on="PatientID",
    how="left",
)
print(f"      KNN dataset ready: {len(data_for_knn)} patients x {X.shape[1]} dimensions")


# ============================================================
# KNN functions
# ============================================================
def prepare_knn(
    df,
    id_col="PatientID",
    meta_cols=["max_stage", "progressed", "time_to_stage4", "time_to_stage5"],
    metric="cosine",
    n_neighbors=50,
    nmf_components=32,
):
    exclude = set([id_col] + meta_cols)
    embedding_cols = [c for c in df.columns if c not in exclude]

    ids = df[id_col].reset_index(drop=True)
    X = df[embedding_cols].to_numpy()

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = imputer.fit_transform(X)

    X = MinMaxScaler().fit_transform(X)

    nmf = NMF(n_components=nmf_components, init="nndsvda", max_iter=2000, random_state=42)
    X = nmf.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X) - 1), metric=metric)
    knn.fit(X)

    meta = df[[id_col] + meta_cols].reset_index(drop=True)

    return {
        "X": X,
        "ids": ids,
        "model": knn,
        "meta": meta,
        "metric": metric,
        "id_col": id_col,
    }


def find_neighbors(query_id, knn_obj, k=10, include_self=False):
    ids = knn_obj["ids"]
    X = knn_obj["X"]
    model = knn_obj["model"]
    metric = knn_obj["metric"]
    meta = knn_obj["meta"]
    id_col = knn_obj["id_col"]

    matches = ids.index[ids == query_id].tolist()
    if len(matches) == 0:
        raise ValueError(f"Patient {query_id} not found")

    idx = matches[0]
    ask_k = k + (0 if include_self else 1)
    distances, indices = model.kneighbors(X[idx].reshape(1, -1), n_neighbors=ask_k)

    idxs = indices[0].tolist()
    dists = distances[0].tolist()

    if not include_self and idx in idxs:
        j = idxs.index(idx)
        idxs.pop(j)
        dists.pop(j)

    idxs = idxs[:k]
    dists = dists[:k]

    neighbors = meta.iloc[idxs].copy().reset_index(drop=True)
    neighbors.insert(0, "rank", np.arange(1, len(idxs) + 1))
    neighbors["distance"] = np.round(dists, 6)

    if metric == "cosine":
        neighbors["similarity"] = np.round(1.0 - neighbors["distance"], 6)

    neighbors = neighbors.rename(columns={id_col: "neighbor_id"})
    neighbors.insert(0, "query_id", query_id)

    return neighbors


# ============================================================
# [5/5] Build KNN model
# ============================================================
print("\n[5/5] Building KNN model...")

n_patients = len(data_for_knn)
nmf_comp = min(32, n_patients - 1)

knn_obj = prepare_knn(
    df=data_for_knn,
    n_neighbors=min(50, n_patients - 1),
    nmf_components=nmf_comp,
)
print(f"      KNN model built with {len(knn_obj['ids'])} patients")

knn_save_path = f"{OUTPUT_DIR}/knn_model_clinicalbert.joblib"
joblib.dump(knn_obj, knn_save_path)
print(f"      Saved to: {knn_save_path}")

print("\n" + "=" * 80)
print("✓ KNN MODEL READY!")
print("=" * 80)

# ============================================================
# Example neighbor searches
# ============================================================
print("\n[EXAMPLES] Finding neighbors for sample patients...\n")

sample_patients = ["M4067615", "M4067809", "M4068374", "M4068576"]
fallback_progressors = knn_obj["meta"].loc[knn_obj["meta"]["progressed"] == 1, "PatientID"].head(2).tolist()
fallback_non_progressors = knn_obj["meta"].loc[knn_obj["meta"]["progressed"] == 0, "PatientID"].head(2).tolist()

query_patients = []
for pid in sample_patients:
    if pid in knn_obj["ids"].values:
        query_patients.append(pid)

if len(query_patients) < 4:
    query_patients = fallback_progressors + fallback_non_progressors

for query_patient in query_patients:
    print(f"\n{'=' * 80}")
    print(f"Query Patient: {query_patient}")

    query_info = knn_obj["meta"][knn_obj["meta"]["PatientID"] == query_patient].iloc[0]
    print(f"  Max Stage: {query_info['max_stage']}")
    print(f"  Progressed: {'Yes' if query_info['progressed'] else 'No'}")
    if query_info["progressed"]:
        t4 = query_info["time_to_stage4"]
        print(f"  Time to Stage 4: {t4:.0f} days" if pd.notna(t4) else "  Time to Stage 4: N/A")

    print(f"\nTop 5 Most Similar Patients:")
    print("-" * 80)

    try:
        nbrs = find_neighbors(query_id=query_patient, knn_obj=knn_obj, k=5, include_self=False)
        print(
            nbrs[
                ["rank", "neighbor_id", "similarity", "max_stage", "progressed", "time_to_stage4"]
            ].to_string(index=False)
        )
        nbrs.to_csv(f"{OUTPUT_DIR}/neighbors_{query_patient}.csv", index=False)
    except Exception as e:
        print(f"Error: {e}")

# ============================================================
# Save summary
# ============================================================
summary = {
    "total_patients": len(knn_obj["ids"]),
    "progressors": int(knn_obj["meta"]["progressed"].sum()),
    "non_progressors": int((1 - knn_obj["meta"]["progressed"]).sum()),
    "embedding_dim_original": embedding_dim,
    "embedding_dim_after_nmf": nmf_comp,
    "metric": "cosine",
    "aggregation_method": "mean_across_encounters",
}

with open(f"{OUTPUT_DIR}/analysis_summary.txt", "w") as f:
    f.write("KNN SIMILARITY ANALYSIS SUMMARY - CLINICALBERT\n")
    f.write("=" * 50 + "\n\n")
    for key, val in summary.items():
        f.write(f"{key}: {val}\n")

print(f"\n\n{'=' * 80}")
print(f"✓ Analysis complete!")
print(f"  Results saved to: {OUTPUT_DIR}/")
print(f"  KNN model: {knn_save_path}")
print("=" * 80)
print(f"\nSummary saved to: {OUTPUT_DIR}/analysis_summary.txt")