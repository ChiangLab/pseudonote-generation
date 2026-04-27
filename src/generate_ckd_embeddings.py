#!/usr/bin/env python

import polars as pl
import os
import sys
sys.path.append('/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline/src')
from embedding_generator import MEDSEmbeddingGenerator

# Configuration for CKD data
OUTPUT_BASE = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline/outputs/ckd_embeddings_full"
# Load and prepare CKD pseudonotes
print("[INFO] Loading CKD pseudonotes from meta_v3_all.csv...", flush=True)
ckd_df = pl.read_csv(
    "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full_v3_icd_stage_filter/meta_v3_all.csv",
    separator="$"
)

print(f"       Loaded {len(ckd_df)} encounters", flush=True)
print(f"       Unique patients: {ckd_df['PatientID'].n_unique()}", flush=True)
print(f"       Columns: {ckd_df.columns}", flush=True)
print(f"       Loaded {len(ckd_df)} encounters", flush=True)
print(f"       Columns: {ckd_df.columns}", flush=True)
print(f"       Sample:\n{ckd_df.head()}", flush=True)

# Prepare data in format expected by embedding generator
pseudonotes_df = ckd_df.select([
    pl.col("PatientID").alias("subject_id"),
    pl.col("enc_summary").alias("pseudonote"),
    pl.col("META_1").alias("encounter_id")
]).with_columns([
    pl.lit(None).alias("first_event"),
    pl.lit(None).alias("last_event")
])

# Create output directory and save parquet
os.makedirs(OUTPUT_BASE, exist_ok=True)
temp_pseudonotes_path = f"{OUTPUT_BASE}/ckd_pseudonotes_full.parquet"
pseudonotes_df.write_parquet(temp_pseudonotes_path)
print(f"[INFO] Saved prepared pseudonotes to: {temp_pseudonotes_path}", flush=True)

# Model configurations
models = [
    {
        "name": "embeddinggemma",
        "batch_size": 16,
        "output_suffix": "gemma"
    }
]

# Generate embeddings with all three models
for model_config in models:
    print("\n" + "="*80, flush=True)
    print(f"GENERATING EMBEDDINGS WITH {model_config['name'].upper()}", flush=True)
    print("="*80 + "\n", flush=True)
    
    output_dir = f"{OUTPUT_BASE}/{model_config['output_suffix']}"
    
    try:
        generator = MEDSEmbeddingGenerator(
            pseudonotes_path=temp_pseudonotes_path,
            output_dir=output_dir,
            model_name=model_config["name"],
            batch_size=model_config["batch_size"],
            cuda_device=2
        )
        
        result = generator.run()
        
        print(f"\n✓ {model_config['name']} embeddings complete!", flush=True)
        print(f"  Output: {output_dir}", flush=True)
        print(f"  Total patients: {result['total_patients']}", flush=True)
        
    except Exception as e:
        print(f"\n✗ Error with {model_config['name']}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*80, flush=True)
print("ALL EMBEDDING GENERATION COMPLETE!", flush=True)
print("="*80, flush=True)
print(f"\nOutputs saved to: {OUTPUT_BASE}/", flush=True)
print("  - clinicalbert/ (768-dim)", flush=True)
print("  - qwen3/ (4096-dim)", flush=True)
print("  - gemma/ (1024-dim)", flush=True)