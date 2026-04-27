#!/usr/bin/env python

import polars as pl
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import pickle
from typing import Optional


class MEDSEmbeddingGenerator:
    
    # Predefined model configurations
    MODEL_CONFIGS = {
        "clinicalbert": {
            "path": "/opt/data/commonfilesharePHI/slee/MEME/clinicalBERT-emily",
            "embed_dim": 768,
            "max_length": 512,
            "pooling": "cls"
        },
        "qwen3-8b": {
            "path": "Qwen/Qwen3-Embedding-8B",
            "embed_dim": 4096,
            "max_length": 8192,
            "pooling": "mean"
        },
        "embeddinggemma": {
            "path": "/opt/data/commonfilesharePHI/slee/MEME/embeddinggemma",
            "embed_dim": 1024,
            "max_length": 512,
            "pooling": "mean"
        }
    }
    
    def __init__(
        self,
        pseudonotes_path: str,
        output_dir: str,
        model_name: str = "clinicalbert",
        custom_model_path: Optional[str] = None,
        custom_embed_dim: Optional[int] = None,
        custom_max_length: Optional[int] = None,
        custom_pooling: Optional[str] = None,
        batch_size: int = 8,
        cuda_device: int = 2,
        checkpoint_every: int = 1000  # Save every N patients
    ):
        self.pseudonotes_path = pseudonotes_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        
        # Load model configuration
        if custom_model_path:
            self.model_path = custom_model_path
            self.embed_dim = custom_embed_dim or 768
            self.max_length = custom_max_length or 512
            self.pooling = custom_pooling or "cls"
            model_display_name = custom_model_path
        elif model_name in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model_name]
            self.model_path = config["path"]
            self.embed_dim = config["embed_dim"]
            self.max_length = config["max_length"]
            self.pooling = config["pooling"]
            model_display_name = model_name
        else:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.MODEL_CONFIGS.keys())}\n"
                f"Or provide custom_model_path for a custom model."
            )
        
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"[INFO] Initializing MEDS Embedding Generator", flush=True)
        print(f"       Pseudonotes: {pseudonotes_path}", flush=True)
        print(f"       Model: {model_display_name}", flush=True)
        print(f"       Model path: {self.model_path}", flush=True)
        print(f"       Embedding dim: {self.embed_dim}", flush=True)
        print(f"       Max length: {self.max_length}", flush=True)
        print(f"       Pooling: {self.pooling}", flush=True)
        print(f"       Device: {self.device}", flush=True)
        print(f"       Batch size: {batch_size}", flush=True)
        print(f"       Checkpoint every: {checkpoint_every} patients", flush=True)
        print(f"       Output: {output_dir}", flush=True)
    
    def load_model(self):
        print(f"[INFO] Loading model from: {self.model_path}", flush=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"       Model loaded successfully", flush=True)
    
    def load_pseudonotes(self):
        print(f"[INFO] Loading pseudonotes...", flush=True)
        
        self.pseudonotes_df = pl.read_parquet(self.pseudonotes_path)
        
        print(f"       Loaded {len(self.pseudonotes_df)} patient pseudonotes", flush=True)
    
    def mean_pool(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_batch_embeddings(self, texts: list) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            embeddings = self.mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        if embeddings.size(1) > self.embed_dim:
            embeddings = embeddings[:, :self.embed_dim]
        elif embeddings.size(1) < self.embed_dim:
            pad = self.embed_dim - embeddings.size(1)
            embeddings = torch.nn.functional.pad(embeddings, (0, pad), value=0)
        
        return embeddings.cpu().numpy()
    
    def check_existing_embeddings(self):
        """Check which patients already have embeddings saved"""
        existing_patients = set()
        
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                patient_dir = os.path.join(self.output_dir, item)
                if os.path.isdir(patient_dir) and item not in ['checkpoints']:
                    npy_file = os.path.join(patient_dir, f"{item}.npy")
                    if os.path.exists(npy_file):
                        existing_patients.add(item)
        
        return existing_patients
    
    def generate_and_save_embeddings(self):
        """Generate embeddings and save them IMMEDIATELY as individual files"""
        print(f"[INFO] Generating and saving embeddings incrementally...", flush=True)
        
        # Check what's already done
        existing_patients = self.check_existing_embeddings()
        if existing_patients:
            print(f"       Found {len(existing_patients)} existing embeddings - will skip these", flush=True)
        
        patient_ids = self.pseudonotes_df["subject_id"].to_list()
        pseudonotes = self.pseudonotes_df["pseudonote"].to_list()
        
        total_patients = len(patient_ids)
        patients_processed = 0
        patients_skipped = 0
        
        # Create progress tracking file
        progress_file = os.path.join(self.checkpoint_dir, "progress.txt")
        
        # Process in batches
        pbar = tqdm(total=total_patients, desc="Processing patients")
        
        i = 0
        while i < total_patients:
            # Get batch of patients
            batch_end = min(i + self.batch_size, total_patients)
            batch_ids = patient_ids[i:batch_end]
            batch_texts = pseudonotes[i:batch_end]
            
            # Skip patients that already have embeddings
            indices_to_process = []
            ids_to_process = []
            texts_to_process = []
            
            for idx, (pid, text) in enumerate(zip(batch_ids, batch_texts)):
                if pid not in existing_patients:
                    indices_to_process.append(idx)
                    ids_to_process.append(pid)
                    texts_to_process.append(text)
                else:
                    patients_skipped += 1
            
            # Generate embeddings for this batch
            if texts_to_process:
                try:
                    batch_embeddings = self.get_batch_embeddings(texts_to_process)
                    
                    # IMMEDIATELY save each embedding
                    for patient_id, embedding in zip(ids_to_process, batch_embeddings):
                        patient_folder = os.path.join(self.output_dir, str(patient_id))
                        os.makedirs(patient_folder, exist_ok=True)
                        
                        npy_path = os.path.join(patient_folder, f"{patient_id}.npy")
                        np.save(npy_path, embedding)
                        
                        patients_processed += 1
                        existing_patients.add(patient_id)  # Mark as done
                    
                    # Update progress file every checkpoint
                    if patients_processed % self.checkpoint_every == 0:
                        with open(progress_file, 'w') as f:
                            f.write(f"Processed: {patients_processed}/{total_patients}\n")
                            f.write(f"Skipped (already done): {patients_skipped}\n")
                            f.write(f"Last patient: {patient_id}\n")
                        print(f"\n       [CHECKPOINT] Saved {patients_processed} embeddings so far", flush=True)
                
                except Exception as e:
                    print(f"\n       [ERROR] Failed on batch starting at patient {i}: {e}", flush=True)
                    print(f"       Continuing with next batch...", flush=True)
            
            pbar.update(batch_end - i)
            i = batch_end
        
        pbar.close()
        
        # Final progress update
        with open(progress_file, 'w') as f:
            f.write(f"COMPLETED\n")
            f.write(f"Processed: {patients_processed}/{total_patients}\n")
            f.write(f"Skipped (already done): {patients_skipped}\n")
        
        print(f"\n       ✓ Processed {patients_processed} new embeddings", flush=True)
        print(f"       ✓ Skipped {patients_skipped} existing embeddings", flush=True)
        print(f"       ✓ Total embeddings saved: {len(existing_patients)}", flush=True)
        
        return existing_patients
    
    def create_metadata(self, patient_ids):
        """Create metadata CSV after all embeddings are saved"""
        print(f"[INFO] Creating metadata file...", flush=True)
        
        try:
            import pandas as pd
            
            # Get metadata for patients that have embeddings
            patient_id_set = set(patient_ids)
            
            meta_data = []
            for row in self.pseudonotes_df.iter_rows(named=True):
                if row['subject_id'] in patient_id_set:
                    meta_data.append({
                        'subject_id': row['subject_id'],
                        'first_event': row.get('first_event'),
                        'last_event': row.get('last_event')
                    })
            
            meta_df = pd.DataFrame(meta_data)
            metadata_path = os.path.join(self.output_dir, "embedding_metadata.csv")
            meta_df.to_csv(metadata_path, index=False)
            print(f"       Saved metadata to: {metadata_path}", flush=True)
            
        except Exception as e:
            print(f"       WARNING: Could not save metadata: {e}", flush=True)
    
    def run(self):
        print("\n" + "="*80, flush=True)
        print("MEDS EMBEDDING GENERATION PIPELINE (INCREMENTAL SAVING)", flush=True)
        print("="*80 + "\n", flush=True)
        
        self.load_model()
        self.load_pseudonotes()
        
        # Generate and save incrementally
        completed_patients = self.generate_and_save_embeddings()
        
        # Create metadata file
        self.create_metadata(completed_patients)
        
        print("\n" + "="*80, flush=True)
        print("✓ Embedding generation complete!", flush=True)
        print(f"✓ All embeddings saved to: {self.output_dir}", flush=True)
        print("="*80 + "\n", flush=True)
        
        return {
            'total_patients': len(completed_patients),
            'output_dir': self.output_dir
        }


def main():
    BASE_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline"
    PSEUDONOTES_PATH = f"{BASE_DIR}/outputs/pseudonotes/pseudonotes_train.parquet"
    OUTPUT_DIR = f"{BASE_DIR}/outputs/embeddings"
    
    generator = MEDSEmbeddingGenerator(
        pseudonotes_path=PSEUDONOTES_PATH,
        output_dir=OUTPUT_DIR,
        model_name="clinicalbert",
        batch_size=16,
        cuda_device=2,
        checkpoint_every=1000
    )
    
    result = generator.run()
    
    print("\nEmbedding Summary:")
    print(f"  Total patients: {result['total_patients']}")


if __name__ == "__main__":
    main()