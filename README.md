# MEDS-Compliant EHR Embedding Pipeline

Generalizable pipeline for converting MEDS-formatted EHR data into text-based pseudonotes and generating embeddings using various language models.

## Overview

This pipeline transforms structured EHR data (labs, diagnoses, medications, procedures) into natural language pseudonotes that are then embedded using large language models for downstream machine learning tasks.

**Key Features:**
- MEDS-compliant (Medical Event Data Standard)
- Agnostic to disease/condition (works with any EHR data)
- Multiple embedding model support (ClinicalBERT, Qwen3, EmbeddingGemma, custom models)
- Efficient batch processing using Polars
- Outputs both parquet and individual .npy embeddings

## Pipeline Structure
```
meds_embedding_pipeline/
├── data/
│   └── toy_meds/              # Synthetic MEDS data for testing
├── src/
│   ├── generate_toy_meds.py   # Generate synthetic MEDS data
│   ├── meds_to_pseudonotes.py # Convert MEDS → pseudonotes
│   └── embedding_generator.py # Generate embeddings from pseudonotes
├── outputs/
│   ├── pseudonotes/           # Generated pseudonotes
│   ├── embeddings_clinicalbert/  # ClinicalBERT embeddings
│   ├── embeddings_qwen3/      # Qwen3 embeddings
│   └── embeddings_gemma/      # EmbeddingGemma embeddings
└── legacy/
    └── embedding_gen_pl.py    # Original CKD-specific script
```

## MEDS Format Requirements

**Events (train/0.parquet, tuning/0.parquet, held_out/0.parquet):**
- `subject_id`: Patient identifier
- `time`: Event timestamp
- `code`: Medical code (ICD-10, LOINC, RxNorm, CPT, etc.)
- `numeric_value`: Numeric value (for labs, vitals)

**Codes (metadata/codes.parquet):**
- `code`: Medical code
- `description`: Human-readable description
- `parent_codes`: Code system (e.g., ["ICD10"], ["LOINC"])

## Usage

### 1. Generate Synthetic Test Data (Optional)
```bash
conda activate meds_embedding_pipeline
python src/generate_toy_meds.py
```

### 2. Generate Pseudonotes from MEDS Data
```bash
python src/meds_to_pseudonotes.py
```

Outputs:
- `outputs/pseudonotes/pseudonotes_train.parquet` - Full pseudonotes
- `outputs/pseudonotes/metadata_train.csv` - Patient metadata
- `outputs/pseudonotes/sample_pseudonotes_train.txt` - Human-readable samples

### 3. Generate Embeddings

#### Option A: Using ClinicalBERT (default, 768-dim)
```python
from src.embedding_generator import MEDSEmbeddingGenerator

generator = MEDSEmbeddingGenerator(
    pseudonotes_path="outputs/pseudonotes/pseudonotes_train.parquet",
    output_dir="outputs/embeddings_clinicalbert",
    model_name="clinicalbert",
    batch_size=16,
    cuda_device=2
)
embeddings_df = generator.run()
```

#### Option B: Using Qwen3-8B (4096-dim, state-of-the-art)
```python
generator = MEDSEmbeddingGenerator(
    pseudonotes_path="outputs/pseudonotes/pseudonotes_train.parquet",
    output_dir="outputs/embeddings_qwen3",
    model_name="qwen3-8b",
    batch_size=4,  # Smaller batch for larger model
    cuda_device=2
)
embeddings_df = generator.run()
```

#### Option C: Using EmbeddingGemma (1024-dim, efficient)
```python
generator = MEDSEmbeddingGenerator(
    pseudonotes_path="outputs/pseudonotes/pseudonotes_train.parquet",
    output_dir="outputs/embeddings_gemma",
    model_name="embeddinggemma",
    batch_size=16,
    cuda_device=2
)
embeddings_df = generator.run()
```

#### Option D: Using a Custom Model
```python
generator = MEDSEmbeddingGenerator(
    pseudonotes_path="outputs/pseudonotes/pseudonotes_train.parquet",
    output_dir="outputs/embeddings_custom",
    custom_model_path="sentence-transformers/all-MiniLM-L6-v2",
    custom_embed_dim=384,
    custom_max_length=512,
    custom_pooling="mean",  # Options: "cls" or "mean"
    batch_size=32,
    cuda_device=2
)
embeddings_df = generator.run()
```

## Supported Embedding Models

| Model | Dimensions | Max Length | Pooling | Best For | Batch Size |
|-------|-----------|------------|---------|----------|------------|
| **clinicalbert** | 768 | 512 | CLS | Clinical text, baseline | 16 |
| **qwen3-8b** | 4096 | 8192 | Mean | Long documents, SOTA performance | 4 |
| **embeddinggemma** | 1024 | 512 | Mean | Efficient, good balance | 16 |
| **custom** | Variable | Variable | Variable | Any HuggingFace model | Adjust |

### Model Selection Guidelines

- **ClinicalBERT**: Best for clinical notes, pretrained on medical text, standard choice
- **Qwen3-8B**: State-of-the-art performance, handles very long pseudonotes (8K tokens), requires more GPU memory
- **EmbeddingGemma**: Good balance of performance and efficiency, newer architecture
- **Custom**: Use any HuggingFace model for specific use cases

### Memory Requirements

Approximate GPU memory needed (batch size 1):
- ClinicalBERT: ~2GB
- EmbeddingGemma: ~1.5GB
- Qwen3-8B: ~16GB (use smaller batch sizes)

Adjust `batch_size` based on your GPU memory. Larger models require smaller batch sizes.

## Example Pseudonote
```
Patient 1 is Male, Black or African American.

On 2022-05-10, the patient had the following records:
 - Outpatient encounter
 - Diagnosis: Type 2 diabetes mellitus without complications (ICD-10: E11.9)
 - Diagnosis: Essential (primary) hypertension (ICD-10: I10)
 - Lab: Creatinine [Mass/volume] in Serum or Plasma: 0.80
 - Lab: Glucose [Mass/volume] in Serum or Plasma: 138.22
 - Medication administered: Lisinopril 10 MG Oral Tablet
```

## Customization

### Using Your Own MEDS Data

Edit `src/meds_to_pseudonotes.py` and `src/embedding_generator.py` to point to your data:
```python
MEDS_DIR = "/path/to/your/meds/data"
meds_data_path = f"{MEDS_DIR}/train/0.parquet"
codes_path = f"{MEDS_DIR}/metadata/codes.parquet"
```

### Comparing Multiple Models

Run the pipeline with different models to compare embeddings:
```bash
# Generate embeddings with all three models
python -c "
from src.embedding_generator import MEDSEmbeddingGenerator

for model in ['clinicalbert', 'qwen3-8b', 'embeddinggemma']:
    print(f'\\nGenerating embeddings with {model}...')
    gen = MEDSEmbeddingGenerator(
        pseudonotes_path='outputs/pseudonotes/pseudonotes_train.parquet',
        output_dir=f'outputs/embeddings_{model}',
        model_name=model,
        batch_size=16 if model != 'qwen3-8b' else 4,
        cuda_device=2
    )
    gen.run()
"
```

## Technical Details

- **Pooling Strategies**:
  - CLS: Uses [CLS] token embedding (BERT-style models)
  - Mean: Average pooling across all tokens (often better for retrieval)
- **Data Processing**: Polars (faster than Pandas)
- **Token Limits**: Automatically truncates long pseudonotes
- **GPU Support**: Automatic CUDA detection, configurable device

## Next Steps

1. **Apply to Real Data**: Replace toy data with UCLA DDR MEDS output
2. **Model Comparison**: Generate embeddings with multiple models and evaluate performance
3. **Downstream Tasks**: Use embeddings for similarity search, clustering, prediction
4. **OLLAMA Integration**: If local LLM deployment succeeds, adapt pipeline
5. **Share Pipeline**: MEDS compliance enables cross-institutional use

## Dependencies
```bash
conda install polars numpy pandas
pip install transformers torch tqdm pyarrow
```

## References

- MEDS Standard: https://github.com/Medical-Event-Data-Standard/MEDS-DEV
- MEDS Transforms: https://meds-transforms.readthedocs.io/
- Qwen3-Embedding: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- EmbeddingGemma: https://huggingface.co/google/embeddinggemma-300m
- ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT