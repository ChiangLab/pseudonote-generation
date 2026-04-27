# Generate synthetic MEDS-compliant data for testing the embedding pipeline.
# We want to create realistic patient trajectories with diagnoses, labs, medications, and procedures.


import polars as pl
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
N_PATIENTS = 100
OUTPUT_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline/data/toy_meds"
SEED = 42

np.random.seed(SEED)

# MEDS-compliant vocabulary
# ICD-10 codes for common conditions
ICD10_CODES = {
    "I10": "Essential (primary) hypertension",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E78.5": "Hyperlipidemia, unspecified",
    "N18.3": "Chronic kidney disease, stage 3",
    "N18.4": "Chronic kidney disease, stage 4",
    "N18.5": "Chronic kidney disease, stage 5",
    "N18.6": "End stage renal disease",
    "I50.9": "Heart failure, unspecified",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "I25.10": "Atherosclerotic heart disease",
}

# LOINC codes for common labs
LOINC_CODES = {
    "2160-0": "Creatinine [Mass/volume] in Serum or Plasma",
    "33914-3": "Glomerular filtration rate/1.73 sq M.predicted",
    "2345-7": "Glucose [Mass/volume] in Serum or Plasma",
    "4548-4": "Hemoglobin A1c/Hemoglobin.total in Blood",
    "2951-2": "Sodium [Moles/volume] in Serum or Plasma",
    "2823-3": "Potassium [Moles/volume] in Serum or Plasma",
    "17861-6": "Calcium [Mass/volume] in Serum or Plasma",
    "2777-1": "Phosphate [Mass/volume] in Serum or Plasma",
    "1742-6": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "1920-8": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
}

# RxNorm codes for medications
RXNORM_CODES = {
    "197361": "Lisinopril 10 MG Oral Tablet",
    "860975": "Metformin hydrochloride 500 MG Oral Tablet",
    "617311": "Atorvastatin 20 MG Oral Tablet",
    "308136": "Furosemide 40 MG Oral Tablet",
    "197803": "Amlodipine 5 MG Oral Tablet",
    "313782": "Aspirin 81 MG Oral Tablet",
    "104080": "Insulin, Regular, Human 100 UNT/ML Injectable Solution",
}

# CPT codes for procedures
CPT_CODES = {
    "99213": "Office or other outpatient visit, established patient, 20-29 minutes",
    "99214": "Office or other outpatient visit, established patient, 30-39 minutes",
    "36415": "Routine venipuncture for collection of specimen",
    "80053": "Comprehensive metabolic panel",
    "93000": "Electrocardiogram, routine ECG with at least 12 leads",
    "36556": "Insertion of non-tunneled centrally inserted central venous catheter",
}

# Encounter types
ENCOUNTER_CODES = {
    "ENC_OUTPATIENT": "Outpatient encounter",
    "ENC_EMERGENCY": "Emergency department encounter",
    "ENC_INPATIENT": "Inpatient hospital encounter",
    "ENC_TELEHEALTH": "Telehealth encounter",
}

# Demographics
DEMO_CODES = {
    "GENDER_M": "Male",
    "GENDER_F": "Female",
    "RACE_WHITE": "White or Caucasian",
    "RACE_BLACK": "Black or African American",
    "RACE_ASIAN": "Asian",
    "RACE_HISPANIC": "Hispanic or Latino",
    "RACE_OTHER": "Other race",
}

# Generate the codes.parquet file with code, description, parent_codes

def generate_codes_metadata():
    
    
    all_codes = []
    
    # Add ICD-10 codes
    for code, desc in ICD10_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["ICD10"]
        })
    
    # Add Labs codes
    for code, desc in LOINC_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["LOINC"]
        })
    
    # Add Medications codes
    for code, desc in RXNORM_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["RXNORM"]
        })
    
    # Add Basic Procedures codes
    for code, desc in CPT_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["CPT"]
        })
    
    # Add encounter codes
    for code, desc in ENCOUNTER_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["ENCOUNTER"]
        })
    
    # Add demographic codes
    for code, desc in DEMO_CODES.items():
        all_codes.append({
            "code": code,
            "description": desc,
            "parent_codes": ["DEMOGRAPHICS"]
        })
    
    codes_df = pl.DataFrame(all_codes)
    return codes_df

#    Generate demographic events for a patient

def generate_patient_demographics(patient_id):
    
    events = []
    base_date = datetime(2020, 1, 1)
    
    # Gender
    gender = np.random.choice(["GENDER_M", "GENDER_F"])
    events.append({
        "subject_id": patient_id,
        "time": base_date,
        "code": gender,
        "numeric_value": None
    })
    
    # Race
    race = np.random.choice([
        "RACE_WHITE", "RACE_BLACK", "RACE_ASIAN", 
        "RACE_HISPANIC", "RACE_OTHER"
    ], p=[0.6, 0.15, 0.1, 0.1, 0.05])
    events.append({
        "subject_id": patient_id,
        "time": base_date,
        "code": race,
        "numeric_value": None
    })
    
    return events

#     Generate a realistic clinical trajectory for one patient

def generate_patient_trajectory(patient_id):
    
    events = []
    
    # Start date between 2020-2023
    start_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095))
    current_date = start_date
    
    # Add demographics
    events.extend(generate_patient_demographics(patient_id))
    
    # Determine CKD stage progression (some patients progress, some don't)
    has_ckd = np.random.random() < 0.4  # 40% have CKD
    if has_ckd:
        ckd_stage = np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2])
    
    # Generate 5-15 encounters over 1-3 years
    n_encounters = np.random.randint(5, 16)
    
    for enc_num in range(n_encounters):
        # Encounter event
        enc_type = np.random.choice(
            list(ENCOUNTER_CODES.keys()),
            p=[0.7, 0.15, 0.1, 0.05]
        )
        events.append({
            "subject_id": patient_id,
            "time": current_date,
            "code": enc_type,
            "numeric_value": None
        })
        
        # Add diagnoses
        n_diagnoses = np.random.randint(1, 4)
        common_dx = ["I10", "E11.9", "E78.5"]  # HTN, DM, Hyperlipidemia
        
        for dx_code in np.random.choice(common_dx, size=min(n_diagnoses, len(common_dx)), replace=False):
            events.append({
                "subject_id": patient_id,
                "time": current_date + timedelta(minutes=5),
                "code": dx_code,
                "numeric_value": None
            })
        
        # Add CKD diagnosis if applicable
        if has_ckd:
            ckd_code = f"N18.{ckd_stage}"
            events.append({
                "subject_id": patient_id,
                "time": current_date + timedelta(minutes=5),
                "code": ckd_code,
                "numeric_value": None
            })
        
        # Add lab results
        n_labs = np.random.randint(3, 8)
        for lab_code in np.random.choice(list(LOINC_CODES.keys()), size=n_labs, replace=False):
            # Generate realistic values
            if lab_code == "2160-0":  # Creatinine
                value = np.random.uniform(0.8, 3.5 if has_ckd else 1.3)
            elif lab_code == "33914-3":  # eGFR
                value = np.random.uniform(15, 90 if has_ckd else 120)
            elif lab_code == "2345-7":  # Glucose
                value = np.random.uniform(70, 200)
            elif lab_code == "4548-4":  # HbA1c
                value = np.random.uniform(5.0, 9.5)
            elif lab_code == "2951-2":  # Sodium
                value = np.random.uniform(135, 145)
            elif lab_code == "2823-3":  # Potassium
                value = np.random.uniform(3.5, 5.5)
            else:
                value = np.random.uniform(10, 100)
            
            events.append({
                "subject_id": patient_id,
                "time": current_date + timedelta(minutes=30),
                "code": lab_code,
                "numeric_value": value
            })
        
        # Add medications
        n_meds = np.random.randint(2, 5)
        for med_code in np.random.choice(list(RXNORM_CODES.keys()), size=n_meds, replace=False):
            events.append({
                "subject_id": patient_id,
                "time": current_date + timedelta(hours=1),
                "code": med_code,
                "numeric_value": None
            })
        
        # Add procedures
        if np.random.random() < 0.3:  # 30% chance of procedure
            proc_code = np.random.choice(list(CPT_CODES.keys()))
            events.append({
                "subject_id": patient_id,
                "time": current_date + timedelta(hours=2),
                "code": proc_code,
                "numeric_value": None
            })
        
        # Next encounter in 30-90 days
        current_date += timedelta(days=np.random.randint(30, 91))
        
        # CKD progression (10% chance to worsen per encounter)
        if has_ckd and ckd_stage < 6 and np.random.random() < 0.1:
            ckd_stage += 1
    
    return events

#     Generate events for all patients

def generate_all_patients():
    
    all_events = []
    
    print(f"Generating {N_PATIENTS} synthetic patients...")
    for patient_id in range(1, N_PATIENTS + 1):
        if patient_id % 20 == 0:
            print(f"  Generated {patient_id}/{N_PATIENTS} patients")
        
        patient_events = generate_patient_trajectory(patient_id)
        all_events.extend(patient_events)
    
    # Convert to DataFrame
    events_df = pl.DataFrame(all_events)
    
    # Sort by subject_id and time
    events_df = events_df.sort(["subject_id", "time"])
    
    return events_df

#     Split patients into train/tuning/held_out sets in accordance with MEDS

def split_data(events_df, train_frac=0.7, tuning_frac=0.15, held_out_frac=0.15):
    
    unique_patients = events_df["subject_id"].unique().sort()
    n_patients = len(unique_patients)
    
    n_train = int(n_patients * train_frac)
    n_tuning = int(n_patients * tuning_frac)
    
    train_patients = unique_patients[:n_train]
    tuning_patients = unique_patients[n_train:n_train + n_tuning]
    held_out_patients = unique_patients[n_train + n_tuning:]
    
    train_df = events_df.filter(pl.col("subject_id").is_in(train_patients))
    tuning_df = events_df.filter(pl.col("subject_id").is_in(tuning_patients))
    held_out_df = events_df.filter(pl.col("subject_id").is_in(held_out_patients))
    
    # Create subject splits metadata
    splits_data = []
    for pid in train_patients:
        splits_data.append({"subject_id": pid, "split": "train"})
    for pid in tuning_patients:
        splits_data.append({"subject_id": pid, "split": "tuning"})
    for pid in held_out_patients:
        splits_data.append({"subject_id": pid, "split": "held_out"})
    
    splits_df = pl.DataFrame(splits_data)
    
    return train_df, tuning_df, held_out_df, splits_df


def main():
    """Generate and save synthetic MEDS data"""
    
    print("Generating MEDS-compliant synthetic data...")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(f"{OUTPUT_DIR}/metadata", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/tuning", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/held_out", exist_ok=True)
    
    # Generate codes metadata
    print("\n1. Generating codes metadata...")
    codes_df = generate_codes_metadata()
    codes_df.write_parquet(f"{OUTPUT_DIR}/metadata/codes.parquet")
    print(f"   Saved {len(codes_df)} codes to metadata/codes.parquet")
    print(f"   Sample codes:\n{codes_df.head()}")
    
    # Generate patient events
    print("\n2. Generating patient trajectories...")
    events_df = generate_all_patients()
    print(f"   Generated {len(events_df)} total events")
    print(f"   Sample events:\n{events_df.head()}")
    
    # Split into train/tuning/held_out
    print("\n3. Splitting into train/tuning/held_out...")
    train_df, tuning_df, held_out_df, splits_df = split_data(events_df)
    
    # Save splits
    train_df.write_parquet(f"{OUTPUT_DIR}/train/0.parquet")
    print(f"   Train: {len(train_df)} events, {train_df['subject_id'].n_unique()} patients")
    
    tuning_df.write_parquet(f"{OUTPUT_DIR}/tuning/0.parquet")
    print(f"   Tuning: {len(tuning_df)} events, {tuning_df['subject_id'].n_unique()} patients")
    
    held_out_df.write_parquet(f"{OUTPUT_DIR}/held_out/0.parquet")
    print(f"   Held-out: {len(held_out_df)} events, {held_out_df['subject_id'].n_unique()} patients")
    
    splits_df.write_parquet(f"{OUTPUT_DIR}/metadata/subject_splits.parquet")
    print(f"   Saved subject splits to metadata/subject_splits.parquet")
    
    print("\n" + "=" * 60)
    print("✓ Synthetic MEDS data generation complete!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - metadata/codes.parquet (code definitions)")
    print("  - metadata/subject_splits.parquet (train/tuning/held_out assignments)")
    print("  - train/0.parquet (training events)")
    print("  - tuning/0.parquet (tuning events)")
    print("  - held_out/0.parquet (held-out events)")


if __name__ == "__main__":
    main()