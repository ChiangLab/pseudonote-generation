import polars as pl
import os
from datetime import datetime
from typing import Optional, Dict


class MEDSPseudonoteGenerator:
    
    def __init__(
        self,
        meds_data_path: str,
        codes_path: str,
        output_dir: str,
        split: str = "train"
    ):
        self.meds_data_path = meds_data_path
        self.codes_path = codes_path
        self.output_dir = output_dir
        self.split = split
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] Initializing MEDS Pseudonote Generator")
        print(f"       Data: {meds_data_path}")
        print(f"       Codes: {codes_path}")
        print(f"       Output: {output_dir}")
    
    # Load MEDS events and code definitions
    def load_data(self):
        print("[INFO] Loading MEDS data...")
        
        self.events_df = pl.read_parquet(self.meds_data_path)
        print(f"       Loaded {len(self.events_df)} events for {self.events_df['subject_id'].n_unique()} patients")
        
        self.codes_df = pl.read_parquet(self.codes_path)
        print(f"       Loaded {len(self.codes_df)} code definitions")
        
        # Create code lookup dictionaries
        self.code_map = dict(zip(
            self.codes_df["code"],
            self.codes_df["description"]
        ))
        
        self.parent_map = dict(zip(
            self.codes_df["code"],
            self.codes_df["parent_codes"]
        ))
    
    # Extract demographic information for each patient
    def extract_demographics(self):
        print("[INFO] Extracting demographics...")
        
        # Filter demographic events
        demo_events = (
            self.events_df
            .filter(
                pl.col("code").is_in(
                    [code for code, parents in self.parent_map.items() 
                     if "DEMOGRAPHICS" in parents]
                )
            )
            .with_columns(
                pl.col("code").replace(self.code_map).alias("description")
            )
        )
        
        # Create demographic sentence for each patient
        demo_df = (
            demo_events
            .group_by("subject_id")
            .agg([
                pl.col("description").str.concat(", ").alias("demo_text")
            ])
            .with_columns(
                pl.format(
                    "Patient {} is {}.",
                    pl.col("subject_id"),
                    pl.col("demo_text")
                ).alias("demographic_sentence")
            )
            .select(["subject_id", "demographic_sentence"])
        )
        
        self.demographic_map = dict(zip(
            demo_df["subject_id"],
            demo_df["demographic_sentence"]
        ))
        
        print(f"       Generated demographics for {len(self.demographic_map)} patients")
    
    # Determine event type from parent codes
    def classify_event_type(self, code: str) -> str:
        parents = self.parent_map.get(code, [])
        
        if "ICD10" in parents:
            return "diagnosis"
        elif "LOINC" in parents:
            return "lab"
        elif "RXNORM" in parents:
            return "medication"
        elif "CPT" in parents:
            return "procedure"
        elif "ENCOUNTER" in parents:
            return "encounter"
        elif "DEMOGRAPHICS" in parents:
            return "demographics"
        else:
            return "other"
    
    # Convert a single event into a natural language sentence
    def event_to_sentence(self, event_type: str, code: str, description: str, numeric_value: Optional[float]) -> str:
        if event_type == "diagnosis":
            return f" - Diagnosis: {description} (ICD-10: {code})"
        
        elif event_type == "lab":
            if numeric_value is not None:
                return f" - Lab: {description}: {numeric_value:.2f}"
            else:
                return f" - Lab: {description}"
        
        elif event_type == "medication":
            return f" - Medication administered: {description}"
        
        elif event_type == "procedure":
            return f" - Procedure performed: {description}"
        
        elif event_type == "encounter":
            return f" - {description}"
        
        else:
            return f" - Event: {description}"
    
    # Generate pseudonotes from MEDS events
    def generate_pseudonotes(self):
        print("[INFO] Generating pseudonotes...")
        
        # Filter out demographics (already handled)
        clinical_events = (
            self.events_df
            .filter(
                ~pl.col("code").is_in(
                    [code for code, parents in self.parent_map.items() 
                     if "DEMOGRAPHICS" in parents]
                )
            )
        )
        
        # Add descriptions and event types
        clinical_events = (
            clinical_events
            .with_columns([
                pl.col("code").replace(self.code_map, default="Unknown").alias("description"),
                pl.col("time").cast(pl.Datetime).alias("datetime"),
                pl.col("time").cast(pl.Datetime).dt.date().alias("event_date")
            ])
        )
        
        # Classify event types
        event_types = [
            self.classify_event_type(code) 
            for code in clinical_events["code"].to_list()
        ]
        clinical_events = clinical_events.with_columns(
            pl.Series("event_type", event_types)
        )
        
        # Convert events to sentences
        sentences = [
            self.event_to_sentence(
                row["event_type"],
                row["code"],
                row["description"],
                row["numeric_value"]
            )
            for row in clinical_events.iter_rows(named=True)
        ]
        clinical_events = clinical_events.with_columns(
            pl.Series("sentence", sentences)
        )
        
        # Group by patient and date
        daily_notes = (
            clinical_events
            .group_by(["subject_id", "event_date"])
            .agg([
                pl.col("sentence").str.concat("\n").alias("day_summary"),
                pl.col("datetime").min().alias("first_event_time")
            ])
            .sort(["subject_id", "first_event_time"])
            .with_columns(
                pl.format(
                    "On {}, the patient had the following records:\n{}",
                    pl.col("event_date"),
                    pl.col("day_summary")
                ).alias("daily_note")
            )
        )
        
        # Group by patient to create full pseudonotes
        patient_notes = (
            daily_notes
            .group_by("subject_id")
            .agg([
                pl.col("daily_note").str.concat("\n\n").alias("clinical_summary"),
                pl.col("first_event_time").min().alias("first_event"),
                pl.col("first_event_time").max().alias("last_event")
            ])
        )
        
        # Add demographics
        patient_notes = (
            patient_notes
            .with_columns(
                pl.col("subject_id")
                .replace(self.demographic_map, default="Patient demographics unavailable.")
                .alias("demographics")
            )
            .with_columns(
                pl.format(
                    "{}\n\n{}",
                    pl.col("demographics"),
                    pl.col("clinical_summary")
                ).alias("pseudonote")
            )
        )
        
        self.pseudonotes_df = patient_notes
        
        print(f"       Generated pseudonotes for {len(self.pseudonotes_df)} patients")
    
    # Save pseudonotes and metadata
    def save_pseudonotes(self):
        print("[INFO] Saving pseudonotes...")
        
        # Save full pseudonotes
        output_path = os.path.join(self.output_dir, f"pseudonotes_{self.split}.parquet")
        self.pseudonotes_df.write_parquet(output_path)
        print(f"       Saved to: {output_path}")
        
        # Save metadata (without full pseudonote text)
        metadata_path = os.path.join(self.output_dir, f"metadata_{self.split}.csv")
        (
            self.pseudonotes_df
            .select(["subject_id", "first_event", "last_event"])
            .write_csv(metadata_path)
        )
        print(f"       Saved metadata to: {metadata_path}")
        
        # Save sample pseudonotes for inspection
        sample_path = os.path.join(self.output_dir, f"sample_pseudonotes_{self.split}.txt")
        with open(sample_path, "w") as f:
            for i, row in enumerate(self.pseudonotes_df.head(3).iter_rows(named=True)):
                f.write(f"{'='*80}\n")
                f.write(f"PATIENT {row['subject_id']}\n")
                f.write(f"{'='*80}\n")
                f.write(row['pseudonote'])
                f.write(f"\n\n")
        
        print(f"       Saved sample to: {sample_path}")
    
    # Execute the full pipeline
    def run(self):
        print("\n" + "="*80)
        print("MEDS PSEUDONOTE GENERATION PIPELINE")
        print("="*80 + "\n")
        
        self.load_data()
        self.extract_demographics()
        self.generate_pseudonotes()
        self.save_pseudonotes()
        
        print("\n" + "="*80)
        print("✓ Pipeline complete!")
        print("="*80 + "\n")
        
        return self.pseudonotes_df


def main():
    # Configuration
    BASE_DIR = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/meds_embedding_pipeline"
    MEDS_DIR = f"{BASE_DIR}/data/toy_meds"
    OUTPUT_DIR = f"{BASE_DIR}/outputs/pseudonotes"
    
    # Process train split
    generator = MEDSPseudonoteGenerator(
        meds_data_path=f"{MEDS_DIR}/train/0.parquet",
        codes_path=f"{MEDS_DIR}/metadata/codes.parquet",
        output_dir=OUTPUT_DIR,
        split="train"
    )
    
    pseudonotes_df = generator.run()
    
    # Show sample
    print("\nSample pseudonote:")
    print("-" * 80)
    print(pseudonotes_df["pseudonote"][0])


if __name__ == "__main__":
    main()