"""
Data Preparation Module for Hospital Course Summarization.

This module handles the preparation of clinical notes data for fine-tuning
an LLM on hospital course summarization task, following the methodology
from "Harmonising the Clinical Melody" (arXiv:2409.14638).

The paper uses MIMIC-III data where:
- INPUT: Concatenated clinical notes from various categories (Nursing, Physician, 
  Radiology, ECG, Respiratory, etc.)
- OUTPUT: "Brief Hospital Course" section from discharge summaries
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Generator
import pandas as pd
from tqdm import tqdm
import jsonlines

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, DataConfig


@dataclass
class ClinicalRecord:
    """Represents a single clinical record for fine-tuning."""
    hadm_id: str  # Hospital admission ID
    clinical_notes: str  # Concatenated clinical notes (input)
    brief_hospital_course: str  # Target summary (output)
    note_types_included: List[str]  # Types of notes in the input


class DataPreparator:
    """
    Prepares clinical data for Azure OpenAI fine-tuning.
    
    Follows the data format required by Azure AI Foundry:
    - JSONL format with chat completion structure
    - System prompt defining the task
    - User message with clinical notes
    - Assistant message with the summary
    """
    
    SYSTEM_PROMPT = """You are a clinical documentation specialist. Your task is to generate a 
concise "Brief Hospital Course" summary from the provided clinical notes. This summary should:
1. Capture the essential clinical trajectory of the patient's hospital stay
2. Include key diagnoses, procedures, and treatments
3. Highlight significant clinical events and their resolutions
4. Be suitable for clinical coding purposes
5. Use appropriate medical terminology
6. Be organized chronologically when relevant

Output only the Brief Hospital Course summary without any additional commentary."""

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the data preparator."""
        self.config = config or get_config().data
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.config.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_mimic_notes(self, notes_file: str) -> pd.DataFrame:
        """
        Load clinical notes from MIMIC-III NOTEEVENTS table format.
        
        Expected columns:
        - HADM_ID: Hospital admission ID
        - CATEGORY: Note category (e.g., 'Nursing', 'Physician')
        - TEXT: Note content
        - CHARTDATE: Date of note
        
        Args:
            notes_file: Path to the notes CSV file
            
        Returns:
            DataFrame with clinical notes
        """
        print(f"Loading clinical notes from {notes_file}...")
        df = pd.read_csv(notes_file, low_memory=False)
        
        # Filter to relevant note types
        df = df[df['CATEGORY'].isin(self.config.clinical_note_types)]
        print(f"Loaded {len(df)} notes across categories: {df['CATEGORY'].unique().tolist()}")
        
        return df
    
    def load_discharge_summaries(self, discharge_file: str) -> pd.DataFrame:
        """
        Load and parse discharge summaries to extract Brief Hospital Course.
        
        Args:
            discharge_file: Path to discharge summaries CSV
            
        Returns:
            DataFrame with HADM_ID and brief_hospital_course columns
        """
        print(f"Loading discharge summaries from {discharge_file}...")
        df = pd.read_csv(discharge_file, low_memory=False)
        
        # Extract Brief Hospital Course section
        df['brief_hospital_course'] = df['TEXT'].apply(self._extract_hospital_course)
        
        # Filter out records without valid hospital course
        df = df[df['brief_hospital_course'].notna() & (df['brief_hospital_course'].str.len() > 50)]
        print(f"Extracted {len(df)} valid hospital course summaries")
        
        return df[['HADM_ID', 'brief_hospital_course']]
    
    def _extract_hospital_course(self, discharge_text: str) -> Optional[str]:
        """
        Extract the Brief Hospital Course section from a discharge summary.
        
        The section typically appears between "Brief Hospital Course:" and 
        the next section header (e.g., "Medications on Admission:", "Discharge Medications:").
        """
        if pd.isna(discharge_text):
            return None
        
        text = str(discharge_text)
        
        # Common section headers that mark the end of Brief Hospital Course
        end_markers = [
            "Medications on Admission:",
            "Discharge Medications:",
            "Discharge Disposition:",
            "Discharge Diagnosis:",
            "Discharge Condition:",
            "Discharge Instructions:",
            "Followup Instructions:"
        ]
        
        # Find start of Brief Hospital Course
        start_markers = ["Brief Hospital Course:", "BRIEF HOSPITAL COURSE:", "Hospital Course:"]
        start_idx = -1
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                start_idx = idx + len(marker)
                break
        
        if start_idx == -1:
            return None
        
        # Find end of section
        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker, start_idx)
            if idx != -1 and idx < end_idx:
                end_idx = idx
        
        hospital_course = text[start_idx:end_idx].strip()
        
        # Validate extracted content
        if len(hospital_course) < 50:  # Too short to be valid
            return None
            
        return hospital_course
    
    def create_training_records(
        self, 
        notes_df: pd.DataFrame, 
        summaries_df: pd.DataFrame
    ) -> List[ClinicalRecord]:
        """
        Create training records by matching notes to summaries.
        
        Following the paper's methodology:
        - Concatenate all clinical notes for each admission
        - Pair with the Brief Hospital Course from discharge summary
        
        Args:
            notes_df: DataFrame with clinical notes
            summaries_df: DataFrame with extracted hospital course summaries
            
        Returns:
            List of ClinicalRecord objects
        """
        print("Creating training records...")
        records = []
        
        # Get unique admissions with summaries
        valid_admissions = set(summaries_df['HADM_ID'].unique())
        
        # Group notes by admission
        grouped_notes = notes_df.groupby('HADM_ID')
        
        for hadm_id, group in tqdm(grouped_notes, desc="Processing admissions"):
            if hadm_id not in valid_admissions:
                continue
            
            # Sort notes chronologically if CHARTDATE exists
            if 'CHARTDATE' in group.columns:
                group = group.sort_values('CHARTDATE')
            
            # Concatenate clinical notes with category headers
            clinical_text_parts = []
            note_types = group['CATEGORY'].unique().tolist()
            
            for category in note_types:
                category_notes = group[group['CATEGORY'] == category]['TEXT'].tolist()
                clinical_text_parts.append(f"\n=== {category.upper()} NOTES ===\n")
                clinical_text_parts.extend([str(note) for note in category_notes if pd.notna(note)])
            
            clinical_text = "\n".join(clinical_text_parts)
            
            # Get corresponding summary
            summary = summaries_df[summaries_df['HADM_ID'] == hadm_id]['brief_hospital_course'].iloc[0]
            
            records.append(ClinicalRecord(
                hadm_id=str(hadm_id),
                clinical_notes=clinical_text,
                brief_hospital_course=summary,
                note_types_included=note_types
            ))
        
        print(f"Created {len(records)} training records")
        return records
    
    def format_for_azure_openai(self, record: ClinicalRecord) -> Dict:
        """
        Format a clinical record for Azure OpenAI fine-tuning.
        
        Returns the chat completion format required by Azure:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generate a Brief Hospital Course summary for the following clinical notes:\n\n{record.clinical_notes}"
                },
                {
                    "role": "assistant",
                    "content": record.brief_hospital_course
                }
            ]
        }
    
    def split_and_save(
        self, 
        records: List[ClinicalRecord],
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Split records into train/validation/test sets and save as JSONL files.
        
        Args:
            records: List of clinical records
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
        """
        if shuffle:
            random.seed(seed)
            random.shuffle(records)
        
        n = len(records)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.validation_ratio)
        
        splits = {
            self.config.train_file: records[:train_end],
            self.config.validation_file: records[train_end:val_end],
            self.config.test_file: records[val_end:]
        }
        
        for filename, split_records in splits.items():
            filepath = Path(self.config.processed_data_dir) / filename
            print(f"Saving {len(split_records)} records to {filepath}...")
            
            with jsonlines.open(filepath, mode='w') as writer:
                for record in split_records:
                    formatted = self.format_for_azure_openai(record)
                    writer.write(formatted)
        
        # Save statistics
        self._save_statistics(splits)
    
    def _save_statistics(self, splits: Dict[str, List[ClinicalRecord]]):
        """Save dataset statistics."""
        stats = {
            "total_records": sum(len(v) for v in splits.values()),
            "splits": {
                k.replace('.jsonl', ''): {
                    "count": len(v),
                    "note_types": list(set(
                        nt for r in v for nt in r.note_types_included
                    ))
                }
                for k, v in splits.items()
            }
        }
        
        stats_path = Path(self.config.processed_data_dir) / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")
    
    def create_sample_data(self, num_samples: int = 20):
        """
        Create sample synthetic data for testing the pipeline.
        
        This generates realistic-looking clinical data structure
        for pipeline validation before using real MIMIC data.
        """
        print(f"Creating {num_samples} sample records for testing...")
        
        sample_records = []
        
        # Sample clinical scenarios
        scenarios = [
            {
                "condition": "Acute Myocardial Infarction",
                "notes": "Patient admitted with chest pain radiating to left arm. ECG shows ST elevation in leads V1-V4. Troponin elevated at 2.5 ng/mL.",
                "course": "Patient was admitted with STEMI. Underwent emergent cardiac catheterization with PCI to LAD with drug-eluting stent placement. Post-procedure course uncomplicated. Started on dual antiplatelet therapy with aspirin and clopidogrel. Echo showed EF of 45%."
            },
            {
                "condition": "Community Acquired Pneumonia",
                "notes": "72-year-old presenting with fever, productive cough, and shortness of breath. CXR shows right lower lobe consolidation. WBC 15,000.",
                "course": "Patient admitted with CAP, started on IV ceftriaxone and azithromycin. Respiratory status improved over 3 days. Transitioned to oral antibiotics. Oxygen weaned to room air. Discharged with 5-day course of oral antibiotics."
            },
            {
                "condition": "Diabetic Ketoacidosis",
                "notes": "Type 1 diabetic with nausea, vomiting, and altered mental status. Glucose 450 mg/dL, pH 7.1, ketones positive.",
                "course": "Patient admitted to ICU for DKA management. Started on insulin drip and aggressive IV fluid resuscitation. Anion gap closed within 18 hours. Transitioned to subcutaneous insulin. Endocrine consulted for insulin regimen optimization."
            }
        ]
        
        for i in range(num_samples):
            scenario = scenarios[i % len(scenarios)]
            
            # Create synthetic clinical notes
            nursing_notes = f"""
Patient Assessment:
Vital Signs: BP 138/85, HR 88, RR 18, Temp 37.2C, SpO2 96% on RA
Chief Complaint: {scenario['condition']}
Assessment Notes: {scenario['notes']}
Interventions: IV access established, labs drawn, medications administered per orders.
"""
            
            physician_notes = f"""
History of Present Illness:
{scenario['notes']}

Physical Examination:
General: Alert, oriented, in mild distress
Cardiovascular: Regular rate and rhythm
Respiratory: Clear to auscultation bilaterally
Abdomen: Soft, non-tender

Assessment and Plan:
1. {scenario['condition']} - will manage per protocol
"""
            
            combined_notes = f"""
=== NURSING NOTES ===
{nursing_notes}

=== PHYSICIAN NOTES ===
{physician_notes}
"""
            
            sample_records.append(ClinicalRecord(
                hadm_id=f"SAMPLE_{i+1:04d}",
                clinical_notes=combined_notes,
                brief_hospital_course=scenario['course'],
                note_types_included=["Nursing", "Physician"]
            ))
        
        # Save as training data
        self.split_and_save(sample_records, shuffle=True)
        print("Sample data created successfully!")
        
        return sample_records


def main():
    """Main function to run data preparation."""
    preparator = DataPreparator()
    
    # Check if we have real MIMIC data or should create sample data
    raw_data_path = Path(preparator.config.raw_data_dir)
    notes_file = raw_data_path / "NOTEEVENTS.csv"
    
    if notes_file.exists():
        # Process real MIMIC data
        print("Found MIMIC data, processing...")
        notes_df = preparator.load_mimic_notes(str(notes_file))
        summaries_df = preparator.load_discharge_summaries(
            str(raw_data_path / "DISCHARGE_SUMMARIES.csv")
        )
        records = preparator.create_training_records(notes_df, summaries_df)
        preparator.split_and_save(records)
    else:
        # Create sample data for testing
        print("No MIMIC data found. Creating sample data for pipeline testing...")
        print(f"Place MIMIC-III NOTEEVENTS.csv in {raw_data_path} to use real data.")
        preparator.create_sample_data(num_samples=50)


if __name__ == "__main__":
    main()
