"""
Synthetic Clinical Data Generation using Azure AI Foundry.

This module generates realistic clinical training data for the hospital course
summarization task using Azure AI Foundry models. This is useful when:
- Real clinical data (MIMIC-III) is not available
- Additional training samples are needed for data augmentation
- Testing the fine-tuning pipeline

The generator creates diverse clinical scenarios with:
- Multiple clinical note types (Nursing, Physician, Radiology, etc.)
- Varied medical conditions and complexities
- Realistic hospital course progression
- Ground truth Brief Hospital Course summaries
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from tqdm import tqdm
import jsonlines

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
from src.data_preparation import DataPreparator, ClinicalRecord

load_dotenv()


@dataclass
class ClinicalScenario:
    """Defines a clinical scenario for data generation."""
    condition: str
    category: str  # e.g., "Cardiology", "Pulmonology", "Infectious Disease"
    severity: str  # "mild", "moderate", "severe"
    patient_demographics: Dict[str, str] = field(default_factory=dict)
    comorbidities: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)


class SyntheticDataGenerator:
    """
    Generates synthetic clinical training data using Azure AI Foundry.
    
    Uses a two-step generation process:
    1. Generate detailed clinical notes from a scenario
    2. Generate the corresponding Brief Hospital Course summary
    
    This ensures the summary accurately reflects the generated notes,
    creating high-quality training pairs.
    """
    
    # Clinical specialties and common conditions
    CLINICAL_CATEGORIES = {
        "Cardiology": [
            "Acute Myocardial Infarction (STEMI)",
            "Acute Myocardial Infarction (NSTEMI)",
            "Unstable Angina",
            "Heart Failure Exacerbation",
            "Atrial Fibrillation with RVR",
            "Hypertensive Emergency",
            "Aortic Dissection",
            "Pericarditis",
            "Endocarditis",
            "Cardiogenic Shock"
        ],
        "Pulmonology": [
            "Community Acquired Pneumonia",
            "Hospital Acquired Pneumonia",
            "COPD Exacerbation",
            "Asthma Exacerbation",
            "Pulmonary Embolism",
            "Acute Respiratory Failure",
            "Pleural Effusion",
            "Pneumothorax",
            "COVID-19 Pneumonia",
            "Aspiration Pneumonia"
        ],
        "Gastroenterology": [
            "Acute Pancreatitis",
            "GI Bleeding (Upper)",
            "GI Bleeding (Lower)",
            "Acute Cholecystitis",
            "Bowel Obstruction",
            "Diverticulitis",
            "Hepatic Encephalopathy",
            "Acute Liver Failure",
            "C. difficile Colitis",
            "Acute Appendicitis"
        ],
        "Nephrology": [
            "Acute Kidney Injury",
            "Chronic Kidney Disease Exacerbation",
            "Nephrotic Syndrome",
            "Pyelonephritis",
            "Hyperkalemia",
            "Uremic Emergency",
            "Rhabdomyolysis"
        ],
        "Endocrinology": [
            "Diabetic Ketoacidosis",
            "Hyperosmolar Hyperglycemic State",
            "Severe Hypoglycemia",
            "Thyroid Storm",
            "Myxedema Coma",
            "Adrenal Crisis",
            "Hypercalcemia"
        ],
        "Infectious Disease": [
            "Sepsis",
            "Severe Sepsis with Organ Dysfunction",
            "Septic Shock",
            "Cellulitis with Bacteremia",
            "Osteomyelitis",
            "Meningitis",
            "Endocarditis",
            "Necrotizing Fasciitis"
        ],
        "Neurology": [
            "Ischemic Stroke",
            "Hemorrhagic Stroke",
            "Transient Ischemic Attack",
            "Status Epilepticus",
            "Guillain-Barré Syndrome",
            "Encephalitis",
            "Subarachnoid Hemorrhage"
        ],
        "Surgery": [
            "Post-operative Complications",
            "Surgical Site Infection",
            "Anastomotic Leak",
            "Post-operative Ileus",
            "Wound Dehiscence"
        ],
        "Orthopedics": [
            "Hip Fracture",
            "Total Hip Arthroplasty",
            "Total Knee Arthroplasty",
            "Spinal Fusion",
            "Open Fracture Repair"
        ],
        "Hematology/Oncology": [
            "Febrile Neutropenia",
            "Tumor Lysis Syndrome",
            "Deep Vein Thrombosis",
            "Sickle Cell Crisis",
            "Acute Leukemia"
        ]
    }
    
    COMORBIDITIES = [
        "Hypertension", "Type 2 Diabetes Mellitus", "Type 1 Diabetes Mellitus",
        "Coronary Artery Disease", "Congestive Heart Failure", "Atrial Fibrillation",
        "COPD", "Asthma", "Chronic Kidney Disease Stage III", "Chronic Kidney Disease Stage IV",
        "Cirrhosis", "Obesity (BMI > 30)", "Morbid Obesity (BMI > 40)",
        "Hyperlipidemia", "Hypothyroidism", "History of Stroke",
        "Peripheral Vascular Disease", "History of DVT/PE", "Active Smoker",
        "Former Smoker", "Alcohol Use Disorder", "Dementia"
    ]
    
    # Prompts for generation
    NOTES_GENERATION_PROMPT = """You are a clinical documentation expert. Generate realistic clinical notes for a hospitalized patient.

SCENARIO:
- Primary Condition: {condition}
- Category: {category}
- Severity: {severity}
- Patient: {age}-year-old {gender}
- Comorbidities: {comorbidities}
- Length of Stay: {los} days

Generate detailed clinical notes in the following format. Be specific with vital signs, lab values, medications, and clinical progression. Include realistic complications or responses to treatment.

=== NURSING NOTES ===
[Generate nursing documentation for each day, including:
- Vital signs with specific values
- Patient assessment
- Interventions performed
- Patient's response to treatment
- Intake/output if relevant]

=== PHYSICIAN NOTES ===
[Generate physician documentation including:
- Admission H&P summary
- Daily progress notes
- Assessment and plan
- Consultant notes if relevant]

=== RADIOLOGY/LAB RESULTS ===
[Include relevant imaging and laboratory findings with specific values]

Make the notes realistic, detailed, and medically accurate. Include specific medication names, dosages, and lab values."""

    SUMMARY_GENERATION_PROMPT = """You are a clinical coding specialist. Based on the following clinical notes, generate a "Brief Hospital Course" summary suitable for clinical coding and discharge documentation.

CLINICAL NOTES:
{clinical_notes}

Generate a Brief Hospital Course that:
1. Captures the essential clinical trajectory
2. Includes key diagnoses, procedures, and treatments with specific details
3. Highlights significant clinical events and their resolutions
4. Uses appropriate medical terminology
5. Is organized chronologically
6. Is suitable for clinical coding purposes (includes all billable diagnoses/procedures)

The summary should be 150-300 words, concise but comprehensive.

Output ONLY the Brief Hospital Course text, no additional commentary or headers."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the synthetic data generator."""
        self.config = get_config()
        self._setup_client()
        self.model = model_name or self.config.azure.model_deployment_name
        print(f"Synthetic Data Generator initialized with model: {self.model}")
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model deployment exists with a test call."""
        try:
            # Quick test to verify model exists
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print(f"✓ Model deployment '{self.model}' verified successfully")
        except Exception as e:
            error_msg = str(e)
            if "DeploymentNotFound" in error_msg:
                print(f"\n{'='*60}")
                print(f"ERROR: Model deployment '{self.model}' not found!")
                print(f"{'='*60}")
                print(f"\nTo fix this:")
                print(f"1. Go to Azure AI Foundry Portal → Your Project → Deployments")
                print(f"2. Find an existing deployment name (e.g., gpt-4o, gpt-4)")
                print(f"3. Either:")
                print(f"   - Update MODEL_DEPLOYMENT_NAME in your .env file")
                print(f"   - Or use: python main.py generate --model <deployment-name>")
                print(f"\n{'='*60}\n")
            raise
    
    def _setup_client(self):
        """Set up Azure AI client."""
        self.credential = DefaultAzureCredential()
        
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        
        if project_endpoint:
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=self.credential
            )
            self.openai_client = self.project_client.get_openai_client(api_version="2024-10-21")
            print(f"Connected to Azure AI Projects: {project_endpoint}")
        else:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ValueError("Either PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT required")
            
            self.project_client = None
            self.openai_client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=self._get_token,
                api_version="2024-10-21"
            )
    
    def _get_token(self) -> str:
        """Get Azure AD token."""
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token
    
    def generate_scenario(self) -> ClinicalScenario:
        """Generate a random clinical scenario."""
        category = random.choice(list(self.CLINICAL_CATEGORIES.keys()))
        condition = random.choice(self.CLINICAL_CATEGORIES[category])
        severity = random.choice(["mild", "moderate", "severe"])
        
        # Generate patient demographics
        age = random.randint(25, 90)
        gender = random.choice(["male", "female"])
        
        # Select 0-4 comorbidities with higher probability for older patients
        num_comorbidities = min(random.randint(0, 4) + (1 if age > 65 else 0), 5)
        comorbidities = random.sample(self.COMORBIDITIES, num_comorbidities)
        
        return ClinicalScenario(
            condition=condition,
            category=category,
            severity=severity,
            patient_demographics={"age": str(age), "gender": gender},
            comorbidities=comorbidities
        )
    
    def _determine_los(self, scenario: ClinicalScenario) -> int:
        """Determine length of stay based on severity."""
        base_los = {"mild": (2, 4), "moderate": (3, 7), "severe": (5, 14)}
        min_los, max_los = base_los.get(scenario.severity, (3, 7))
        
        # Adjust for comorbidities
        max_los += len(scenario.comorbidities)
        
        return random.randint(min_los, min(max_los, 21))
    
    def generate_clinical_notes(self, scenario: ClinicalScenario) -> str:
        """Generate clinical notes for a scenario using LLM."""
        los = self._determine_los(scenario)
        
        prompt = self.NOTES_GENERATION_PROMPT.format(
            condition=scenario.condition,
            category=scenario.category,
            severity=scenario.severity,
            age=scenario.patient_demographics.get("age", "65"),
            gender=scenario.patient_demographics.get("gender", "male"),
            comorbidities=", ".join(scenario.comorbidities) if scenario.comorbidities else "None",
            los=los
        )
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical documentation expert creating realistic hospital documentation for training purposes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.8  # Higher temperature for variety
        )
        
        return response.choices[0].message.content or ""
    
    def generate_summary(self, clinical_notes: str) -> str:
        """Generate Brief Hospital Course summary from clinical notes."""
        prompt = self.SUMMARY_GENERATION_PROMPT.format(clinical_notes=clinical_notes)
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical coding specialist creating Brief Hospital Course summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3  # Lower temperature for consistency
        )
        
        return response.choices[0].message.content or ""
    
    def generate_training_pair(self, scenario: Optional[ClinicalScenario] = None) -> Tuple[str, str]:
        """
        Generate a complete training pair (clinical notes + summary).
        
        Returns:
            Tuple of (clinical_notes, brief_hospital_course)
        """
        if scenario is None:
            scenario = self.generate_scenario()
        
        clinical_notes = self.generate_clinical_notes(scenario)
        summary = self.generate_summary(clinical_notes)
        
        return clinical_notes, summary
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        categories: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[ClinicalRecord]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Number of training samples to generate
            output_dir: Directory to save generated data
            categories: Limit to specific clinical categories
            show_progress: Show progress bar
            
        Returns:
            List of ClinicalRecord objects
        """
        output_dir = output_dir or self.config.data.processed_data_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  Synthetic Clinical Data Generation")
        print(f"  Generating {num_samples} training samples using Azure AI Foundry")
        print(f"{'='*60}\n")
        
        records = []
        failed = 0
        
        # Filter categories if specified
        available_categories = categories or list(self.CLINICAL_CATEGORIES.keys())
        
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating samples")
        
        for i in iterator:
            try:
                # Generate scenario with category distribution
                scenario = ClinicalScenario(
                    condition=random.choice(
                        self.CLINICAL_CATEGORIES[random.choice(available_categories)]
                    ),
                    category=random.choice(available_categories),
                    severity=random.choice(["mild", "moderate", "severe"]),
                    patient_demographics={
                        "age": str(random.randint(25, 90)),
                        "gender": random.choice(["male", "female"])
                    },
                    comorbidities=random.sample(
                        self.COMORBIDITIES, 
                        random.randint(0, 4)
                    )
                )
                
                clinical_notes, summary = self.generate_training_pair(scenario)
                
                records.append(ClinicalRecord(
                    hadm_id=f"SYNTH_{datetime.now().strftime('%Y%m%d')}_{i+1:05d}",
                    clinical_notes=clinical_notes,
                    brief_hospital_course=summary,
                    note_types_included=["Nursing", "Physician", "Radiology"]
                ))
                
            except Exception as e:
                failed += 1
                if show_progress:
                    tqdm.write(f"Error generating sample {i+1}: {e}")
                continue
        
        print(f"\nGenerated {len(records)} samples ({failed} failed)")
        
        # Save using DataPreparator format
        if records:
            self._save_dataset(records, output_dir)
        
        return records
    
    def _save_dataset(self, records: List[ClinicalRecord], output_dir: str):
        """Save generated dataset in fine-tuning format."""
        preparator = DataPreparator()
        preparator.config.processed_data_dir = output_dir
        
        # Split and save
        preparator.split_and_save(records, shuffle=True)
        
        # Save generation metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "num_samples": len(records),
            "model_used": self.model,
            "categories_distribution": self._compute_category_distribution(records)
        }
        
        metadata_path = Path(output_dir) / "synthetic_data_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Metadata saved to {metadata_path}")
    
    def _compute_category_distribution(self, records: List[ClinicalRecord]) -> Dict[str, int]:
        """Compute distribution of clinical categories in dataset."""
        # This is a simplified version - in practice, you'd parse the notes
        return {"total_records": len(records)}
    
    def generate_augmented_sample(
        self,
        original_notes: str,
        augmentation_type: str = "paraphrase"
    ) -> str:
        """
        Generate an augmented version of existing clinical notes.
        
        Useful for data augmentation when you have some real data.
        
        Args:
            original_notes: Original clinical notes
            augmentation_type: Type of augmentation 
                - "paraphrase": Rewrite with different wording
                - "add_detail": Add more clinical detail
                - "simplify": Create a simplified version
        """
        augmentation_prompts = {
            "paraphrase": """Rewrite the following clinical notes using different medical terminology 
and phrasing while preserving all clinical information. Maintain the same structure 
(Nursing Notes, Physician Notes, etc.).

Original Notes:
{notes}

Rewritten Notes:""",
            
            "add_detail": """Expand the following clinical notes with additional realistic clinical details.
Add more specific vital signs, lab values, medication dosages, and nursing interventions
while maintaining consistency with the original narrative.

Original Notes:
{notes}

Expanded Notes:""",
            
            "simplify": """Condense the following clinical notes while preserving all essential 
clinical information needed for coding. Remove redundant information but keep all 
diagnoses, procedures, and key events.

Original Notes:
{notes}

Condensed Notes:"""
        }
        
        prompt = augmentation_prompts.get(augmentation_type, augmentation_prompts["paraphrase"])
        prompt = prompt.format(notes=original_notes)
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical documentation specialist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.7
        )
        
        return response.choices[0].message.content or ""
    
    def interactive_generation(self):
        """Run interactive data generation demo."""
        print("\n" + "="*60)
        print("  Interactive Synthetic Data Generation Demo")
        print("="*60 + "\n")
        
        print("Generating a random clinical scenario...")
        scenario = self.generate_scenario()
        
        print(f"\nScenario Generated:")
        print(f"  Condition: {scenario.condition}")
        print(f"  Category: {scenario.category}")
        print(f"  Severity: {scenario.severity}")
        print(f"  Patient: {scenario.patient_demographics['age']}yo {scenario.patient_demographics['gender']}")
        print(f"  Comorbidities: {', '.join(scenario.comorbidities) or 'None'}")
        
        print("\nGenerating clinical notes...")
        clinical_notes, summary = self.generate_training_pair(scenario)
        
        print("\n" + "-"*40)
        print("GENERATED CLINICAL NOTES (excerpt):")
        print("-"*40)
        print(clinical_notes[:1500] + "..." if len(clinical_notes) > 1500 else clinical_notes)
        
        print("\n" + "-"*40)
        print("GENERATED BRIEF HOSPITAL COURSE:")
        print("-"*40)
        print(summary)
        
        return clinical_notes, summary


def main():
    """Main function to run synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical training data using Azure AI Foundry"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of training samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model deployment name to use for generation"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Limit to specific clinical categories"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo (generate one sample)"
    )
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(model_name=args.model)
    
    if args.demo:
        generator.interactive_generation()
    else:
        records = generator.generate_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            categories=args.categories
        )
        print(f"\nGeneration complete! Created {len(records)} training samples.")


if __name__ == "__main__":
    main()
