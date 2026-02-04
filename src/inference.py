"""
Inference Module for Hospital Course Summarization.

This module provides utilities to run inference using either:
1. Azure OpenAI fine-tuned model (deployed)
2. Azure AI Foundry Projects client

Follows the Azure AI Foundry SDK patterns from the documentation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Generator
from datetime import datetime
from tqdm import tqdm
import jsonlines

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
from src.data_preparation import DataPreparator

load_dotenv()


class ClinicalSummarizer:
    """
    Clinical summarization inference engine.
    
    Supports:
    - Single document summarization
    - Batch processing
    - Comparison between base and fine-tuned models
    """
    
    SYSTEM_PROMPT = DataPreparator.SYSTEM_PROMPT
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        use_fine_tuned: bool = True
    ):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Specific model deployment name to use
            use_fine_tuned: Whether to use fine-tuned model (if available)
        """
        self.config = get_config()
        self._setup_client()
        
        # Determine model to use
        if model_name:
            self.model = model_name
        elif use_fine_tuned:
            self.model = self._find_fine_tuned_model()
        else:
            self.model = self.config.azure.model_deployment_name
        
        print(f"Using model: {self.model}")
    
    def _setup_client(self):
        """Set up Azure AI clients."""
        self.credential = DefaultAzureCredential()
        
        # Option 1: Using Azure AI Projects (recommended)
        project_endpoint = os.getenv("PROJECT_ENDPOINT")
        
        if project_endpoint:
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=self.credential
            )
            self.openai_client = self.project_client.get_openai_client(api_version="2024-10-21")
            print(f"Using Azure AI Projects endpoint: {project_endpoint}")
        else:
            # Option 2: Direct Azure OpenAI
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "Either PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT is required"
                )
            
            self.project_client = None
            self.openai_client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=self._get_token,
                api_version="2024-10-21"
            )
            print(f"Using Azure OpenAI endpoint: {endpoint}")
    
    def _get_token(self) -> str:
        """Get Azure AD token for authentication."""
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token
    
    def _find_fine_tuned_model(self) -> str:
        """
        Find the latest fine-tuned model from results.
        
        Returns:
            Model deployment name
        """
        import os
        
        # First, check if fine-tuned model is specified in environment
        fine_tuned_model = os.getenv("FINE_TUNED_MODEL_NAME")
        if fine_tuned_model:
            print(f"Using fine-tuned model from environment: {fine_tuned_model}")
            return fine_tuned_model
        
        # Look for pipeline results with fine-tuned model info
        results_dir = Path(self.config.data.output_dir)
        for results_file in sorted(results_dir.glob("results_*.json"), reverse=True):
            with open(results_file) as f:
                data = json.load(f)
                if "fine_tuned_model" in data:
                    return data["fine_tuned_model"]
        
        # Fall back to default model
        print("No fine-tuned model found, using base model")
        return self.config.azure.model_deployment_name
    
    def summarize(
        self, 
        clinical_notes: str,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a Brief Hospital Course summary from clinical notes.
        
        Args:
            clinical_notes: Input clinical documentation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more focused)
            
        Returns:
            Generated summary
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a Brief Hospital Course summary for the following clinical notes:\n\n{clinical_notes}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def summarize_batch(
        self, 
        clinical_notes_list: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        save_path: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate summaries for a batch of clinical notes.
        
        Args:
            clinical_notes_list: List of clinical notes to summarize
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            save_path: Optional path to save results as JSONL
            
        Returns:
            List of dictionaries with input and prediction
        """
        results = []
        
        for notes in tqdm(clinical_notes_list, desc="Generating summaries"):
            try:
                summary = self.summarize(notes, max_tokens, temperature)
                results.append({
                    "input": notes[:500] + "..." if len(notes) > 500 else notes,
                    "prediction": summary
                })
            except Exception as e:
                print(f"Error processing notes: {e}")
                results.append({
                    "input": notes[:500] + "...",
                    "prediction": f"ERROR: {str(e)}"
                })
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(save_path, mode='w') as writer:
                for result in results:
                    writer.write(result)
            print(f"Results saved to {save_path}")
        
        return results
    
    def process_test_set(
        self, 
        test_file: str,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Process the test set and generate predictions.
        
        Args:
            test_file: Path to test JSONL file
            output_file: Path to save predictions
            
        Returns:
            List of results with inputs, predictions, and references
        """
        print(f"\nProcessing test set: {test_file}")
        
        results = []
        
        with jsonlines.open(test_file) as reader:
            records = list(reader)
        
        for record in tqdm(records, desc="Processing test samples"):
            messages = record["messages"]
            
            # Extract user input
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"),
                None
            )
            
            # Extract reference
            reference = next(
                (m["content"] for m in messages if m["role"] == "assistant"),
                None
            )
            
            if user_msg:
                # Extract just the clinical notes (remove instruction prefix)
                clinical_notes = user_msg.replace(
                    "Generate a Brief Hospital Course summary for the following clinical notes:\n\n",
                    ""
                )
                
                try:
                    prediction = self.summarize(clinical_notes)
                except Exception as e:
                    prediction = f"ERROR: {str(e)}"
                
                results.append({
                    "input": clinical_notes[:300] + "...",
                    "prediction": prediction,
                    "reference": reference
                })
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(output_file, mode='w') as writer:
                for result in results:
                    writer.write(result)
            print(f"Predictions saved to {output_file}")
        
        return results
    
    def compare_models(
        self, 
        clinical_notes: str,
        models: List[str]
    ) -> Dict[str, str]:
        """
        Compare summaries from different models.
        
        Args:
            clinical_notes: Input clinical notes
            models: List of model deployment names to compare
            
        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}
        
        for model in models:
            print(f"Generating with {model}...")
            original_model = self.model
            self.model = model
            
            try:
                results[model] = self.summarize(clinical_notes)
            except Exception as e:
                results[model] = f"ERROR: {str(e)}"
            
            self.model = original_model
        
        return results
    
    def interactive_demo(self):
        """Run an interactive demonstration."""
        print("\n" + "="*60)
        print("  Clinical Summarization Interactive Demo")
        print("  Model:", self.model)
        print("="*60 + "\n")
        
        sample_notes = """
=== NURSING NOTES ===
Day 1: Patient admitted with chest pain and shortness of breath.
Vital signs: BP 145/90, HR 102, RR 22, SpO2 94% on 2L NC.
IV access established, cardiac monitoring initiated.
Troponin I: 0.8 ng/mL (elevated).

Day 2: Patient underwent cardiac catheterization.
Found 80% stenosis in LAD, stent placed successfully.
Post-procedure: stable, no complications.
Started on aspirin, clopidogrel, and statin.

Day 3: Cardiac rehab evaluation completed.
Patient ambulating with assist.
Echo shows EF 50%, mild LV dysfunction.
Pain controlled, no further chest pain episodes.

=== PHYSICIAN NOTES ===
Assessment: NSTEMI with successful PCI to LAD.
Plan: Continue dual antiplatelet therapy, optimize medical management.
Cardiology follow-up in 2 weeks.
"""
        
        print("Sample clinical notes:")
        print("-" * 40)
        print(sample_notes[:500] + "...")
        print("-" * 40)
        
        print("\nGenerating summary...")
        summary = self.summarize(sample_notes)
        
        print("\nGenerated Brief Hospital Course:")
        print("=" * 40)
        print(summary)
        print("=" * 40)
        
        return summary


def main():
    """Main function to run inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference with the clinical summarization model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model deployment name to use"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test JSONL file to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.jsonl",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo"
    )
    parser.add_argument(
        "--use-base",
        action="store_true",
        help="Use base model instead of fine-tuned"
    )
    
    args = parser.parse_args()
    
    summarizer = ClinicalSummarizer(
        model_name=args.model,
        use_fine_tuned=not args.use_base
    )
    
    if args.demo:
        summarizer.interactive_demo()
    elif args.test_file:
        summarizer.process_test_set(args.test_file, args.output)
    else:
        summarizer.interactive_demo()


if __name__ == "__main__":
    main()
