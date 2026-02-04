"""
Fine-tuning Pipeline for Hospital Course Summarization.

This module implements the fine-tuning workflow using Azure AI Foundry
Python SDK, following the methodology from "Harmonising the Clinical Melody"
(arXiv:2409.14638).

The pipeline:
1. Uploads training and validation data
2. Creates a fine-tuning job
3. Monitors training progress
4. Deploys the fine-tuned model
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

load_dotenv()


class FineTuningPipeline:
    """
    Azure AI Foundry fine-tuning pipeline for clinical summarization.
    
    This class encapsulates the complete fine-tuning workflow:
    - Data upload to Azure OpenAI
    - Fine-tuning job creation and monitoring
    - Model deployment
    - Result analysis
    """
    
    def __init__(self):
        """Initialize the fine-tuning pipeline."""
        self.config = get_config()
        self._setup_clients()
        self._ensure_output_dir()
    
    def _setup_clients(self):
        """Set up Azure AI clients."""
        # Get credentials
        self.credential = DefaultAzureCredential()
        
        # Set up Azure OpenAI client for fine-tuning operations
        # Fine-tuning uses the Azure OpenAI API directly
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = "2024-10-21"
        
        if not self.endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable is required. "
                "Set it to your Azure OpenAI resource endpoint (e.g., https://your-resource.openai.azure.com/)"
            )
        
        # Create Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=self._get_token,
            api_version=self.api_version
        )
        
        print(f"Connected to Azure OpenAI at: {self.endpoint}")
    
    def _get_token(self) -> str:
        """Get Azure AD token for authentication."""
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        Path(self.config.data.output_dir).mkdir(parents=True, exist_ok=True)
    
    def upload_training_files(self) -> Dict[str, str]:
        """
        Upload training and validation files to Azure OpenAI.
        
        Returns:
            Dictionary with file IDs for training and validation files
        """
        print("\n=== Uploading Training Files ===\n")
        
        file_ids = {}
        files_to_upload = {
            "training": Path(self.config.data.processed_data_dir) / self.config.data.train_file,
            "validation": Path(self.config.data.processed_data_dir) / self.config.data.validation_file
        }
        
        for file_type, file_path in files_to_upload.items():
            if not file_path.exists():
                print(f"Warning: {file_type} file not found at {file_path}")
                continue
            
            print(f"Uploading {file_type} file: {file_path}")
            
            with open(file_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            file_ids[file_type] = response.id
            print(f"  Uploaded successfully. File ID: {response.id}")
            
            # Wait for file to be processed
            self._wait_for_file_processing(response.id, file_type)
        
        # Save file IDs for reference
        self._save_file_ids(file_ids)
        
        return file_ids
    
    def _wait_for_file_processing(self, file_id: str, file_type: str, timeout: int = 300):
        """
        Wait for uploaded file to be processed by Azure.
        
        Args:
            file_id: The file ID to check
            file_type: Type of file (training/validation) for display
            timeout: Maximum seconds to wait (default 300)
        """
        import time
        
        print(f"  Waiting for {file_type} file to be processed...", end="", flush=True)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                file_info = self.client.files.retrieve(file_id)
                status = file_info.status
                
                if status == "processed" or status == "succeeded":
                    print(f" ✓ Ready (status: {status})")
                    return
                elif status == "failed" or status == "error":
                    print(f" ✗ Failed")
                    raise RuntimeError(f"File processing failed with status: {status}")
                
                # Still processing
                print(".", end="", flush=True)
                time.sleep(5)
                
            except Exception as e:
                if "does not exist" in str(e).lower():
                    # File might not be visible yet, continue waiting
                    time.sleep(5)
                else:
                    raise
        
        raise TimeoutError(f"File processing timed out after {timeout} seconds")
    
    def _save_file_ids(self, file_ids: Dict[str, str]):
        """Save file IDs to a JSON file for reference."""
        output_path = Path(self.config.data.output_dir) / "file_ids.json"
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "file_ids": file_ids
            }, f, indent=2)
        print(f"File IDs saved to {output_path}")
    
    def create_fine_tuning_job(
        self, 
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            training_file_id: ID of the uploaded training file
            validation_file_id: ID of the uploaded validation file (optional)
            suffix: Custom suffix for the fine-tuned model name
            
        Returns:
            Fine-tuning job ID
        """
        print("\n=== Creating Fine-tuning Job ===\n")
        
        ft_config = self.config.fine_tuning
        suffix = suffix or f"clinical-summary-{datetime.now().strftime('%Y%m%d')}"
        
        # Build hyperparameters
        hyperparameters = {
            "n_epochs": ft_config.n_epochs,
            "batch_size": ft_config.batch_size,
            "learning_rate_multiplier": ft_config.learning_rate_multiplier
        }
        
        print(f"Base model: {self.config.azure.base_model}")
        print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
        print(f"Training type: {self.config.azure.training_type}")
        
        # Create fine-tuning job
        job_params = {
            "training_file": training_file_id,
            "model": self.config.azure.base_model,
            "hyperparameters": hyperparameters,
            "suffix": suffix,
            "seed": ft_config.seed
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        try:
            response = self.client.fine_tuning.jobs.create(**job_params)
        except Exception as e:
            error_msg = str(e)
            if "does not support fine-tuning" in error_msg or "invalidPayload" in error_msg or "deprecated" in error_msg.lower():
                print(f"\n{'='*70}")
                print("ERROR: The specified base model does not support fine-tuning!")
                print(f"{'='*70}")
                print(f"\nAttempted model: {self.config.azure.base_model}")
                print(f"\nSupported fine-tuning models and regions (2026):")
                print("-" * 70)
                print("| Model                    | Supported Regions                        |")
                print("-" * 70)
                print("| gpt-4.1-mini (2025-04-14)| North Central US, Sweden Central         |")
                print("| gpt-4.1-nano (2025-04-14)| North Central US, Sweden Central         |")
                print("| gpt-4.1 (2025-04-14)     | North Central US, Sweden Central         |")
                print("| gpt-4o (2024-08-06)      | East US2, North Central US, Sweden Central|")
                print("| gpt-4o-mini (2024-07-18) | North Central US, Sweden Central         |")
                print("| o4-mini (2025-04-16)     | East US2, Sweden Central                 |")
                print("-" * 70)
                print("NOTE: gpt-35-turbo models are deprecated for fine-tuning.")
                print(f"\nTo fix:")
                print(f"1. Check your Azure OpenAI resource region in Azure Portal")
                print(f"2. Update BASE_MODEL in .env to a model supported in your region")
                print(f"   Example: BASE_MODEL=gpt-4.1-mini")
                print(f"\n{'='*70}\n")
            raise
        
        job_id = response.id
        print(f"\nFine-tuning job created successfully!")
        print(f"Job ID: {job_id}")
        print(f"Status: {response.status}")
        
        # Save job info
        self._save_job_info(response)
        
        return job_id
    
    def _save_job_info(self, job_response: Any):
        """Save job information to a JSON file."""
        output_path = Path(self.config.data.output_dir) / f"job_{job_response.id}.json"
        
        job_info = {
            "job_id": job_response.id,
            "status": job_response.status,
            "model": job_response.model,
            "created_at": datetime.now().isoformat(),
            "hyperparameters": job_response.hyperparameters.model_dump() if hasattr(job_response.hyperparameters, 'model_dump') else str(job_response.hyperparameters)
        }
        
        with open(output_path, "w") as f:
            json.dump(job_info, f, indent=2)
        print(f"Job info saved to {output_path}")
    
    def monitor_job(
        self, 
        job_id: str, 
        poll_interval: int = 60,
        max_wait_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Monitor the fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
            max_wait_hours: Maximum hours to wait for completion
            
        Returns:
            Final job status information
        """
        print(f"\n=== Monitoring Fine-tuning Job: {job_id} ===\n")
        
        max_iterations = (max_wait_hours * 3600) // poll_interval
        
        for i in range(max_iterations):
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status = job.status
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
            
            # Check for training metrics
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"  Trained tokens: {job.trained_tokens}")
            
            # Terminal states
            if status == "succeeded":
                print("\n✓ Fine-tuning completed successfully!")
                self._save_final_results(job)
                return {
                    "status": "succeeded",
                    "fine_tuned_model": job.fine_tuned_model,
                    "job_id": job_id
                }
            
            elif status == "failed":
                error = getattr(job, 'error', 'Unknown error')
                print(f"\n✗ Fine-tuning failed: {error}")
                return {
                    "status": "failed",
                    "error": str(error),
                    "job_id": job_id
                }
            
            elif status == "cancelled":
                print("\n✗ Fine-tuning was cancelled")
                return {
                    "status": "cancelled",
                    "job_id": job_id
                }
            
            time.sleep(poll_interval)
        
        print(f"\nWarning: Maximum wait time ({max_wait_hours} hours) exceeded")
        return {"status": "timeout", "job_id": job_id}
    
    def _save_final_results(self, job: Any):
        """Save final job results."""
        output_path = Path(self.config.data.output_dir) / f"results_{job.id}.json"
        
        results = {
            "job_id": job.id,
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "completed_at": datetime.now().isoformat(),
            "trained_tokens": getattr(job, 'trained_tokens', None)
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def download_training_results(self, job_id: str) -> Optional[str]:
        """
        Download the training results/metrics file.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            Path to downloaded results file, or None if not available
        """
        print(f"\n=== Downloading Training Results ===\n")
        
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        
        if not hasattr(job, 'result_files') or not job.result_files:
            print("No result files available yet")
            return None
        
        for result_file_id in job.result_files:
            print(f"Downloading result file: {result_file_id}")
            
            content = self.client.files.content(result_file_id)
            
            output_path = Path(self.config.data.output_dir) / f"training_results_{job_id}.csv"
            with open(output_path, "wb") as f:
                f.write(content.read())
            
            print(f"Results saved to {output_path}")
            return str(output_path)
        
        return None
    
    def list_jobs(self, limit: int = 10) -> list:
        """List recent fine-tuning jobs."""
        print("\n=== Recent Fine-tuning Jobs ===\n")
        
        jobs = list(self.client.fine_tuning.jobs.list(limit=limit))
        
        for job in jobs:
            print(f"Job ID: {job.id}")
            print(f"  Model: {job.model}")
            print(f"  Status: {job.status}")
            print(f"  Created: {job.created_at}")
            if job.fine_tuned_model:
                print(f"  Fine-tuned model: {job.fine_tuned_model}")
            print()
        
        return jobs
    
    def cancel_job(self, job_id: str):
        """Cancel a running fine-tuning job."""
        print(f"Cancelling job: {job_id}")
        self.client.fine_tuning.jobs.cancel(job_id)
        print("Job cancelled")
    
    def run_pipeline(
        self, 
        upload_new: bool = True,
        monitor: bool = True,
        model_suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            upload_new: Whether to upload new training files
            monitor: Whether to wait for job completion
            model_suffix: Custom suffix for the model name
            
        Returns:
            Pipeline execution results
        """
        print("\n" + "="*60)
        print("  Hospital Course Summarization Fine-tuning Pipeline")
        print("  Based on: 'Harmonising the Clinical Melody' (arXiv:2409.14638)")
        print("="*60 + "\n")
        
        results = {"timestamp": datetime.now().isoformat()}
        
        try:
            # Step 1: Upload training files
            if upload_new:
                file_ids = self.upload_training_files()
                if "training" not in file_ids:
                    raise ValueError("Training file upload failed")
                results["file_ids"] = file_ids
            else:
                # Load existing file IDs
                file_ids_path = Path(self.config.data.output_dir) / "file_ids.json"
                if not file_ids_path.exists():
                    raise ValueError("No existing file IDs found. Set upload_new=True")
                with open(file_ids_path) as f:
                    file_ids = json.load(f)["file_ids"]
            
            # Step 2: Create fine-tuning job
            job_id = self.create_fine_tuning_job(
                training_file_id=file_ids["training"],
                validation_file_id=file_ids.get("validation"),
                suffix=model_suffix
            )
            results["job_id"] = job_id
            
            # Step 3: Monitor job (optional)
            if monitor:
                job_results = self.monitor_job(job_id)
                results.update(job_results)
                
                # Download training metrics if successful
                if job_results["status"] == "succeeded":
                    metrics_path = self.download_training_results(job_id)
                    if metrics_path:
                        results["metrics_file"] = metrics_path
            
            print("\n" + "="*60)
            print("  Pipeline Execution Complete")
            print("="*60)
            
        except Exception as e:
            results["error"] = str(e)
            print(f"\nPipeline error: {e}")
            raise
        
        # Save final results
        final_path = Path(self.config.data.output_dir) / "pipeline_results.json"
        with open(final_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Main function to run the fine-tuning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for hospital course summarization"
    )
    parser.add_argument(
        "--upload", 
        action="store_true", 
        default=True,
        help="Upload new training files"
    )
    parser.add_argument(
        "--no-monitor", 
        action="store_true",
        help="Don't wait for job completion"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom suffix for the fine-tuned model"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent fine-tuning jobs"
    )
    parser.add_argument(
        "--job-status",
        type=str,
        default=None,
        help="Check status of a specific job"
    )
    
    args = parser.parse_args()
    
    pipeline = FineTuningPipeline()
    
    if args.list_jobs:
        pipeline.list_jobs()
        return
    
    if args.job_status:
        result = pipeline.monitor_job(args.job_status, poll_interval=1, max_wait_hours=0)
        print(f"Job status: {result}")
        return
    
    # Run the full pipeline
    results = pipeline.run_pipeline(
        upload_new=args.upload,
        monitor=not args.no_monitor,
        model_suffix=args.suffix
    )
    
    print(f"\nFinal results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
