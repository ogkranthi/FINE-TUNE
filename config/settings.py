"""
Configuration settings for the Healthcare Fine-tuning Project.
Based on: "Harmonising the Clinical Melody: Tuning Large Language Models 
for Hospital Course Summarisation in Clinical Coding" (arXiv:2409.14638)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AzureConfig:
    """Azure AI Foundry configuration."""
    project_endpoint: str = field(default_factory=lambda: os.getenv("PROJECT_ENDPOINT", ""))
    model_deployment_name: str = field(default_factory=lambda: os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini"))
    
    # Fine-tuning specific - reads from BASE_MODEL env var
    base_model: str = field(default_factory=lambda: os.getenv("BASE_MODEL", "gpt-4.1-mini"))
    training_type: str = "global"  # Options: "standard", "global", "developer"
    
    # Supported fine-tuning models and their regions (as of 2026)
    # gpt-4.1-mini (2025-04-14): North Central US, Sweden Central
    # gpt-4.1-nano (2025-04-14): North Central US, Sweden Central
    # gpt-4.1 (2025-04-14): North Central US, Sweden Central
    # gpt-4o (2024-08-06): East US2, North Central US, Sweden Central
    # gpt-4o-mini (2024-07-18): North Central US, Sweden Central
    # o4-mini (2025-04-16): East US2, Sweden Central
    # NOTE: gpt-35-turbo models are deprecated for fine-tuning


@dataclass
class DataConfig:
    """Data configuration for clinical summarization task."""
    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs"
    
    # File names
    train_file: str = "train.jsonl"
    validation_file: str = "validation.jsonl"
    test_file: str = "test.jsonl"
    
    # Data splits (from the paper methodology)
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Clinical notes to include (based on MIMIC-III data structure)
    clinical_note_types: List[str] = field(default_factory=lambda: [
        "Nursing",
        "Physician",
        "Radiology",
        "ECG",
        "Respiratory",
        "Discharge Summary"
    ])
    
    # Maximum tokens for input/output
    max_input_tokens: int = 8192  # For clinical text
    max_output_tokens: int = 1024  # For brief hospital course


@dataclass
class FineTuningConfig:
    """Fine-tuning hyperparameters based on the paper's QLoRA approach."""
    # Azure OpenAI fine-tuning hyperparameters
    n_epochs: int = 3
    batch_size: int = 4
    learning_rate_multiplier: float = 0.1
    
    # Seed for reproducibility
    seed: int = 42
    
    # Validation settings
    validation_split: Optional[float] = None  # Use separate validation file


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration (from the paper)."""
    # BERTScore settings - using lighter model to avoid disk space issues
    # Options: distilbert-base-uncased (smaller), microsoft/deberta-base-mnli (medium)
    bert_model: str = "distilbert-base-uncased"
    
    # ROUGE variants to compute
    rouge_types: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    
    # Clinical coding specific metrics
    use_clinical_metrics: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration."""
    azure: AzureConfig = field(default_factory=AzureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> ProjectConfig:
    """Load and return project configuration."""
    return ProjectConfig()
