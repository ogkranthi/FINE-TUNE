# Hospital Course Summarization with Azure AI Foundry

A fine-tuning project for clinical documentation using Azure AI Foundry and Python SDK, based on the research paper ["Harmonising the Clinical Melody: Tuning Large Language Models for Hospital Course Summarisation in Clinical Coding"](https://arxiv.org/abs/2409.14638).

## Overview

This project implements a complete pipeline for fine-tuning Large Language Models to generate "Brief Hospital Course" summaries from clinical notes. These summaries are essential for clinical coding and healthcare documentation.

### Research Background

The paper (arXiv:2409.14638) addresses the challenge of summarizing hospital courses from Electronic Medical Records (EMR) for clinical coding purposes. Key findings:

- **Models**: Adapted Llama 3, BioMistral, and Mistral Instruct v0.1 using QLoRA (Quantized Low-Rank Adaptation)
- **Data**: Used MIMIC-III clinical notes concatenated as input, paired with Brief Hospital Course sections
- **Evaluation**: BERTScore and ROUGE metrics, plus custom clinical coding utility metrics
- **Result**: Fine-tuned models significantly outperform base models for clinical summarization

This implementation adapts the methodology for **Azure AI Foundry** using supported models (GPT-4o-mini, GPT-4o, GPT-3.5-Turbo).

## Project Structure

```
fine-tune/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration dataclasses
├── src/
│   ├── __init__.py
│   ├── data_preparation.py  # Data processing for MIMIC-III format
│   ├── synthetic_data.py    # Synthetic data generation with Azure AI
│   ├── fine_tuning.py       # Azure OpenAI fine-tuning pipeline
│   ├── evaluation.py        # BERTScore, ROUGE, clinical metrics
│   └── inference.py         # Model inference utilities
├── data/
│   ├── raw/                  # Place MIMIC-III data here
│   ├── processed/            # Generated JSONL files
│   └── samples/              # Sample training data
├── outputs/                  # Training results and evaluations
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

### Azure Resources

1. **Azure Subscription** with access to Azure AI Foundry
2. **Azure AI Foundry Resource** (formerly Azure AI Studio)
3. **Azure OpenAI Resource** with fine-tuning capability
4. **Role Requirements**: 
   - Azure AI Owner role for deploying fine-tuned models
   - Azure AI User role for training (but not deployment)

### Supported Models for Fine-tuning

| Model | Recommended For |
|-------|-----------------|
| `gpt-4o-mini-2024-07-18` | Cost-effective, good quality |
| `gpt-4o-2024-08-06` | Highest quality |
| `gpt-35-turbo-0125` | Budget option |

## Quick Start

### 1. Clone and Setup

```bash
cd fine-tune

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

Edit `.env` with your Azure credentials:

```env
PROJECT_ENDPOINT=https://your-foundry-resource.services.ai.azure.com/api/projects/your-project
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
MODEL_DEPLOYMENT_NAME=gpt-4o-mini
```

### 3. Authenticate with Azure

```bash
az login
```

### 4. Prepare Data

**Option A: Use sample data (for testing)**
```bash
python src/data_preparation.py
# Creates sample training data in data/processed/
```

**Option B: Generate synthetic data with Azure AI Foundry (recommended)**
```bash
# Generate 100 synthetic clinical training samples
python main.py generate --num-samples 100

# Interactive demo - generate one sample
python main.py generate --demo

# Generate for specific clinical categories
python main.py generate --num-samples 50 --categories Cardiology Pulmonology
```

**Option C: Use MIMIC-III data (for production)**
1. Obtain MIMIC-III access from [PhysioNet](https://physionet.org/content/mimiciii/)
2. Place `NOTEEVENTS.csv` in `data/raw/`
3. Run data preparation:
```bash
python src/data_preparation.py
```

### 5. Run Fine-tuning

```bash
python src/fine_tuning.py
```

This will:
1. Upload training/validation files to Azure OpenAI
2. Create a fine-tuning job
3. Monitor progress until completion
4. Save results to `outputs/`

### 6. Run Inference

```bash
# Interactive demo
python src/inference.py --demo

# Process test set
python src/inference.py --test-file data/processed/test.jsonl --output outputs/predictions.jsonl
```

### 7. Evaluate Results

```bash
python src/evaluation.py \
    --predictions outputs/predictions.jsonl \
    --references data/processed/test.jsonl \
    --output outputs/evaluation_report.txt
```

## Detailed Usage

### Data Preparation

The `DataPreparator` class handles conversion of clinical notes to Azure OpenAI fine-tuning format (JSONL with chat completion structure):

```python
from src.data_preparation import DataPreparator

preparator = DataPreparator()

# Create sample data for testing
preparator.create_sample_data(num_samples=100)

# Or process real MIMIC data
notes_df = preparator.load_mimic_notes("data/raw/NOTEEVENTS.csv")
summaries_df = preparator.load_discharge_summaries("data/raw/DISCHARGE_SUMMARIES.csv")
records = preparator.create_training_records(notes_df, summaries_df)
preparator.split_and_save(records)
```

**Data Format (JSONL)**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a clinical documentation specialist..."},
    {"role": "user", "content": "Generate a Brief Hospital Course summary for..."},
    {"role": "assistant", "content": "The patient is a 68-year-old male..."}
  ]
}
```

### Synthetic Data Generation with Azure AI Foundry

When real clinical data (MIMIC-III) is not available, you can generate synthetic training data using Azure AI Foundry models:

```python
from src.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator()

# Generate a complete dataset
records = generator.generate_dataset(
    num_samples=100,
    output_dir="data/processed",
    categories=["Cardiology", "Pulmonology", "Infectious Disease"]
)

# Generate a single training pair
clinical_notes, summary = generator.generate_training_pair()

# Augment existing data
augmented_notes = generator.generate_augmented_sample(
    original_notes,
    augmentation_type="paraphrase"  # or "add_detail", "simplify"
)
```

**Supported Clinical Categories**:

| Category | Example Conditions |
|----------|-------------------|
| Cardiology | STEMI, NSTEMI, Heart Failure, AFib with RVR |
| Pulmonology | CAP, COPD Exacerbation, PE, Respiratory Failure |
| Gastroenterology | Acute Pancreatitis, GI Bleeding, Cholecystitis |
| Nephrology | AKI, CKD Exacerbation, Pyelonephritis |
| Endocrinology | DKA, HHS, Thyroid Storm |
| Infectious Disease | Sepsis, Septic Shock, Meningitis |
| Neurology | Ischemic Stroke, Hemorrhagic Stroke, Status Epilepticus |
| Surgery/Orthopedics | Post-op Complications, Hip Fracture, TKA |
| Hematology/Oncology | Febrile Neutropenia, Tumor Lysis Syndrome |

**Generation Process**:
1. Random clinical scenario selection (condition, severity, demographics, comorbidities)
2. LLM generates detailed clinical notes (Nursing, Physician, Radiology)
3. LLM generates corresponding Brief Hospital Course summary
4. Output saved in JSONL format ready for fine-tuning

### Fine-tuning Pipeline

```python
from src.fine_tuning import FineTuningPipeline

pipeline = FineTuningPipeline()

# Full pipeline
results = pipeline.run_pipeline(
    upload_new=True,
    monitor=True,
    model_suffix="clinical-summary"
)

# Or step by step
file_ids = pipeline.upload_training_files()
job_id = pipeline.create_fine_tuning_job(
    training_file_id=file_ids["training"],
    validation_file_id=file_ids.get("validation")
)
pipeline.monitor_job(job_id)
```

**Hyperparameters** (configurable in `config/settings.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Batch size |
| `learning_rate_multiplier` | 0.1 | Learning rate scale |

### Evaluation Metrics

The evaluation follows the paper's methodology:

```python
from src.evaluation import ClinicalEvaluator

evaluator = ClinicalEvaluator()

# Full evaluation
result = evaluator.evaluate(predictions, references)
print(evaluator.generate_report(result))
```

**Metrics computed**:

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | N-gram overlap scores |
| BERTScore | Semantic similarity using transformers |
| Clinical Term Overlap | Domain-specific terminology coverage |
| Diagnosis Coverage | Coverage of diagnosis-related terms |
| Procedure Coverage | Coverage of procedure-related terms |

### Inference

```python
from src.inference import ClinicalSummarizer

summarizer = ClinicalSummarizer(use_fine_tuned=True)

# Single summary
clinical_notes = "Patient admitted with chest pain..."
summary = summarizer.summarize(clinical_notes)

# Batch processing
summaries = summarizer.summarize_batch(notes_list)

# Compare models
comparison = summarizer.compare_models(
    clinical_notes,
    models=["gpt-4o-mini", "ft:gpt-4o-mini:your-model"]
)
```

## Configuration

All settings are in `config/settings.py`:

```python
@dataclass
class FineTuningConfig:
    n_epochs: int = 3
    batch_size: int = 4
    learning_rate_multiplier: float = 0.1
    seed: int = 42

@dataclass
class EvaluationConfig:
    bert_model: str = "microsoft/deberta-xlarge-mnli"
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
```

## Command-Line Interface

### Synthetic Data Generation Commands

```bash
# Generate 100 synthetic samples
python main.py generate --num-samples 100

# Interactive demo (generate and display one sample)
python main.py generate --demo

# Generate for specific clinical categories
python main.py generate --num-samples 50 --categories Cardiology Pulmonology

# Specify output directory
python main.py generate --num-samples 100 --output-dir data/synthetic

# Use a specific model
python main.py generate --num-samples 50 --model gpt-4o
```

### Fine-tuning Commands

```bash
# Run full pipeline
python src/fine_tuning.py

# List recent jobs
python src/fine_tuning.py --list-jobs

# Check specific job status
python src/fine_tuning.py --job-status ftjob-xxx

# Custom model suffix
python src/fine_tuning.py --suffix my-clinical-model
```

### Inference Commands

```bash
# Interactive demo
python src/inference.py --demo

# Use base model (not fine-tuned)
python src/inference.py --demo --use-base

# Process test file
python src/inference.py --test-file data/processed/test.jsonl

# Specify model
python src/inference.py --model ft:gpt-4o-mini:xxx --demo
```

### Evaluation Commands

```bash
# Full evaluation
python src/evaluation.py \
    --predictions outputs/predictions.jsonl \
    --references data/processed/test.jsonl

# Skip BERTScore (faster)
python src/evaluation.py \
    --predictions outputs/predictions.jsonl \
    --references data/processed/test.jsonl \
    --skip-bertscore
```

## Expected Results

Based on the paper's findings, fine-tuned models should show:

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| ROUGE-1 F1 | ~0.25 | ~0.45 | +80% |
| ROUGE-L F1 | ~0.22 | ~0.40 | +82% |
| BERTScore F1 | ~0.70 | ~0.82 | +17% |

*Actual results will vary based on data quality and model selection.*

## Cost Considerations

### Fine-tuning Costs

| Tier | Use Case | Cost |
|------|----------|------|
| Global | Best value, uses global capacity | Lower |
| Standard | Data residency requirements | Standard |
| Developer | Experimentation (preview) | Lowest |

### Tips to Reduce Costs

1. Start with sample data to validate pipeline
2. Use `gpt-4o-mini` for initial experiments
3. Use Developer tier for experimentation
4. Limit epochs (3 is often sufficient)

## Troubleshooting

### Common Issues

**"Rate limit exceeded"**
- Wait and retry, or request quota increase in Azure portal

**"Model not found"**
- Verify model name supports fine-tuning
- Check deployment status in Azure AI Foundry

**"Authentication failed"**
- Run `az login` to refresh credentials
- Verify role assignments (Azure AI Owner/User)

**"File upload failed"**
- Ensure JSONL format is valid
- Check file size (< 512MB)
- Verify UTF-8 encoding with BOM

## References

- **Paper**: [Harmonising the Clinical Melody (arXiv:2409.14638)](https://arxiv.org/abs/2409.14638)
- **Azure AI Foundry Docs**: [Fine-tuning Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning)
- **Azure AI Projects SDK**: [Python SDK Reference](https://learn.microsoft.com/en-us/python/api/azure-ai-projects/)
- **MIMIC-III**: [PhysioNet](https://physionet.org/content/mimiciii/)

## License

This project is for educational and research purposes. Users must:
- Comply with MIMIC-III data use agreements if using clinical data
- Follow Azure AI responsible use guidelines
- Adhere to healthcare data privacy regulations (HIPAA, etc.)

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Tests pass
- Documentation is updated
