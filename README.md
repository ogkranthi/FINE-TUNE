# Healthcare AI Fine-Tuning Project 🏥

Transform clinical documentation with AI! This project fine-tunes GPT-4o to automatically generate accurate Hospital Course summaries from complex medical documentation, helping clinical coders save hours per patient record.

[![Azure](https://img.shields.io/badge/Azure-AI%20Foundry-blue)](https://azure.microsoft.com/en-us/products/ai-foundry/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)](https://openai.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)](https://python.org/)

## 🎯 Project Overview

**The Challenge:** Clinical coders spend hours reading lengthy patient notes to create Brief Hospital Course summaries for billing and compliance. This manual process is time-consuming and prone to inconsistencies.

**The Solution:** Fine-tuned GPT-4o to automatically generate concise, accurate clinical summaries from complex medical documentation using Azure AI Foundry.

## 🚀 Key Features

- ✅ **Fine-tuned GPT-4o** using Azure AI Foundry
- ✅ **End-to-end pipeline**: data prep → training → inference → evaluation
- ✅ **Synthetic clinical data generator** for training
- ✅ **Comprehensive evaluation** with ROUGE metrics
- ✅ **Full Python SDK integration** with Azure OpenAI

## 📊 Results

### Model Performance Comparison

Our fine-tuned model shows significant improvements in generating clinical summaries:

```
============================================================
        HOSPITAL COURSE SUMMARIZATION EVALUATION REPORT
        Based on: "Harmonising the Clinical Melody" (arXiv:2409.14638)
============================================================

ROUGE SCORES
------------
ROUGE-1:
  Precision: 0.7696  ← High precision in word overlap
  Recall:    0.7500  ← Good coverage of reference content
  F1:        0.7597  ← Strong overall performance

ROUGE-2:
  Precision: 0.4803  ← Effective bigram matching
  Recall:    0.4681  ← Consistent phrase capture
  F1:        0.4741  ← Solid semantic understanding

ROUGE-L:
  Precision: 0.5261  ← Good sentence structure matching
  Recall:    0.5127  ← Maintains clinical narrative flow
  F1:        0.5193  ← Balanced performance

CLINICAL METRICS
----------------
  Clinical Term Overlap: 0.3750  ← Preserves medical terminology
  Avg Generated Length:  223.0 words
  Avg Reference Length:  228.0 words
  Length Ratio:          0.98    ← Optimal summary length
```

### Sample Output Comparison

**Input:** Complex 7-day hospitalization documentation for bowel obstruction patient

**Fine-tuned Model Output:**
```
The patient, a 38-year-old female, was admitted with a diagnosis of mechanical bowel 
obstruction, presenting with severe abdominal pain, nausea, and constipation for four days. 
Initial management included the insertion of a nasogastric tube (NGT) for bowel decompression, 
administration of intravenous fluids, and pain control with IV morphine...

[Continues with structured, clinically accurate summary]
```

**Reference Standard:**
```
The patient, a 38-year-old female, was admitted with a diagnosis of mechanical small bowel 
obstruction, presenting with severe abdominal pain, nausea, and abdominal distension. 
Initial imaging (X-ray and CT) confirmed moderate small bowel distension...

[Clinical reference summary]
```

**Key Improvements:**
- 📈 **76% ROUGE-1 F1 Score** - Excellent content capture
- ⚡ **98% Length Ratio** - Optimal summary conciseness
- 🎯 **37.5% Clinical Term Overlap** - Preserves medical accuracy
- 🔄 **Consistent Structure** - Maintains clinical documentation standards

## 📁 Project Structure

```
fine-tune/
├── data/
│   ├── processed/           # Processed training/validation data
│   │   ├── train.jsonl     # Training dataset
│   │   ├── validation.jsonl # Validation dataset
│   │   └── test.jsonl      # Test dataset
│   └── samples/            # Sample training data
├── outputs/                # Model outputs and evaluations
│   ├── evaluation_report.txt    # Detailed performance metrics
│   ├── predictions_finetuned.jsonl  # Fine-tuned model predictions
│   └── predictions_base.jsonl      # Base model predictions
├── src/                    # Source code
├── config/                 # Configuration files
└── linkedin_post.md       # Project showcase content
```

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Azure AI Foundry account

### Installation
```bash
# Clone the repository
git clone https://github.com/ogkranthi/fine-tune.git
cd fine-tune

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure API keys and endpoints
```

### Quick Start
```python
# Load and use the fine-tuned model
from azure.openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-api-key",
    api_version="2024-02-01",
    azure_endpoint="your-endpoint"
)

# Generate hospital course summary
response = client.chat.completions.create(
    model="your-finetuned-model",
    messages=[{
        "role": "user", 
        "content": f"Generate a hospital course summary: {clinical_documentation}"
    }]
)

summary = response.choices[0].message.content
```

## 📈 Business Impact

- **⏱️ Time Saving**: Reduces clinical coding time from hours to minutes
- **📋 Consistency**: Standardizes summary format across all patient records  
- **🎯 Accuracy**: Maintains clinical accuracy with 76% ROUGE-1 F1 score
- **💰 Cost Effective**: Scales to thousands of patient records automatically

## 🔬 Technical Deep Dive

### Data Pipeline
1. **Synthetic Data Generation**: Created realistic clinical scenarios
2. **Data Preprocessing**: Structured clinical notes for training
3. **Fine-tuning**: Used Azure AI Foundry for model customization
4. **Evaluation**: Comprehensive metrics including ROUGE scores

### Model Architecture
- **Base Model**: GPT-4o
- **Fine-tuning Method**: Supervised fine-tuning on clinical data
- **Evaluation Framework**: ROUGE metrics + clinical-specific measures
- **Deployment**: Azure OpenAI Service integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Based on research from "Harmonising the Clinical Melody" (arXiv:2409.14638)
- Built with Azure AI Foundry and OpenAI GPT-4o
- Inspired by the need to improve clinical documentation efficiency

---

**Tech Stack:** Azure AI Foundry | OpenAI GPT-4o | Python | Azure OpenAI SDK

*Always learning, always building!* 🚀

#ArtificialIntelligence #MachineLearning #Healthcare #Azure #LLM #FineTuning #ClinicalAI