"""
Evaluation Module for Hospital Course Summarization.

This module implements the evaluation metrics from the paper
"Harmonising the Clinical Melody" (arXiv:2409.14638):
- BERTScore: For semantic similarity evaluation
- ROUGE: For n-gram overlap evaluation
- Custom clinical metrics for coding utility assessment

The paper emphasizes that standard metrics may not fully capture
the quality of clinical summaries for coding purposes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
import jsonlines

# Evaluation metrics
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# For clinical term extraction
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # ROUGE scores
    rouge1_precision: float = 0.0
    rouge1_recall: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_precision: float = 0.0
    rouge2_recall: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_precision: float = 0.0
    rougeL_recall: float = 0.0
    rougeL_f1: float = 0.0
    
    # BERTScore
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0
    
    # Clinical-specific metrics
    clinical_term_overlap: float = 0.0
    diagnosis_coverage: float = 0.0
    procedure_coverage: float = 0.0
    
    # Summary statistics
    avg_generated_length: float = 0.0
    avg_reference_length: float = 0.0
    length_ratio: float = 0.0


class ClinicalEvaluator:
    """
    Evaluator for clinical summarization quality.
    
    Implements the evaluation methodology from the paper, combining:
    1. Standard NLG metrics (ROUGE, BERTScore)
    2. Clinical-specific metrics for coding utility
    """
    
    # Common clinical terms for domain-specific evaluation
    CLINICAL_INDICATORS = {
        "diagnoses": [
            "diagnosed", "diagnosis", "dx", "assessment", "impression",
            "condition", "disease", "syndrome", "disorder", "infection",
            "failure", "insufficiency", "acute", "chronic"
        ],
        "procedures": [
            "procedure", "surgery", "operation", "intervention", "catheterization",
            "biopsy", "resection", "implant", "transplant", "stent",
            "intubation", "ventilation", "dialysis"
        ],
        "medications": [
            "started", "initiated", "continued", "discontinued", "administered",
            "mg", "mcg", "units", "dose", "therapy", "treatment"
        ],
        "outcomes": [
            "improved", "resolved", "stable", "worsened", "deteriorated",
            "recovered", "discharged", "transferred", "expired"
        ]
    }
    
    def __init__(self, config=None):
        """Initialize the evaluator."""
        self.config = config or get_config().evaluation
        self._setup_nltk()
        self._setup_rouge()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def _setup_rouge(self):
        """Initialize ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.config.rouge_types,
            use_stemmer=True
        )
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE scores for predictions vs references.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        print("Computing ROUGE scores...")
        
        all_scores = {
            rouge_type: {"precision": [], "recall": [], "f1": []}
            for rouge_type in self.config.rouge_types
        }
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
            scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type in self.config.rouge_types:
                score = scores[rouge_type]
                all_scores[rouge_type]["precision"].append(score.precision)
                all_scores[rouge_type]["recall"].append(score.recall)
                all_scores[rouge_type]["f1"].append(score.fmeasure)
        
        # Compute averages
        avg_scores = {}
        for rouge_type, metrics in all_scores.items():
            avg_scores[rouge_type] = {
                "precision": np.mean(metrics["precision"]),
                "recall": np.mean(metrics["recall"]),
                "f1": np.mean(metrics["f1"])
            }
        
        return avg_scores
    
    def compute_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Compute BERTScore for semantic similarity evaluation.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            batch_size: Batch size for BERTScore computation
            
        Returns:
            Dictionary with BERTScore metrics
        """
        print(f"Computing BERTScore using {self.config.bert_model}...")
        
        # Truncate very long texts to avoid overflow errors
        max_length = 500  # tokens
        predictions_truncated = [pred[:max_length] for pred in predictions]
        references_truncated = [ref[:max_length] for ref in references]
        
        P, R, F1 = bert_score(
            predictions_truncated, 
            references_truncated,
            model_type=self.config.bert_model,
            batch_size=batch_size,
            verbose=True,
            num_layers=9  # Use fewer layers for efficiency
        )
        
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    
    def compute_clinical_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute clinical-specific metrics for coding utility.
        
        This implements a simplified version of the paper's 
        "hospital course summary assessment metric" focusing on:
        - Clinical term overlap
        - Diagnosis coverage
        - Procedure coverage
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with clinical metrics
        """
        print("Computing clinical-specific metrics...")
        
        all_metrics = {
            "term_overlap": [],
            "diagnosis_coverage": [],
            "procedure_coverage": []
        }
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
            pred_tokens = set(word_tokenize(pred.lower()))
            ref_tokens = set(word_tokenize(ref.lower()))
            
            # General clinical term overlap
            pred_clinical = self._extract_clinical_terms(pred_tokens)
            ref_clinical = self._extract_clinical_terms(ref_tokens)
            
            if ref_clinical:
                overlap = len(pred_clinical & ref_clinical) / len(ref_clinical)
                all_metrics["term_overlap"].append(overlap)
            
            # Diagnosis-related term coverage
            pred_diag = self._extract_category_terms(pred_tokens, "diagnoses")
            ref_diag = self._extract_category_terms(ref_tokens, "diagnoses")
            
            if ref_diag:
                diag_coverage = len(pred_diag & ref_diag) / len(ref_diag)
                all_metrics["diagnosis_coverage"].append(diag_coverage)
            
            # Procedure-related term coverage
            pred_proc = self._extract_category_terms(pred_tokens, "procedures")
            ref_proc = self._extract_category_terms(ref_tokens, "procedures")
            
            if ref_proc:
                proc_coverage = len(pred_proc & ref_proc) / len(ref_proc)
                all_metrics["procedure_coverage"].append(proc_coverage)
        
        return {
            "clinical_term_overlap": np.mean(all_metrics["term_overlap"]) if all_metrics["term_overlap"] else 0.0,
            "diagnosis_coverage": np.mean(all_metrics["diagnosis_coverage"]) if all_metrics["diagnosis_coverage"] else 0.0,
            "procedure_coverage": np.mean(all_metrics["procedure_coverage"]) if all_metrics["procedure_coverage"] else 0.0
        }
    
    def _extract_clinical_terms(self, tokens: set) -> set:
        """Extract clinical terms from token set."""
        clinical_terms = set()
        for category_terms in self.CLINICAL_INDICATORS.values():
            clinical_terms.update(tokens & set(category_terms))
        return clinical_terms
    
    def _extract_category_terms(self, tokens: set, category: str) -> set:
        """Extract terms from a specific clinical category."""
        return tokens & set(self.CLINICAL_INDICATORS.get(category, []))
    
    def compute_length_statistics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length-related statistics.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with length statistics
        """
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        return {
            "avg_generated_length": np.mean(pred_lengths),
            "avg_reference_length": np.mean(ref_lengths),
            "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        }
    
    def evaluate(
        self, 
        predictions: List[str], 
        references: List[str],
        compute_bertscore: bool = True,
        compute_clinical: bool = True
    ) -> EvaluationResult:
        """
        Run complete evaluation suite.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            compute_bertscore: Whether to compute BERTScore (can be slow)
            compute_clinical: Whether to compute clinical-specific metrics
            
        Returns:
            EvaluationResult with all metrics
        """
        print(f"\n{'='*60}")
        print("  Hospital Course Summarization Evaluation")
        print(f"  Samples: {len(predictions)}")
        print(f"{'='*60}\n")
        
        result = EvaluationResult()
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(predictions, references)
        result.rouge1_precision = rouge_scores["rouge1"]["precision"]
        result.rouge1_recall = rouge_scores["rouge1"]["recall"]
        result.rouge1_f1 = rouge_scores["rouge1"]["f1"]
        result.rouge2_precision = rouge_scores["rouge2"]["precision"]
        result.rouge2_recall = rouge_scores["rouge2"]["recall"]
        result.rouge2_f1 = rouge_scores["rouge2"]["f1"]
        result.rougeL_precision = rouge_scores["rougeL"]["precision"]
        result.rougeL_recall = rouge_scores["rougeL"]["recall"]
        result.rougeL_f1 = rouge_scores["rougeL"]["f1"]
        
        # BERTScore
        if compute_bertscore:
            bert_scores = self.compute_bertscore(predictions, references)
            result.bert_precision = bert_scores["precision"]
            result.bert_recall = bert_scores["recall"]
            result.bert_f1 = bert_scores["f1"]
        
        # Clinical metrics
        if compute_clinical:
            clinical_metrics = self.compute_clinical_metrics(predictions, references)
            result.clinical_term_overlap = clinical_metrics["clinical_term_overlap"]
            result.diagnosis_coverage = clinical_metrics["diagnosis_coverage"]
            result.procedure_coverage = clinical_metrics["procedure_coverage"]
        
        # Length statistics
        length_stats = self.compute_length_statistics(predictions, references)
        result.avg_generated_length = length_stats["avg_generated_length"]
        result.avg_reference_length = length_stats["avg_reference_length"]
        result.length_ratio = length_stats["length_ratio"]
        
        return result
    
    def load_test_data(self, test_file: str) -> Tuple[List[str], List[str]]:
        """
        Load test data from JSONL file.
        
        Args:
            test_file: Path to test JSONL file
            
        Returns:
            Tuple of (inputs, references)
        """
        inputs = []
        references = []
        
        with jsonlines.open(test_file) as reader:
            for record in reader:
                messages = record["messages"]
                
                # Extract user input (clinical notes)
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"),
                    None
                )
                
                # Extract reference (ground truth summary)
                assistant_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"),
                    None
                )
                
                if user_msg and assistant_msg:
                    inputs.append(user_msg)
                    references.append(assistant_msg)
        
        return inputs, references
    
    def generate_report(
        self, 
        result: EvaluationResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            result: EvaluationResult object
            output_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = f"""
{'='*60}
        HOSPITAL COURSE SUMMARIZATION EVALUATION REPORT
        Based on: "Harmonising the Clinical Melody" (arXiv:2409.14638)
{'='*60}

ROUGE SCORES
------------
ROUGE-1:
  Precision: {result.rouge1_precision:.4f}
  Recall:    {result.rouge1_recall:.4f}
  F1:        {result.rouge1_f1:.4f}

ROUGE-2:
  Precision: {result.rouge2_precision:.4f}
  Recall:    {result.rouge2_recall:.4f}
  F1:        {result.rouge2_f1:.4f}

ROUGE-L:
  Precision: {result.rougeL_precision:.4f}
  Recall:    {result.rougeL_recall:.4f}
  F1:        {result.rougeL_f1:.4f}

BERTSCORE
---------
  Precision: {result.bert_precision:.4f}
  Recall:    {result.bert_recall:.4f}
  F1:        {result.bert_f1:.4f}

CLINICAL METRICS
----------------
  Clinical Term Overlap: {result.clinical_term_overlap:.4f}
  Diagnosis Coverage:    {result.diagnosis_coverage:.4f}
  Procedure Coverage:    {result.procedure_coverage:.4f}

SUMMARY STATISTICS
------------------
  Avg Generated Length: {result.avg_generated_length:.1f} words
  Avg Reference Length: {result.avg_reference_length:.1f} words
  Length Ratio:         {result.length_ratio:.2f}

{'='*60}
"""
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")
            
            # Also save as JSON for programmatic access
            json_path = output_path.replace(".txt", ".json")
            with open(json_path, "w") as f:
                json.dump(result.__dict__, f, indent=2)
        
        return report


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate hospital course summarization model"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--references",
        type=str,
        required=True,
        help="Path to references JSONL file (test set)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_report.txt",
        help="Path to save evaluation report"
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation (faster)"
    )
    
    args = parser.parse_args()
    
    evaluator = ClinicalEvaluator()
    
    # Load data
    print(f"Loading predictions from {args.predictions}...")
    with jsonlines.open(args.predictions) as reader:
        predictions = [record["prediction"] for record in reader]
    
    print(f"Loading references from {args.references}...")
    _, references = evaluator.load_test_data(args.references)
    
    # Run evaluation
    result = evaluator.evaluate(
        predictions,
        references,
        compute_bertscore=not args.skip_bertscore
    )
    
    # Generate report
    report = evaluator.generate_report(result, args.output)
    print(report)


if __name__ == "__main__":
    main()
