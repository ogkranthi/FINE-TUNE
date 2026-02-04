"""
Hospital Course Summarization - Main Entry Point

This script provides a unified CLI for the complete fine-tuning pipeline:
1. Data preparation
2. Fine-tuning
3. Inference
4. Evaluation
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Hospital Course Summarization with Azure AI Foundry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py prepare --sample           Create sample training data
  python main.py generate --num-samples 100 Generate synthetic data with Azure AI
  python main.py generate --demo            Demo synthetic generation
  python main.py train                      Run fine-tuning pipeline
  python main.py train --no-monitor         Start training without waiting
  python main.py infer --demo               Run interactive demo
  python main.py evaluate                   Evaluate predictions
  python main.py status --job-id ftjob-xxx  Check job status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data preparation command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training data")
    prepare_parser.add_argument(
        "--sample", 
        action="store_true",
        help="Create sample data for testing"
    )
    prepare_parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of sample records to create"
    )
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Start training without waiting for completion"
    )
    train_parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom suffix for the fine-tuned model"
    )
    train_parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Use existing uploaded files"
    )
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo"
    )
    infer_parser.add_argument(
        "--test-file",
        type=str,
        help="Process test file"
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.jsonl",
        help="Output file for predictions"
    )
    infer_parser.add_argument(
        "--use-base",
        action="store_true",
        help="Use base model instead of fine-tuned"
    )
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    eval_parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/predictions.jsonl",
        help="Path to predictions file"
    )
    eval_parser.add_argument(
        "--references",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to reference file"
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_report.txt",
        help="Path for evaluation report"
    )
    eval_parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check training status")
    status_parser.add_argument(
        "--job-id",
        type=str,
        help="Specific job ID to check"
    )
    status_parser.add_argument(
        "--list",
        action="store_true",
        help="List all recent jobs"
    )
    
    # Synthetic data generation command
    generate_parser = subparsers.add_parser(
        "generate", 
        help="Generate synthetic clinical data using Azure AI Foundry"
    )
    generate_parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of synthetic samples to generate"
    )
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for generated data"
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model deployment name for generation"
    )
    generate_parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Limit to specific clinical categories (e.g., Cardiology Pulmonology)"
    )
    generate_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo (generate one sample)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "prepare":
        from src.data_preparation import DataPreparator
        
        preparator = DataPreparator()
        if args.sample:
            preparator.create_sample_data(num_samples=args.num_samples)
        else:
            # Try to process real data
            from src.data_preparation import main as prepare_main
            prepare_main()
    
    elif args.command == "train":
        from src.fine_tuning import FineTuningPipeline
        
        pipeline = FineTuningPipeline()
        results = pipeline.run_pipeline(
            upload_new=not args.no_upload,
            monitor=not args.no_monitor,
            model_suffix=args.suffix
        )
        print(f"Training results: {results}")
    
    elif args.command == "infer":
        from src.inference import ClinicalSummarizer
        
        summarizer = ClinicalSummarizer(use_fine_tuned=not args.use_base)
        
        if args.demo:
            summarizer.interactive_demo()
        elif args.test_file:
            summarizer.process_test_set(args.test_file, args.output)
        else:
            summarizer.interactive_demo()
    
    elif args.command == "evaluate":
        from src.evaluation import ClinicalEvaluator
        import jsonlines
        
        evaluator = ClinicalEvaluator()
        
        # Load predictions
        with jsonlines.open(args.predictions) as reader:
            predictions = [record["prediction"] for record in reader]
        
        # Load references
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
    
    elif args.command == "status":
        from src.fine_tuning import FineTuningPipeline
        
        pipeline = FineTuningPipeline()
        
        if args.list:
            pipeline.list_jobs()
        elif args.job_id:
            result = pipeline.monitor_job(args.job_id, poll_interval=1, max_wait_hours=0)
            print(f"Job status: {result}")
        else:
            pipeline.list_jobs()
    
    elif args.command == "generate":
        from src.synthetic_data import SyntheticDataGenerator
        
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
