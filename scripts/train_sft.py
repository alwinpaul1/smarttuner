#!/usr/bin/env python3
"""
Script to run supervised fine-tuning as described in the article.
This is the warmup phase before GRPO training.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sft_trainer import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning for reasoning models")
    parser.add_argument("--model_name", type=str, default="HuggingfaceTB/SmolLM-135M-Instruct",
                       help="Model to fine-tune")
    parser.add_argument("--environment", type=str, default="syllogism", 
                       choices=["syllogism", "propositional_logic"],
                       help="Environment to train on")
    parser.add_argument("--num_datapoints", type=int, default=200,
                       help="Number of training examples to generate")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to use for data generation")
    parser.add_argument("--output_dir", type=str, default="models/sft",
                       help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation on existing model")
    parser.add_argument("--dataset_seed", type=int, default=42,
                       help="Seed for training data generation")
    parser.add_argument("--eval_seed", type=int, default=123,
                       help="Seed for evaluation data generation")
    
    args = parser.parse_args()
    
    # Create config
    config = SFTConfig(
        model_name=args.model_name,
        environment_name=args.environment,
        num_datapoints=args.num_datapoints,
        openai_model=args.openai_model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dataset_seed=args.dataset_seed,
        eval_seed=args.eval_seed
    )
    
    print(f"SFT Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Environment: {config.environment_name}")
    print(f"  Data points: {config.num_datapoints}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Eval only: {args.eval_only}")
    print()
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    if args.eval_only:
        # Load existing model and evaluate
        print("Running evaluation only...")
        results = trainer.evaluate_sft_model()
        print(f"Results: {results}")
    else:
        # Run full training
        print("Starting SFT training...")
        model, tokenizer, results = trainer.train()
        
        # Save results
        os.makedirs("results", exist_ok=True)
        results_file = f"results/sft_{config.environment_name}_{config.model_name.split('/')[-1]}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Training completed!")
        print(f"Results saved to: {results_file}")
        print(f"Final results: {results}")

if __name__ == "__main__":
    main()