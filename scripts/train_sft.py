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
    
    # Visualization options
    parser.add_argument("--show_plots", action="store_true",
                       help="Show training plots after completion")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save training plots automatically")
    
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
        if args.show_plots:
            print("📊 Plot visualization enabled")
        if args.save_plots:
            print("💾 Plot saving enabled")
            
        model, tokenizer, results = trainer.train()
        
        # Save results with training history
        os.makedirs("results", exist_ok=True)
        results_file = f"results/sft_{config.environment_name}_{config.model_name.split('/')[-1]}.json"
        
        # Add training history to results
        complete_results = {
            **results,
            "training_history": trainer.get_training_summary()["training_history"],
            "config": {
                "model_name": config.model_name,
                "environment_name": config.environment_name,
                "num_datapoints": config.num_datapoints,
                "num_epochs": config.num_epochs
            }
        }
        
        with open(results_file, "w") as f:
            json.dump(complete_results, f, indent=4)
        
        print(f"Training completed!")
        print(f"Results saved to: {results_file}")
        print(f"Final results: {results}")
        
        # Generate plots if requested
        if args.save_plots or args.show_plots:
            print("\n📊 Generating SFT training plots...")
            try:
                plot_name = f"sft_{config.environment_name}_{config.model_name.split('/')[-1]}"
                
                if args.show_plots and args.save_plots:
                    trainer.show_training_plots(plot_name)  # Show and save
                elif args.show_plots:
                    trainer.show_training_plots(plot_name)  # Show only
                elif args.save_plots:
                    trainer.save_training_plots(plot_name)  # Save only
                
                print("✅ SFT plots generated successfully!")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not generate SFT plots: {e}")

if __name__ == "__main__":
    main()