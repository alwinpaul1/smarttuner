#!/usr/bin/env python3
"""
Complete training pipeline: SFT -> GRPO as described in the article.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sft_trainer import SFTTrainer, SFTConfig, run_sft_training
from grpo_trainer import GRPOTrainer, GRPOConfig

def main():
    parser = argparse.ArgumentParser(description="Run complete training pipeline: SFT -> GRPO")
    parser.add_argument("--base_model", type=str, default="HuggingfaceTB/SmolLM-135M-Instruct",
                       help="Base model to start with")
    parser.add_argument("--environment", type=str, default="syllogism", 
                       choices=["syllogism", "propositional_logic"],
                       help="Environment to train on")
    
    # SFT parameters
    parser.add_argument("--sft_datapoints", type=int, default=200,
                       help="Number of SFT training examples")
    parser.add_argument("--sft_epochs", type=int, default=3,
                       help="SFT training epochs")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model for SFT data generation")
    
    # Model path options
    parser.add_argument("--grpo_model_path", type=str, default=None,
                       help="Explicit model path for GRPO stage (overrides default SFT output)")
    
    # GRPO parameters  
    parser.add_argument("--grpo_iterations", type=int, default=10,
                       help="Number of GRPO iterations")
    parser.add_argument("--grpo_dataset_size", type=int, default=100,
                       help="Dataset size per GRPO iteration")
    
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Base output directory")
    parser.add_argument("--skip_sft", action="store_true",
                       help="Skip SFT and use existing model")
    
    args = parser.parse_args()
    
    base_output = Path(args.output_dir)
    sft_output = base_output / "sft" / args.environment
    grpo_output = base_output / "grpo" / args.environment
    
    print("=" * 60)
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Environment: {args.environment}")
    print(f"SFT output: {sft_output}")
    print(f"GRPO output: {grpo_output}")
    print(f"Skip SFT: {args.skip_sft}")
    print()
    
    # Stage 1: Supervised Fine-tuning
    if not args.skip_sft:
        print("=" * 60)
        print("STAGE 1: SUPERVISED FINE-TUNING")
        print("=" * 60)
        
        sft_config = SFTConfig(
            model_name=args.base_model,
            environment_name=args.environment,
            num_datapoints=args.sft_datapoints,
            num_epochs=args.sft_epochs,
            openai_model=args.openai_model,
            output_dir=str(sft_output)
        )
        
        print(f"Training SFT model with {args.sft_datapoints} examples...")
        model, tokenizer, sft_results = run_sft_training(sft_config)
        
        print(f"SFT Results: {sft_results}")
        print(f"SFT model saved to: {sft_output}")
        print()
        
        # Save SFT results
        os.makedirs("results", exist_ok=True)
        sft_results_file = f"results/sft_{args.environment}_{Path(args.base_model).name}.json"
        with open(sft_results_file, "w") as f:
            json.dump(sft_results, f, indent=4)
        
        sft_model_path = str(sft_output)
    else:
        print("Skipping SFT - using existing model...")
        
        # Use explicit path if provided, otherwise use default SFT output path
        if args.grpo_model_path:
            sft_model_path = args.grpo_model_path
            print(f"Using explicit model path: {sft_model_path}")
        else:
            sft_model_path = str(sft_output)
            print(f"Using default SFT output path: {sft_model_path}")
        
        if not os.path.exists(sft_model_path):
            print(f"ERROR: Model not found at {sft_model_path}")
            if args.grpo_model_path:
                print("Please check the --grpo_model_path argument")
            else:
                print("Please run SFT first or use --grpo_model_path to specify model location")
            return
        
        print(f"✅ Found model at: {sft_model_path}")
    
    # Stage 2: GRPO Training
    print("=" * 60)  
    print("STAGE 2: GRPO REINFORCEMENT LEARNING")
    print("=" * 60)
    
    grpo_config = GRPOConfig(
        model_name=sft_model_path,
        environment_name=args.environment,
        # Use hyperparameters from the article
        max_new_tokens=300,
        exploration_batchsize=8,
        G=6,
        temperature=0.7,
        batch_size=16,
        gradient_accumulation_steps=12,
        learning_rate=1e-6,
        top_p=0.95,
        buffer_size=500
    )
    
    print(f"Training GRPO model for {args.grpo_iterations} iterations...")
    trainer = GRPOTrainer(grpo_config, output_dir=str(grpo_output))
    
    # Baseline evaluation
    print("Running baseline evaluation...")
    baseline_results = trainer.evaluate()
    print(f"Baseline results: {baseline_results}")
    
    # GRPO training
    trainer.train(
        num_iterations=args.grpo_iterations,
        dataset_size_per_iteration=args.grpo_dataset_size
    )
    
    # Final evaluation
    print("Running final evaluation...")
    final_results = trainer.evaluate()
    print(f"Final results: {final_results}")
    
    # Calculate improvements
    accuracy_improvement = final_results["accuracy"] - baseline_results["accuracy"]
    format_improvement = final_results["format_accuracy"] - baseline_results["format_accuracy"]
    
    # Save complete results
    complete_results = {
        "pipeline": "SFT -> GRPO",
        "base_model": args.base_model,
        "environment": args.environment,
        "sft_config": {
            "datapoints": args.sft_datapoints,
            "epochs": args.sft_epochs,
            "openai_model": args.openai_model
        },
        "grpo_config": {
            "iterations": args.grpo_iterations,
            "dataset_size": args.grpo_dataset_size,
            "hyperparameters": {
                "max_new_tokens": grpo_config.max_new_tokens,
                "exploration_batchsize": grpo_config.exploration_batchsize,
                "G": grpo_config.G,
                "temperature": grpo_config.temperature,
                "batch_size": grpo_config.batch_size,
                "learning_rate": grpo_config.learning_rate,
                "top_p": grpo_config.top_p,
                "buffer_size": grpo_config.buffer_size
            }
        },
        "results": {
            "baseline": baseline_results,
            "final": final_results,
            "improvements": {
                "accuracy": accuracy_improvement,
                "format_accuracy": format_improvement
            }
        }
    }
    
    pipeline_results_file = f"results/pipeline_{args.environment}_{Path(args.base_model).name}.json"
    with open(pipeline_results_file, "w") as f:
        json.dump(complete_results, f, indent=4)
    
    print()
    print("=" * 60)
    print("PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Environment: {args.environment}")
    print(f"SFT model: {sft_output}")
    print(f"Final GRPO model: {grpo_output}")
    print()
    print("Results Summary:")
    print(f"  Baseline accuracy: {baseline_results['accuracy']:.3f}")
    print(f"  Final accuracy: {final_results['accuracy']:.3f}")
    print(f"  Accuracy improvement: {accuracy_improvement:.3f}")
    print(f"  Format accuracy improvement: {format_improvement:.3f}")
    print()
    print(f"Complete results saved to: {pipeline_results_file}")
    
    if accuracy_improvement > 0:
        print(f"✅ SUCCESS: Model improved by {accuracy_improvement:.3f} accuracy points!")
    else:
        print(f"⚠️  WARNING: No improvement observed. Consider adjusting hyperparameters.")

if __name__ == "__main__":
    main()