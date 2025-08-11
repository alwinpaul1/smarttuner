#!/usr/bin/env python3
"""
Main GRPO training script following the implementation from the article.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from grpo_trainer import GRPOTrainer, GRPOConfig

def main():
    parser = argparse.ArgumentParser(description="Run GRPO training for reasoning models")
    parser.add_argument("--model_name", type=str, default="models/sft",
                       help="Model to train (should be SFT model path)")
    parser.add_argument("--environment", type=str, default="syllogism", 
                       choices=["syllogism", "propositional_logic"],
                       help="Environment to train on")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of GRPO training iterations")
    parser.add_argument("--dataset_size_per_iteration", type=int, default=100,
                       help="Dataset size for each iteration")
    parser.add_argument("--output_dir", type=str, default="models/grpo",
                       help="Output directory for trained model")
    
    # GRPO hyperparameters from the article
    parser.add_argument("--max_new_tokens", type=int, default=300,
                       help="Max tokens to generate")
    parser.add_argument("--exploration_batchsize", type=int, default=8,
                       help="Batch size for exploration")
    parser.add_argument("--G", type=int, default=6,
                       help="Number of responses per group")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="Learning rate (keep low)")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling")
    parser.add_argument("--buffer_size", type=int, default=500,
                       help="Experience buffer size")
    parser.add_argument("--dataset_seed", type=int, default=42,
                       help="Seed for experience collection data generation")
    parser.add_argument("--eval_seed", type=int, default=123,
                       help="Seed for evaluation data generation")
    
    # Visualization options
    parser.add_argument("--show_plots", action="store_true",
                       help="Show real-time training plots during training")
    parser.add_argument("--save_plots", action="store_true", 
                       help="Save training plots automatically")
    
    args = parser.parse_args()
    
    # Create config with hyperparameters from the article
    config = GRPOConfig(
        model_name=args.model_name,
        environment_name=args.environment,
        max_new_tokens=args.max_new_tokens,
        exploration_batchsize=args.exploration_batchsize,
        G=args.G,
        temperature=args.temperature,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        top_p=args.top_p,
        buffer_size=args.buffer_size,
        dataset_seed=args.dataset_seed,
        eval_seed=args.eval_seed
    )
    
    print("GRPO Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Environment: {config.environment_name}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Dataset per iteration: {args.dataset_size_per_iteration}")
    print(f"  Output dir: {args.output_dir}")
    print()
    print("GRPO Hyperparameters (from article):")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Exploration batch size: {config.exploration_batchsize}")
    print(f"  G (responses per group): {config.G}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Buffer size: {config.buffer_size}")
    print()
    
    # Create trainer
    trainer = GRPOTrainer(config, output_dir=args.output_dir)
    
    # Run baseline evaluation
    print("Running baseline evaluation...")
    baseline_results = trainer.evaluate()
    print(f"Baseline results: {baseline_results}")
    
    # Run GRPO training
    print("Starting GRPO training...")
    if args.show_plots:
        print("📊 Real-time plots enabled")
    if args.save_plots:
        print("💾 Plot saving enabled")
        
    trainer.train(
        num_iterations=args.num_iterations,
        dataset_size_per_iteration=args.dataset_size_per_iteration,
        show_plots=args.show_plots,
        save_plots=args.save_plots
    )
    
    # Final evaluation
    print("Running final evaluation...")
    final_results = trainer.evaluate()
    print(f"Final results: {final_results}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "baseline": baseline_results,
        "final": final_results,
        "improvement": {
            "accuracy": final_results["accuracy"] - baseline_results["accuracy"],
            "format_accuracy": final_results["format_accuracy"] - baseline_results["format_accuracy"]
        },
        "config": {
            "model_name": config.model_name,
            "environment_name": config.environment_name,
            "num_iterations": args.num_iterations,
            "dataset_size_per_iteration": args.dataset_size_per_iteration,
            "hyperparameters": {
                "max_new_tokens": config.max_new_tokens,
                "exploration_batchsize": config.exploration_batchsize,
                "G": config.G,
                "temperature": config.temperature,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "top_p": config.top_p,
                "buffer_size": config.buffer_size
            }
        }
    }
    
    results_file = f"results/grpo_{config.environment_name}_{Path(config.model_name).name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {results_file}")
    print(f"Accuracy improvement: {results['improvement']['accuracy']:.3f}")
    print(f"Format accuracy improvement: {results['improvement']['format_accuracy']:.3f}")
    
    # Generate final plots if requested
    if args.save_plots or args.show_plots:
        print("\n📊 Generating final training plots...")
        try:
            # Add training history to results for visualization
            results["training_history"] = trainer.get_training_summary()["training_history"]
            
            # Re-save with training history
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
            
            # Create visualizations
            from pathlib import Path
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "src"))
            from visualizer import TrainingVisualizer
            
            visualizer = TrainingVisualizer()
            plot_name = f"grpo_{config.environment_name}_{Path(config.model_name).name}"
            
            # Training curves
            visualizer.plot_grpo_training_curves(results["training_history"], plot_name)
            
            # Comparison plot
            visualizer.plot_comparison(baseline_results, final_results, 
                                     config.model_name, config.environment_name, plot_name)
            
            print("✅ Plots generated successfully!")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")

if __name__ == "__main__":
    main()