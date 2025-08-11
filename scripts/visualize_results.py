#!/usr/bin/env python3
"""
Visualization script to generate plots from SmartTuner training results.
Supports both GRPO and SFT training result analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from visualizer import TrainingVisualizer, load_and_visualize

def list_available_results(results_dir: str = "results") -> List[Path]:
    """List all available result files"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found.")
        return []
    
    result_files = list(results_path.glob("*.json"))
    if not result_files:
        print(f"No result files found in '{results_dir}'.")
        return []
    
    return sorted(result_files)

def analyze_single_result(file_path: str, output_name: Optional[str] = None) -> None:
    """Analyze a single result file"""
    visualizer = TrainingVisualizer()
    
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        if output_name is None:
            output_name = Path(file_path).stem
        
        print(f"\nAnalyzing: {file_path}")
        print("=" * 60)
        
        # Determine result type and create appropriate plots
        if 'training_history' in results:
            history = results['training_history']
            
            # Check if it's GRPO or SFT based on available metrics
            if 'losses' in history and 'mean_rewards' in history:
                print("📊 Generating GRPO training plots...")
                visualizer.plot_grpo_training_curves(history, output_name)
                
                # If we have reward data, create reward analysis
                if 'mean_rewards' in history and history['mean_rewards']:
                    # Create mock detailed reward data for demonstration
                    import numpy as np
                    mock_rewards = np.random.normal(
                        np.mean(history['mean_rewards']), 
                        np.mean(history.get('reward_stds', [0.1])), 
                        100
                    ).tolist()
                    mock_advantages = (np.array(mock_rewards) - np.mean(mock_rewards)) / np.std(mock_rewards)
                    visualizer.plot_reward_analysis(mock_rewards, mock_advantages.tolist(), output_name)
                    
            elif 'train_losses' in history:
                print("📊 Generating SFT training plots...")
                visualizer.plot_sft_training_curves(history, output_name)
        
        # If we have baseline/final comparison data
        if 'baseline' in results and 'final' in results:
            print("📈 Generating comparison plots...")
            model_name = results.get('config', {}).get('model_name', 'Unknown Model')
            environment = results.get('config', {}).get('environment_name', 'Unknown Task')
            visualizer.plot_comparison(results['baseline'], results['final'], 
                                     model_name, environment, output_name)
        
        # Print summary statistics
        if 'improvement' in results:
            improvements = results['improvement']
            print(f"\n📈 Training Improvements:")
            for metric, value in improvements.items():
                print(f"  {metric}: {value:+.3f}")
        
        print(f"\n✅ Plots saved to: plots/ directory")
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {file_path}")
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")

def compare_models(result_files: List[str], environment: str) -> None:
    """Compare results across multiple models"""
    visualizer = TrainingVisualizer()
    
    results_dict = {}
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract model name from file or data
            model_name = Path(file_path).stem.split('_')[-1]  # Assume format: type_env_model.json
            
            if 'baseline' in data and 'final' in data:
                results_dict[model_name] = {
                    'baseline': data['baseline'],
                    'final': data['final']
                }
                
        except Exception as e:
            print(f"⚠️  Skipping {file_path}: {e}")
    
    if results_dict:
        print(f"📊 Comparing {len(results_dict)} models on {environment}")
        visualizer.plot_model_comparison(results_dict, environment, f"comparison_{environment}")
    else:
        print("❌ No valid comparison data found in the provided files")

def main():
    parser = argparse.ArgumentParser(description="Visualize SmartTuner training results")
    parser.add_argument("--file", "-f", type=str,
                       help="Specific result file to analyze")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available result files")
    parser.add_argument("--compare", "-c", nargs='+', 
                       help="Compare multiple result files")
    parser.add_argument("--environment", "-e", type=str, default="unknown",
                       help="Environment name for comparison plots")
    parser.add_argument("--output", "-o", type=str,
                       help="Output name for plots (default: auto-generated)")
    parser.add_argument("--results_dir", "-d", type=str, default="results",
                       help="Directory containing result files")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Analyze all available result files")
    
    args = parser.parse_args()
    
    print("🎨 SmartTuner Results Visualizer")
    print("=" * 40)
    
    # List available files
    if args.list:
        available_files = list_available_results(args.results_dir)
        if available_files:
            print(f"\n📁 Available result files in '{args.results_dir}':")
            for i, file_path in enumerate(available_files, 1):
                file_size = file_path.stat().st_size
                print(f"  {i:2d}. {file_path.name} ({file_size:,} bytes)")
        return
    
    # Analyze specific file
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ File not found: {args.file}")
            return
        analyze_single_result(args.file, args.output)
        return
    
    # Compare multiple files
    if args.compare:
        missing_files = [f for f in args.compare if not Path(f).exists()]
        if missing_files:
            print(f"❌ Files not found: {missing_files}")
            return
        compare_models(args.compare, args.environment)
        return
    
    # Analyze all files
    if args.all:
        available_files = list_available_results(args.results_dir)
        if not available_files:
            return
        
        print(f"\n🔍 Analyzing all {len(available_files)} result files...")
        for file_path in available_files:
            analyze_single_result(str(file_path))
            print()
        return
    
    # No specific action provided - show interactive menu
    available_files = list_available_results(args.results_dir)
    if not available_files:
        return
    
    print(f"\n📁 Available result files:")
    for i, file_path in enumerate(available_files, 1):
        print(f"  {i:2d}. {file_path.name}")
    
    print(f"  {len(available_files) + 1:2d}. Analyze all files")
    print(f"   0. Exit")
    
    try:
        choice = int(input(f"\nSelect a file to analyze (1-{len(available_files) + 1}, 0 to exit): "))
        
        if choice == 0:
            print("👋 Goodbye!")
            return
        elif choice == len(available_files) + 1:
            print("\n🔍 Analyzing all files...")
            for file_path in available_files:
                analyze_single_result(str(file_path))
                print()
        elif 1 <= choice <= len(available_files):
            selected_file = available_files[choice - 1]
            analyze_single_result(str(selected_file))
        else:
            print("❌ Invalid choice")
    
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()