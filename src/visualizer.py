"""
Training Visualization Module for SmartTuner GRPO Implementation
Provides comprehensive plotting capabilities for training analysis and monitoring.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Comprehensive visualization class for GRPO and SFT training analysis"""
    
    def __init__(self, save_dir: str = "plots", figsize: Tuple[int, int] = (12, 8)):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        
    def plot_grpo_training_curves(self, training_history: Dict[str, List], 
                                save_name: Optional[str] = None, 
                                show_plots: bool = True) -> None:
        """Plot GRPO training curves: loss, rewards, and accuracy over iterations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Training Progress', fontsize=16, fontweight='bold')
        
        iterations = range(1, len(training_history.get('losses', [])) + 1)
        
        # Training Loss
        if 'losses' in training_history:
            axes[0, 0].plot(iterations, training_history['losses'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('PPO Training Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Reward Statistics
        if 'mean_rewards' in training_history and 'reward_stds' in training_history:
            mean_rewards = training_history['mean_rewards']
            reward_stds = training_history['reward_stds']
            
            axes[0, 1].plot(iterations, mean_rewards, 'g-', linewidth=2, marker='s', label='Mean Reward')
            axes[0, 1].fill_between(iterations, 
                                  np.array(mean_rewards) - np.array(reward_stds),
                                  np.array(mean_rewards) + np.array(reward_stds),
                                  alpha=0.3, color='green')
            axes[0, 1].set_title('Reward Distribution (Mean ± Std)', fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Accuracy Progress
        if 'accuracies' in training_history:
            axes[1, 0].plot(iterations, training_history['accuracies'], 'r-', linewidth=2, marker='^')
            axes[1, 0].set_title('Accuracy Over Training', fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Format vs Correctness Rewards
        if 'format_rewards' in training_history and 'correctness_rewards' in training_history:
            axes[1, 1].plot(iterations, training_history['format_rewards'], 'orange', 
                          linewidth=2, marker='d', label='Format Reward (15%)')
            axes[1, 1].plot(iterations, training_history['correctness_rewards'], 'purple', 
                          linewidth=2, marker='v', label='Correctness Reward (85%)')
            axes[1, 1].set_title('Reward Components', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Reward Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_grpo_training.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_sft_training_curves(self, training_history: Dict[str, List], 
                               save_name: Optional[str] = None, 
                               show_plots: bool = True) -> None:
        """Plot SFT training curves: loss and accuracy over epochs"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Supervised Fine-tuning Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(training_history.get('train_losses', [])) + 1)
        
        # Training and Validation Loss
        if 'train_losses' in training_history:
            axes[0].plot(epochs, training_history['train_losses'], 'b-', 
                        linewidth=2, marker='o', label='Training Loss')
            
        if 'val_losses' in training_history:
            axes[0].plot(epochs, training_history['val_losses'], 'r--', 
                        linewidth=2, marker='s', label='Validation Loss')
            
        axes[0].set_title('Training Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Accuracy Progress
        if 'train_accuracy' in training_history:
            axes[1].plot(epochs, training_history['train_accuracy'], 'g-', 
                        linewidth=2, marker='^', label='Training Accuracy')
            
        if 'val_accuracy' in training_history:
            axes[1].plot(epochs, training_history['val_accuracy'], 'orange', 
                        linewidth=2, marker='d', label='Validation Accuracy')
            
        axes[1].set_title('Accuracy Progress', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_sft_training.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_reward_analysis(self, rewards_data: List[float], advantages_data: List[float],
                           save_name: Optional[str] = None) -> None:
        """Plot detailed reward and advantage analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Reward Analysis', fontsize=16, fontweight='bold')
        
        # Reward Distribution
        axes[0, 0].hist(rewards_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(rewards_data), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rewards_data):.3f}')
        axes[0, 0].set_title('Reward Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Reward Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Advantage Distribution
        axes[0, 1].hist(advantages_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(advantages_data), color='red', linestyle='--',
                          label=f'Mean: {np.mean(advantages_data):.3f}')
        axes[0, 1].set_title('Group Relative Advantages', fontweight='bold')
        axes[0, 1].set_xlabel('Advantage Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward vs Advantage Scatter
        axes[1, 0].scatter(rewards_data, advantages_data, alpha=0.6, c='purple')
        axes[1, 0].set_title('Rewards vs Advantages', fontweight='bold')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Advantage')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward Statistics Box Plot
        axes[1, 1].boxplot([rewards_data], labels=['Rewards'])
        axes[1, 1].set_title('Reward Statistics', fontweight='bold')
        axes[1, 1].set_ylabel('Reward Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Reward Stats:
Mean: {np.mean(rewards_data):.3f}
Std: {np.std(rewards_data):.3f}
Min: {np.min(rewards_data):.3f}
Max: {np.max(rewards_data):.3f}

Advantage Stats:
Mean: {np.mean(advantages_data):.3f}
Std: {np.std(advantages_data):.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self, baseline_results: Dict, final_results: Dict, 
                       model_name: str, environment: str, save_name: Optional[str] = None) -> None:
        """Plot before/after comparison as shown in the research article"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Training Results: {model_name} on {environment}', 
                    fontsize=16, fontweight='bold')
        
        # Accuracy Comparison Bar Chart
        categories = ['Baseline (SFT)', 'After GRPO']
        accuracies = [baseline_results.get('accuracy', 0) * 100, 
                     final_results.get('accuracy', 0) * 100]
        format_accuracies = [baseline_results.get('format_accuracy', 0) * 100,
                           final_results.get('format_accuracy', 0) * 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, accuracies, width, label='Task Accuracy', 
                           color='skyblue', edgecolor='black')
        bars2 = axes[0].bar(x + width/2, format_accuracies, width, label='Format Accuracy', 
                           color='lightcoral', edgecolor='black')
        
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Improvement Metrics
        accuracy_improvement = (final_results.get('accuracy', 0) - baseline_results.get('accuracy', 0)) * 100
        format_improvement = (final_results.get('format_accuracy', 0) - baseline_results.get('format_accuracy', 0)) * 100
        
        improvements = ['Task Accuracy', 'Format Accuracy']
        improvement_values = [accuracy_improvement, format_improvement]
        colors = ['green' if x >= 0 else 'red' for x in improvement_values]
        
        bars = axes[1].bar(improvements, improvement_values, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_title('GRPO Training Improvements', fontweight='bold')
        axes[1].set_ylabel('Improvement (percentage points)')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on improvement bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.5 if height >= 0 else -0.5),
                        f'{height:+.1f}pp', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict], environment: str,
                            save_name: Optional[str] = None) -> None:
        """Compare results across different model sizes"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Model Size Comparison on {environment}', fontsize=16, fontweight='bold')
        
        models = list(results_dict.keys())
        baseline_acc = [results_dict[m]['baseline']['accuracy'] * 100 for m in models]
        final_acc = [results_dict[m]['final']['accuracy'] * 100 for m in models]
        improvements = [f - b for f, b in zip(final_acc, baseline_acc)]
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, baseline_acc, width, label='Baseline (SFT)', 
                   color='lightblue', edgecolor='black')
        axes[0].bar(x + width/2, final_acc, width, label='After GRPO', 
                   color='lightgreen', edgecolor='black')
        
        axes[0].set_title('Accuracy by Model Size', fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('SmolLM-', '').replace('-Instruct', '') for m in models], 
                               rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 100)
        
        # Improvement comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = axes[1].bar(models, improvements, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_title('GRPO Improvements by Model Size', fontweight='bold')
        axes[1].set_ylabel('Improvement (percentage points)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{imp:+.1f}pp', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_training_report(self, results_file: str, output_name: Optional[str] = None) -> None:
        """Generate a comprehensive training report from saved results"""
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if output_name is None:
            output_name = Path(results_file).stem
        
        print(f"Generating training report for: {results_file}")
        print("=" * 60)
        
        # Generate appropriate plots based on results structure
        if 'training_history' in results:
            if 'losses' in results['training_history']:
                self.plot_grpo_training_curves(results['training_history'], output_name)
            else:
                self.plot_sft_training_curves(results['training_history'], output_name)
        
        if 'baseline' in results and 'final' in results:
            model_name = results.get('config', {}).get('model_name', 'Unknown Model')
            environment = results.get('config', {}).get('environment_name', 'Unknown Task')
            self.plot_comparison(results['baseline'], results['final'], 
                               model_name, environment, output_name)
        
        print(f"Training report saved to: {self.save_dir}")
    
    def save_grpo_training_curves(self, training_history: Dict[str, List], save_name: str) -> None:
        """Save GRPO training curves to file without displaying them"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Training Progress', fontsize=16, fontweight='bold')
        
        iterations = range(1, len(training_history.get('losses', [])) + 1)
        
        # Training Loss
        if 'losses' in training_history:
            axes[0, 0].plot(iterations, training_history['losses'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('PPO Training Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Reward Statistics
        if 'mean_rewards' in training_history and 'reward_stds' in training_history:
            mean_rewards = training_history['mean_rewards']
            reward_stds = training_history['reward_stds']
            
            axes[0, 1].plot(iterations, mean_rewards, 'g-', linewidth=2, marker='s', label='Mean Reward')
            axes[0, 1].fill_between(iterations, 
                                  np.array(mean_rewards) - np.array(reward_stds),
                                  np.array(mean_rewards) + np.array(reward_stds),
                                  alpha=0.3, color='green')
            axes[0, 1].set_title('Reward Distribution (Mean ± Std)', fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Accuracy Progress
        if 'accuracies' in training_history:
            axes[1, 0].plot(iterations, training_history['accuracies'], 'r-', linewidth=2, marker='^')
            axes[1, 0].set_title('Accuracy Over Training', fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Format vs Correctness Rewards
        if 'format_rewards' in training_history and 'correctness_rewards' in training_history:
            axes[1, 1].plot(iterations, training_history['format_rewards'], 'orange', 
                          linewidth=2, marker='d', label='Format Reward (15%)')
            axes[1, 1].plot(iterations, training_history['correctness_rewards'], 'purple', 
                          linewidth=2, marker='v', label='Correctness Reward (85%)')
            axes[1, 1].set_title('Reward Components', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Reward Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}_grpo_training.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
    
    def save_sft_training_curves(self, training_history: Dict[str, List], save_name: str) -> None:
        """Save SFT training curves to file without displaying them"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Supervised Fine-tuning Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(training_history.get('train_losses', [])) + 1)
        
        # Training and Validation Loss
        if 'train_losses' in training_history:
            axes[0].plot(epochs, training_history['train_losses'], 'b-', 
                        linewidth=2, marker='o', label='Training Loss')
            
        if 'val_losses' in training_history:
            axes[0].plot(epochs, training_history['val_losses'], 'r--', 
                        linewidth=2, marker='s', label='Validation Loss')
            
        axes[0].set_title('Training Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Accuracy Progress
        if 'train_accuracy' in training_history:
            axes[1].plot(epochs, training_history['train_accuracy'], 'g-', 
                        linewidth=2, marker='^', label='Training Accuracy')
            
        if 'val_accuracy' in training_history:
            axes[1].plot(epochs, training_history['val_accuracy'], 'orange', 
                        linewidth=2, marker='d', label='Validation Accuracy')
            
        axes[1].set_title('Accuracy Progress', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}_sft_training.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
    def save_all_plots(self, prefix: str = "training") -> None:
        """Save all generated plots with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_prefix = f"{prefix}_{timestamp}"
        print(f"All plots saved with prefix: {final_prefix}")
        
def load_and_visualize(results_path: str, plot_type: str = "auto") -> None:
    """Utility function to quickly load and visualize training results"""
    
    visualizer = TrainingVisualizer()
    
    if Path(results_path).is_file():
        visualizer.create_training_report(results_path)
    else:
        print(f"Results file not found: {results_path}")
        # List available results files
        results_dir = Path("results")
        if results_dir.exists():
            available_files = list(results_dir.glob("*.json"))
            if available_files:
                print("Available results files:")
                for file in available_files:
                    print(f"  - {file}")
            else:
                print("No results files found in results/ directory")