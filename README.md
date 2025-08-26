# SmartTuner: GRPO Training for Small Language Models

A complete implementation of **Group Relative Policy Optimization (GRPO)** for training small language models to reason.

## Overview

This project implements **RLVR (Reinforcement Learning with Verifiable Rewards)** using the GRPO algorithm to teach small language models (135M-600M parameters) to generate reasoning chains and solve logical tasks.

### Key Features

- ✅ **Complete GRPO Implementation**: Experience collection and training phases
- ✅ **Supervised Fine-tuning (SFT)**: Warmup training with LoRA adapters  
- ✅ **Verifiable Rewards**: Objective reward calculation for logical tasks
- ✅ **Multiple Environments**: Syllogism and propositional logic tasks
- ✅ **Small Model Support**: Optimized for 135M-600M parameter models
- ✅ **PPO Clipped Loss**: Stable training with importance sampling

## Architecture

The training pipeline follows a two-phase approach:

### Phase 1: Supervised Fine-tuning (SFT)
- Generate reasoning data using GPT-4o-mini
- Train small models to generate `<think>...</think>` and `<answer>...</answer>` format
- Use LoRA adapters for parameter-efficient training
- Achieve ~46% baseline accuracy

### Phase 2: GRPO Reinforcement Learning
- **Experience Collection**: Generate multiple responses per question
- **Group Relative Advantages**: Calculate relative goodness within response groups
- **PPO Training**: Update policy using clipped surrogate loss
- **Verifiable Rewards**: Objective scoring using environment feedback

## Installation

```bash
# Clone repository
git clone <repository-url>
cd smarttuner

# Create virtual environment (required)
uv venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies with uv (faster)
uv pip install -r requirements.txt

# Or use pip if you prefer
pip install -r requirements.txt
```

## Quick Start

### Complete Training Pipeline

Run the full SFT → GRPO pipeline:

```bash
# Train on syllogism task (easiest) - with uv
uv run python scripts/full_pipeline.py \
    --base_model "HuggingfaceTB/SmolLM-135M-Instruct" \
    --environment syllogism \
    --sft_datapoints 200 \
    --grpo_iterations 10

# Train on propositional logic (harder) - with uv
uv run python scripts/full_pipeline.py \
    --base_model "HuggingfaceTB/SmolLM-360M-Instruct" \
    --environment propositional_logic \
    --sft_datapoints 200 \
    --grpo_iterations 10

# Or use regular python if you prefer
python scripts/full_pipeline.py --environment syllogism
```

### Step-by-Step Training

#### 1. Supervised Fine-tuning

```bash
# With uv (recommended)
uv run python scripts/train_sft.py \
    --model_name "HuggingfaceTB/SmolLM-135M-Instruct" \
    --environment syllogism \
    --num_datapoints 200 \
    --num_epochs 3 \
    --output_dir models/sft

# Or with regular python
python scripts/train_sft.py --environment syllogism
```

#### 2. GRPO Training

```bash
# With uv (recommended)
uv run python scripts/train_grpo.py \
    --model_name models/sft \
    --environment syllogism \
    --num_iterations 10 \
    --dataset_size_per_iteration 100 \
    --output_dir models/grpo

# Or with regular python
python scripts/train_grpo.py --model_name models/sft
```

### Training with Visualization

```bash
# Show real-time plots during GRPO training
uv run python scripts/train_grpo.py \
    --environment syllogism \
    --show_plots

# Save training plots automatically
uv run python scripts/train_sft.py \
    --environment syllogism \
    --save_plots

# Both real-time and saved plots
uv run python scripts/full_pipeline.py \
    --environment syllogism \
    --show_plots --save_plots
```

### Analyze Training Results

```bash
# Visualize specific result file
uv run python scripts/visualize_results.py --file results/grpo_syllogism_SmolLM-135M-Instruct.json

# Interactive mode - select from available results
uv run python scripts/visualize_results.py

# Analyze all result files
uv run python scripts/visualize_results.py --all

# Compare multiple models
uv run python scripts/visualize_results.py --compare results/sft_*.json --environment syllogism
```

### Seed Configuration for Experimentation

```bash
# Use different seeds for data variations - with uv
uv run python scripts/train_grpo.py \
    --dataset_seed 123 \
    --eval_seed 456 \
    --environment syllogism

# For reproducible results, use default seeds (42 for train, 123 for eval)
uv run python scripts/full_pipeline.py --environment syllogism

# Or with regular python
python scripts/train_grpo.py --dataset_seed 123 --eval_seed 456
```

## Configuration

### Models Supported

| Model | Parameters | Expected Performance |
|-------|------------|---------------------|
| SmolLM-135M-Instruct | 135M | 60% on syllogism |
| SmolLM-360M-Instruct | 360M | ~70% on syllogism |
| Qwen2.5-0.5B-Instruct | 600M | 81% on syllogism |

### Hyperparameters

**GRPO Configuration:**
```yaml
max_new_tokens: 300          # reasoning + answer budget
exploration_batchsize: 8     # questions per rollout batch  
G: 6                        # responses per group
temperature: 0.7            # generation diversity
batch_size: 16              # training minibatch size
gradient_accumulation_steps: 12
learning_rate: 1e-6         # keep low for stability
top_p: 0.95                 # nucleus sampling
buffer_size: 500            # experience buffer
dataset_seed: 42            # seed for training data generation
eval_seed: 123              # seed for evaluation (different for unbiased eval)
```

**LoRA Configuration:**
```yaml
r: 32
lora_alpha: 64
lora_dropout: 0
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", 
                "up_proj", "down_proj", "gate_proj"]
```

## Environments

### Syllogism Task
- **Description**: Logical puzzles with two premises and a conclusion
- **Format**: YES/NO classification 
- **Difficulty**: Easier reasoning task
- **Expected Improvement**: +20% accuracy with GRPO

### Propositional Logic  
- **Description**: Symbolic reasoning with generated conclusions
- **Format**: Generate correct logical conclusion
- **Difficulty**: Harder reasoning task
- **Expected Improvement**: Varies by model size

## Results (from article)

| Model | Task | Baseline | After GRPO | Improvement |
|-------|------|----------|------------|-------------|
| SmolLM-135M | Syllogism | 46% (SFT) | 60% | +14% |
| SmolLM-360M | Syllogism | - | ~70% | ~+20% |
| Qwen2.5-0.5B | Syllogism | - | 81% | ~+20% |

## Visualization & Monitoring

SmartTuner provides comprehensive visualization for training analysis:

### Available Plots

**GRPO Training Visualization:**
- 📈 **PPO Loss Curves**: Training loss over iterations
- 🎯 **Reward Statistics**: Mean rewards with standard deviation bands
- 📊 **Accuracy Progress**: Task accuracy improvement during training  
- ⚖️ **Reward Components**: Format vs correctness reward breakdown
- 📉 **Reward Distribution Analysis**: Histograms and advantage analysis

**SFT Training Visualization:**
- 📈 **Loss Curves**: Training and validation loss over epochs
- 🎯 **Accuracy Progress**: Training and validation accuracy curves
- 📊 **Learning Rate Schedule**: Learning rate changes during training

**Comparison Analysis:**
- 📊 **Before/After Comparison**: Baseline vs final performance
- 🏆 **Model Size Comparison**: Performance across different model sizes
- 📈 **Improvement Metrics**: Percentage point improvements

### Real-time Monitoring

Enable live plotting during training to monitor progress:
```bash
uv run python scripts/train_grpo.py --show_plots --environment syllogism
```

All plots are saved as high-resolution PNG files in the `plots/` directory, perfect for research papers and presentations.

## Implementation Details

### GRPO Algorithm

The implementation follows the GRPO algorithm:

1. **Experience Collection Phase:**
   - Sample questions from reasoning-gym
   - Generate G responses per question
   - Calculate group-relative advantages
   - Store experiences with old log-probabilities

2. **Training Phase:**
   - Sample minibatches from experience buffer
   - Compute new log-probabilities
   - Calculate PPO clipped surrogate loss
   - Update policy parameters

### Reward System

**Total Reward = Correctness (0.85) + Format (0.15)**

- **Correctness Reward**: Binary feedback from environment verifier
- **Format Reward**: Proper `<think>` and `<answer>` tag usage

### Key Features

- **Group Relative Advantages**: No need for separate value network
- **Clipped PPO Loss**: Stable policy updates with trust region
- **Diverse Response Generation**: High temperature/top_p for exploration
- **LoRA Fine-tuning**: Memory-efficient parameter updates

## Directory Structure

```
smarttuner/
├── src/
│   ├── grpo_trainer.py      # Main GRPO implementation
│   └── sft_trainer.py       # Supervised fine-tuning
├── scripts/
│   ├── train_sft.py         # SFT training script
│   ├── train_grpo.py        # GRPO training script
│   └── full_pipeline.py     # Complete pipeline
├── configs/
│   └── small_models.yaml    # Model configurations
└── requirements.txt

Note: data/, models/, and results/ folders are created automatically by training scripts
```

## Key Insights

1. **Small models need SFT warmup** - Cannot rely on pure RL from scratch
2. **Diverse responses are crucial** - High temperature/top_p for good advantages
3. **Group relative advantages work better** - No need for separate value network  
4. **Low learning rates essential** - 1e-6 to 1e-7 for stability
5. **Format rewards help convergence** - Teaching proper output structure
6. **Separate evaluation seeds prevent data leakage** - Different seeds ensure unbiased evaluation