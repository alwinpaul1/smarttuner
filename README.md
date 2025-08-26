# SmartTuner: GRPO Training for Small Language Models

A complete implementation of **Group Relative Policy Optimization (GRPO)** for training small language models to reason.

## Overview

This project implements **RLVR (Reinforcement Learning with Verifiable Rewards)** using the GRPO algorithm to teach small language models to generate reasoning chains and solve logical tasks using Qwen2.5-0.5B-Instruct.

### Key Features

- ✅ **Complete GRPO Implementation**: Experience collection and training phases
- ✅ **Supervised Fine-tuning (SFT)**: Warmup training with LoRA adapters  
- ✅ **Verifiable Rewards**: Objective reward calculation for logical tasks
- ✅ **Multiple Environments**: Syllogism and propositional logic tasks
- ✅ **Small Model Support**: Optimized for Qwen2.5-0.5B-Instruct model
- ✅ **PPO Clipped Loss**: Stable training with importance sampling
- ✅ **Configurable System Prompts**: Easy experimentation via YAML configuration
- ✅ **Robust Error Handling**: Comprehensive validation and informative error messages
- ✅ **Advanced Visualization**: Dedicated save/show functions with memory management

## Architecture

The training pipeline follows a two-phase approach:

### Phase 1: Supervised Fine-tuning (SFT)
- Generate reasoning data using GPT-4o-mini
- Train Qwen2.5-0.5B-Instruct to generate `<think>...</think>` and `<answer>...</answer>` format
- Use LoRA adapters for parameter-efficient training
- Achieve strong baseline accuracy

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

# Create .env file with OpenAI API key (required for SFT data generation)
cp .env.example .env
# Then edit .env and add your actual API key
```

## Quick Start

### Complete Training Pipeline

Run the full SFT → GRPO pipeline:

```bash
# Train on syllogism task - with uv
uv run python scripts/full_pipeline.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --environment syllogism \
    --sft_datapoints 200 \
    --grpo_iterations 10

# Skip SFT and use existing model for GRPO
uv run python scripts/full_pipeline.py \
    --skip_sft \
    --sft_model_path models/sft/syllogism \
    --environment syllogism \
    --grpo_iterations 10

# Train on propositional logic (harder) - with uv
uv run python scripts/full_pipeline.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --environment propositional_logic \
    --sft_datapoints 200 \
    --grpo_iterations 10

# Or use regular python if you prefer
python scripts/full_pipeline.py --base_model "Qwen/Qwen2.5-0.5B-Instruct" --environment syllogism
```

### Step-by-Step Training

#### 1. Supervised Fine-tuning

```bash
# With uv (recommended)
uv run python scripts/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --environment syllogism \
    --num_datapoints 200 \
    --num_epochs 3 \
    --output_dir models/sft

# Or with regular python
python scripts/train_sft.py --model_name "Qwen/Qwen2.5-0.5B-Instruct" --environment syllogism
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
uv run python scripts/visualize_results.py --file results/grpo_syllogism_Qwen2.5-0.5B-Instruct.json

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

### System Prompt Customization

System prompts can be customized via the `configs/small_models.yaml` file:

```yaml
environments:
  syllogism:
    system_prompt: |
      Your custom system prompt here...
      The assistant first thinks about the reasoning process...
  
  propositional_logic:
    system_prompt: |
      Your custom system prompt for propositional logic...
```

This allows easy experimentation with different prompting strategies without modifying Python code.

### Model Supported

| Model | Parameters | Expected Performance |
|-------|------------|---------------------|
| Qwen2.5-0.5B-Instruct | 500M | High performance on reasoning tasks |

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
- **Expected Improvement**: Significant improvement with GRPO

## Results

| Model | Task | Expected Improvement |
|-------|------|---------------------|
| Qwen2.5-0.5B-Instruct | Syllogism | Significant accuracy gains with GRPO |
| Qwen2.5-0.5B-Instruct | Propositional Logic | Strong reasoning performance |

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
- 📈 **Improvement Metrics**: Percentage point improvements
- 🎯 **Task Performance**: Analysis across different reasoning tasks

### Real-time Monitoring

Enable live plotting during training to monitor progress:
```bash
uv run python scripts/train_grpo.py --show_plots --environment syllogism
```

### Advanced Visualization Features

- **Memory Efficient**: Dedicated save functions that automatically close plots to prevent memory leaks
- **Flexible Output**: Separate show and save functionality for different use cases
- **High Quality**: All plots saved as 300 DPI PNG files perfect for publications
- **Automatic Organization**: Results organized by timestamp and model name

```bash
# Save plots without displaying (great for remote servers)
uv run python scripts/train_grpo.py --save_plots --environment syllogism

# Display plots interactively during training
uv run python scripts/train_grpo.py --show_plots --environment syllogism

# Both save and display
uv run python scripts/train_grpo.py --save_plots --show_plots --environment syllogism
```

All plots are saved as high-resolution PNG files in the `plots/` directory, perfect for research papers and presentations.

## Error Handling & Troubleshooting

SmartTuner includes comprehensive error handling for common issues:

### Model Loading Issues
- **Automatic validation**: Checks if model names are correct before loading
- **Informative errors**: Clear messages for network issues or incorrect model names
- **Fallback handling**: Graceful degradation when models are unavailable

### API Key Validation
- **Pre-flight checks**: Validates OpenAI API key before starting data generation
- **Clear instructions**: Step-by-step guidance for setting up API keys
- **Multiple setup methods**: Support for both .env files and environment variables

### Common Error Solutions

**"Failed to load model"**: Check your internet connection and verify the model name is correct.

**"OpenAI API key not found"**: 
```bash
# Method 1: Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Method 2: Export environment variable  
export OPENAI_API_KEY=your_key_here
```

**"Model not found at path"**: Use `--sft_model_path` to specify the exact location of your SFT model.

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

1. **SFT warmup is essential** - Models need supervised fine-tuning before RL
2. **Diverse responses are crucial** - High temperature/top_p for good advantages
3. **Group relative advantages work better** - No need for separate value network  
4. **Low learning rates essential** - 1e-6 to 1e-7 for stability
5. **Format rewards help convergence** - Teaching proper output structure
6. **Separate evaluation seeds prevent data leakage** - Different seeds ensure unbiased evaluation