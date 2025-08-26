"""
GRPO (Group Relative Policy Optimization) Trainer for Small Language Models
Following the implementation described in the TowardsDataScience article.
"""

import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from reasoning_gym import create_dataset, get_score_answer_fn
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class GRPOConfig:
    """Configuration for GRPO training"""
    model_name: str = "HuggingfaceTB/SmolLM-135M-Instruct"
    environment_name: str = "syllogism"  # or "propositional_logic"
    max_new_tokens: int = 300
    exploration_batchsize: int = 8
    G: int = 6  # number of responses per group
    temperature: float = 0.7
    batch_size: int = 16
    gradient_accumulation_steps: int = 12
    learning_rate: float = 1e-6
    top_p: float = 0.95
    buffer_size: int = 500
    num_epochs: int = 3
    ppo_clip_ratio: float = 0.2
    correctness_reward_weight: float = 0.85
    format_reward_weight: float = 0.15
    save_every: int = 100
    
    # Reproducibility and experimentation
    dataset_seed: int = 42
    eval_seed: int = 123
    
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                       "up_proj", "down_proj", "gate_proj"]

class GRPOTrainer:
    """GRPO Trainer implementing the algorithm from the article"""
    
    def __init__(self, config: GRPOConfig, output_dir: str = "models"):
        self.config = config
        self.output_dir = output_dir
        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
        
        # Training metrics tracking
        self.training_history = {
            'losses': [],
            'mean_rewards': [],
            'reward_stds': [],
            'accuracies': [],
            'format_rewards': [],
            'correctness_rewards': []
        }
        
        # System prompt as defined in the article
        self.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.

Do not generate new code. Do not write python code.

You may also be given examples by the user telling you the expected response format.
Follow the format of the examples, but solve the specific problem asked by the user, not the examples.

Very important - Remember again, your output format should be:
<think> reasoning process here </think>
<answer> answer here </answer>

Your response will be scored by extracting the substring between the <answer>...</answer> tags.
It is critical to follow the above format.
Failing to follow the response format will result in a penalty."""
        
        self._setup_model_and_tokenizer()
        self._setup_environment()
        self.memory_buffer = []
        
    def _setup_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA configuration"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
    def _setup_environment(self):
        """Setup reasoning environment"""
        logger.info(f"Setting up environment: {self.config.environment_name}")
        # We'll create dataset on-demand during training
        
    def extract_answer(self, response: str) -> str:
        """Extract answer from response using regex as shown in article"""
        answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer is not None:
            return answer.group(1).strip()
        else:
            return ""
            
    def calculate_format_reward(self, response: str) -> float:
        """Calculate format reward for proper tag usage"""
        has_think_tags = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
        has_answer_tags = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
        
        if has_think_tags and has_answer_tags:
            return 1.0
        elif has_answer_tags:
            return 0.5
        else:
            return 0.0
    
    def calculate_log_probs(self, model, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Calculate log probabilities of generated tokens"""
        with torch.no_grad() if hasattr(self, '_collecting_experience') and self._collecting_experience else torch.enable_grad():
            logits = model(input_ids=input_ids).logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = response_mask[..., 1:].contiguous()
            
            # Calculate log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            selected_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Apply mask to only consider AI-generated tokens
            selected_log_probs = selected_log_probs * shift_mask
            
            return selected_log_probs
    
    def calculate_ppo_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Calculate PPO clipped surrogate loss as described in article"""
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * advantages
        
        # Take minimum and apply mask
        policy_loss = -torch.min(surr1, surr2) * response_mask
        
        # Average over valid tokens
        return policy_loss.sum() / response_mask.sum()
    
    def experience_collection(self, dataset_size: int = 100) -> None:
        """Experience collection phase as described in article"""
        logger.info("Starting experience collection phase...")
        self._collecting_experience = True
        
        # Create dataset
        dataset = create_dataset(self.config.environment_name, seed=self.config.dataset_seed, size=dataset_size)
        
        # Metrics for this experience collection round
        all_rewards = []
        all_correctness_rewards = []
        all_format_rewards = []
        
        # Convert dataset to list for batching
        dataset_items = list(dataset)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(0, len(dataset_items), self.config.exploration_batchsize)):
                batch = dataset_items[batch_idx:batch_idx + self.config.exploration_batchsize]
                
                questions = [d["question"] for d in batch]
                dataset_names = [d["metadata"]["source_dataset"] for d in batch]
                score_fns = [get_score_answer_fn(name) for name in dataset_names]
                
                # Process each question
                for q_idx, (question, score_fn) in enumerate(zip(questions, score_fns)):
                    # Create messages
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ]
                    
                    # Tokenize
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_tensors="pt",
                        add_generation_prompt=True
                    )
                    
                    input_ids = input_ids.to(self.model.device)
                    attention_mask = torch.ones_like(input_ids).to(self.model.device)
                    input_length = input_ids.shape[1]
                    
                    # Generate multiple responses
                    generated_responses = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        top_p=self.config.top_p,
                        num_return_sequences=self.config.G,
                        temperature=self.config.temperature,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    
                    # Decode responses
                    full_sequences = generated_responses.sequences
                    responses = []
                    rewards = []
                    
                    for seq in full_sequences:
                        # Create response mask (1 for generated tokens, 0 for input)
                        response_mask = torch.zeros_like(seq)
                        response_mask[input_length:] = 1
                        
                        # Decode response
                        full_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                        response_text = self.tokenizer.decode(seq[input_length:], skip_special_tokens=True)
                        
                        # Calculate rewards
                        extracted_answer = self.extract_answer(response_text)
                        correctness_reward = score_fn(extracted_answer, batch[q_idx])
                        format_reward = self.calculate_format_reward(response_text)
                        
                        total_reward = (correctness_reward * self.config.correctness_reward_weight + 
                                      format_reward * self.config.format_reward_weight)
                        
                        responses.append(response_text)
                        rewards.append(total_reward)
                        
                        # Track individual reward components for analysis
                        all_rewards.append(total_reward)
                        all_correctness_rewards.append(correctness_reward)
                        all_format_rewards.append(format_reward)
                    
                    # Calculate advantages (group-relative)
                    rewards = np.array(rewards)
                    advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
                    
                    # Calculate old log probabilities
                    for i, (seq, advantage) in enumerate(zip(full_sequences, advantages)):
                        response_mask = torch.zeros_like(seq)
                        response_mask[input_length:] = 1
                        
                        old_log_probs = self.calculate_log_probs(self.model, seq.unsqueeze(0), response_mask.unsqueeze(0))
                        
                        # Store in memory buffer
                        self.memory_buffer.append({
                            "full_response": seq,
                            "response_mask": response_mask,
                            "old_log_probs": old_log_probs.squeeze(0),
                            "advantages": torch.tensor(advantage, dtype=torch.float32)
                        })
                
                if len(self.memory_buffer) >= self.config.buffer_size:
                    break
        
        self._collecting_experience = False
        
        # Store metrics for this experience collection round
        if all_rewards:
            self.training_history['mean_rewards'].append(np.mean(all_rewards))
            self.training_history['reward_stds'].append(np.std(all_rewards))
            self.training_history['correctness_rewards'].append(np.mean(all_correctness_rewards))
            self.training_history['format_rewards'].append(np.mean(all_format_rewards))
        
        logger.info(f"Collected {len(self.memory_buffer)} experiences")
        logger.info(f"Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    
    def training_phase(self) -> float:
        """Training phase using collected experiences"""
        logger.info("Starting training phase...")
        
        if len(self.memory_buffer) == 0:
            logger.warning("No experiences in buffer!")
            return 0.0
        
        # Shuffle buffer
        np.random.shuffle(self.memory_buffer)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_start in range(0, len(self.memory_buffer), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(self.memory_buffer))
            batch = self.memory_buffer[batch_start:batch_end]
            
            # Prepare batch
            full_responses = torch.stack([item["full_response"] for item in batch])
            response_masks = torch.stack([item["response_mask"] for item in batch])
            old_log_probs = torch.stack([item["old_log_probs"] for item in batch])
            advantages = torch.stack([item["advantages"] for item in batch])
            
            # Move to device
            full_responses = full_responses.to(self.model.device)
            response_masks = response_masks.to(self.model.device)
            old_log_probs = old_log_probs.to(self.model.device)
            advantages = advantages.to(self.model.device)
            
            # Calculate new log probabilities
            new_log_probs = self.calculate_log_probs(self.model, full_responses, response_masks)
            
            # Calculate PPO loss
            loss = self.calculate_ppo_loss(new_log_probs, old_log_probs, advantages, response_masks)
            
            # Backward pass
            self.accelerator.backward(loss)
            
            if (num_batches + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Average training loss: {avg_loss:.4f}")
        
        # Store training loss
        self.training_history['losses'].append(avg_loss)
        
        # Clear buffer after training
        self.memory_buffer.clear()
        
        return avg_loss
    
    def train(self, num_iterations: int = 10, dataset_size_per_iteration: int = 100, 
              show_plots: bool = False, save_plots: bool = False):
        """Main GRPO training loop with optional visualization"""
        logger.info("Starting GRPO training...")
        
        for iteration in range(num_iterations):
            logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # Experience collection phase
            self.experience_collection(dataset_size_per_iteration)
            
            # Training phase
            avg_loss = self.training_phase()
            
            # Evaluate accuracy periodically
            if (iteration + 1) % max(1, num_iterations // 5) == 0:  # Evaluate 5 times during training
                eval_results = self.evaluate(test_size=50)  # Quick evaluation
                self.training_history['accuracies'].append(eval_results['accuracy'] * 100)
                logger.info(f"Current accuracy: {eval_results['accuracy']:.3f}")
            
            # Show real-time plots if requested
            if show_plots and len(self.training_history['losses']) > 1:
                self._show_live_plots()
            
            # Save model periodically
            if (iteration + 1) % (self.config.save_every // 10) == 0:
                save_path = os.path.join(self.output_dir, f"checkpoint_iter_{iteration + 1}")
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                logger.info(f"Saved checkpoint at iteration {iteration + 1}")
                
                # Save training history
                self._save_training_history(save_path)
        
        # Save final model
        final_path = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Training completed. Final model saved to {final_path}")
    
    def evaluate(self, test_size: int = 100) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        dataset = create_dataset(self.config.environment_name, seed=self.config.eval_seed, size=test_size)
        
        correct = 0
        total = 0
        format_correct = 0
        
        with torch.no_grad():
            for item in tqdm(dataset):
                question = item["question"]
                dataset_name = item["metadata"]["source_dataset"]
                score_fn = get_score_answer_fn(dataset_name)
                
                # Create messages
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Generate response
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                )
                
                input_ids = input_ids.to(self.model.device)
                attention_mask = torch.ones_like(input_ids).to(self.model.device)
                
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,  # Greedy for evaluation
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode response
                input_length = input_ids.shape[1]
                response = self.tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
                
                # Extract answer and check
                extracted_answer = self.extract_answer(response)
                is_correct = score_fn(extracted_answer, item)
                has_format = self.calculate_format_reward(response) > 0.5
                
                if is_correct:
                    correct += 1
                if has_format:
                    format_correct += 1
                total += 1
        
        accuracy = correct / total
        format_accuracy = format_correct / total
        
        results = {
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "correct": correct,
            "total": total
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def _show_live_plots(self) -> None:
        """Show live training plots during training"""
        try:
            from .visualizer import TrainingVisualizer
            
            visualizer = TrainingVisualizer()
            visualizer.plot_grpo_training_curves(self.training_history, save_name=None)
            
        except ImportError:
            logger.warning("Visualization module not available for live plots")
        except Exception as e:
            logger.warning(f"Error showing live plots: {e}")
    
    def _save_training_history(self, save_path: str) -> None:
        """Save training history to JSON file"""
        history_file = os.path.join(save_path, "training_history.json")
        with open(history_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                else:
                    serializable_history[key] = value
            
            json.dump(serializable_history, f, indent=4)
        logger.info(f"Training history saved to: {history_file}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a comprehensive training summary with metrics"""
        if not self.training_history['losses']:
            return {"message": "No training completed yet"}
            
        summary = {
            "total_iterations": len(self.training_history['losses']),
            "final_loss": self.training_history['losses'][-1] if self.training_history['losses'] else None,
            "best_loss": min(self.training_history['losses']) if self.training_history['losses'] else None,
            "final_accuracy": self.training_history['accuracies'][-1] if self.training_history['accuracies'] else None,
            "best_accuracy": max(self.training_history['accuracies']) if self.training_history['accuracies'] else None,
            "mean_final_reward": self.training_history['mean_rewards'][-1] if self.training_history['mean_rewards'] else None,
            "training_history": self.training_history
        }
        
        return summary