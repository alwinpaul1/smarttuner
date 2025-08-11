"""
Supervised Fine-tuning (SFT) trainer for warming up small language models
before GRPO training, as described in the article.
"""

import json
import asyncio
import openai
import backoff
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from reasoning_gym import create_dataset
import logging
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTMetricsCallback(TrainerCallback):
    """Custom callback to track SFT training metrics"""
    
    def __init__(self, sft_trainer):
        self.sft_trainer = sft_trainer
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None:
            # Track training loss
            if 'train_loss' in logs:
                self.sft_trainer.training_history['train_losses'].append(logs['train_loss'])
            
            # Track learning rate
            if 'learning_rate' in logs:
                self.sft_trainer.training_history['learning_rates'].append(logs['learning_rate'])
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after evaluation"""
        if logs is not None and 'eval_loss' in logs:
            self.sft_trainer.training_history['val_losses'].append(logs['eval_loss'])

@dataclass
class SFTConfig:
    """Configuration for SFT training"""
    model_name: str = "HuggingfaceTB/SmolLM-135M-Instruct"
    environment_name: str = "syllogism"
    num_datapoints: int = 200
    openai_model: str = "gpt-4o-mini"
    max_concurrent_requests: int = 50
    output_dir: str = "models/sft"
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    
    # Reproducibility and experimentation
    dataset_seed: int = 42
    eval_seed: int = 123
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                       "up_proj", "down_proj", "gate_proj"]

class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks with thinking format"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the conversation
        messages = [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["response"]}
        ]
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }

class SFTDataGenerator:
    """Generate SFT data using OpenAI API as described in the article"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.client = openai.AsyncClient()
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # System prompt with answer injection for SFT
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
Failing to follow the response format will result in a penalty.

You will also be provided the real answer. Your thinking should eventually result in producing the real answer."""
    
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    async def generate_response(self, item: Dict) -> Dict:
        """Generate a single response using OpenAI API"""
        async with self.semaphore:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user", 
                    "content": f"""
Question: {item['question']}
Metadata: {item['metadata']}
Answer: {item['answer']}
                    """
                }
            ]
            
            response = await self.client.chat.completions.create(
                messages=messages, 
                model=self.config.openai_model,
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                "question": item["question"],
                "metadata": item["metadata"],
                "answer": item["answer"],
                "response": response.choices[0].message.content,
                "system_prompt": self.system_prompt.split("You will also be provided")[0].strip()  # Remove the answer hint for training
            }
    
    async def generate_dataset(self) -> List[Dict]:
        """Generate complete SFT dataset"""
        logger.info(f"Generating {self.config.num_datapoints} examples for {self.config.environment_name}")
        
        # Create reasoning gym dataset
        dataloader = create_dataset(
            name=self.config.environment_name, 
            seed=self.config.dataset_seed,
            size=self.config.num_datapoints
        )
        
        # Generate responses concurrently
        responses = await asyncio.gather(*[
            self.generate_response(item) for item in dataloader
        ])
        
        # Save raw data
        os.makedirs("data", exist_ok=True)
        filename = f"data/sft_{self.config.environment_name}_{self.config.openai_model}.json"
        with open(filename, "w") as f:
            json.dump(responses, f, indent=4)
        
        logger.info(f"Saved SFT data to {filename}")
        return responses

class SFTTrainer:
    """Supervised Fine-tuning trainer"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        
        # Training metrics tracking
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        self._setup_model_and_tokenizer()
    
    def _setup_model_and_tokenizer(self):
        """Load and setup model with LoRA"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set pad token
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
    
    def train(self, sft_data: List[Dict] = None):
        """Train the model with supervised fine-tuning"""
        
        # Generate data if not provided
        if sft_data is None:
            logger.info("Generating SFT data...")
            generator = SFTDataGenerator(self.config)
            sft_data = asyncio.run(generator.generate_dataset())
        
        # Create dataset
        train_dataset = ReasoningDataset(sft_data, self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to=None,
            save_total_limit=2
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )
        
        # Create trainer with metrics callback
        metrics_callback = SFTMetricsCallback(self)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[metrics_callback]
        )
        
        # Train
        logger.info("Starting SFT training...")
        trainer.train()
        
        # Save training history
        self._save_training_history()
        
        # Save final model
        trainer.save_model()
        logger.info(f"SFT training completed. Model saved to {self.config.output_dir}")
        
        return self.model, self.tokenizer
    
    def evaluate_sft_model(self, test_size: int = 50) -> Dict[str, float]:
        """Evaluate SFT model performance"""
        logger.info("Evaluating SFT model...")
        
        from reasoning_gym import get_score_answer_fn
        import re
        
        # Create test dataset
        test_dataset = create_dataset(self.config.environment_name, seed=self.config.eval_seed, size=test_size)
        
        # System prompt for evaluation (without answer hint)
        eval_system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.

Do not generate new code. Do not write python code.

Very important - Remember again, your output format should be:
<think> reasoning process here </think>
<answer> answer here </answer>

Your response will be scored by extracting the substring between the <answer>...</answer> tags.
It is critical to follow the above format."""
        
        correct = 0
        format_correct = 0
        total = 0
        
        def extract_answer(response: str) -> str:
            answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            return answer.group(1).strip() if answer else ""
        
        def has_proper_format(response: str) -> bool:
            has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
            has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
            return has_think and has_answer
        
        with torch.no_grad():
            for item in test_dataset:
                question = item["question"]
                validation_object = item["metadata"]["source_dataset"]
                score_fn = get_score_answer_fn(validation_object)
                
                # Create messages
                messages = [
                    {"role": "system", "content": eval_system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Generate response
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                )
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                generated = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode response
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
                
                # Evaluate
                extracted_answer = extract_answer(response)
                is_correct = score_fn(extracted_answer, validation_object)
                has_format = has_proper_format(response)
                
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
        
        logger.info(f"SFT Evaluation results: {results}")
        return results
    
    def _save_training_history(self) -> None:
        """Save training history to JSON file"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        history_file = os.path.join(self.config.output_dir, "training_history.json")
        
        with open(history_file, 'w') as f:
            # Convert to serializable format
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                else:
                    serializable_history[key] = value
            
            json.dump(serializable_history, f, indent=4)
        logger.info(f"SFT training history saved to: {history_file}")
    
    def show_training_plots(self, save_name: str = None) -> None:
        """Show SFT training plots"""
        try:
            from .visualizer import TrainingVisualizer
            
            visualizer = TrainingVisualizer()
            visualizer.plot_sft_training_curves(self.training_history, save_name)
            
        except ImportError:
            logger.warning("Visualization module not available")
        except Exception as e:
            logger.warning(f"Error showing training plots: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive SFT training summary"""
        if not self.training_history['train_losses']:
            return {"message": "No training completed yet"}
            
        summary = {
            "total_epochs": len(self.training_history['train_losses']) if self.training_history['train_losses'] else 0,
            "final_train_loss": self.training_history['train_losses'][-1] if self.training_history['train_losses'] else None,
            "best_train_loss": min(self.training_history['train_losses']) if self.training_history['train_losses'] else None,
            "final_val_loss": self.training_history['val_losses'][-1] if self.training_history['val_losses'] else None,
            "best_val_loss": min(self.training_history['val_losses']) if self.training_history['val_losses'] else None,
            "training_history": self.training_history
        }
        
        return summary

def run_sft_training(config: SFTConfig):
    """Run complete SFT training pipeline"""
    trainer = SFTTrainer(config)
    model, tokenizer = trainer.train()
    
    # Evaluate
    results = trainer.evaluate_sft_model()
    
    return model, tokenizer, results