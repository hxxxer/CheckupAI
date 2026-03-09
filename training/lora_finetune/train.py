"""
LoRA Fine-tuning Script for Medical LLM

This script fine-tunes Qwen3-8B model using LoRA for medical report analysis
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch


def prepare_dataset(data_path):
    """
    Prepare training dataset
    
    Args:
        data_path: Path to training data
        
    Returns:
        Dataset object
    """
    dataset = load_dataset('json', data_files=data_path)
    return dataset


def tokenize_function(examples, tokenizer, max_length=2048):
    """
    Tokenize examples
    
    Args:
        examples: Dataset examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )


def main():
    # Model configuration
    base_model = 'Qwen/Qwen-7B-Chat'  # Replace with actual Qwen3-8B path
    output_dir = './lora_checkpoint'
    data_path = '../../data/finetune_data/train.json'
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # LoRA rank
        lora_alpha=32,          # LoRA alpha
        lora_dropout=0.1,       # Dropout
        target_modules=['c_attn', 'c_proj'],  # Target modules (adjust for Qwen)
        bias='none'
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    dataset = prepare_dataset(data_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=100,
        optim='adamw_torch'
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")


if __name__ == '__main__':
    main()
