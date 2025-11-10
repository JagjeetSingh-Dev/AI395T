from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    """Load RFT model."""
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, str(model_path)).to(llm.device)
    llm.model.eval()
    
    # Override format_prompt to NOT use chat template (RFT was trained without it)
    def format_prompt_no_chat(question: str) -> str:
        return question
    
    llm.format_prompt = format_prompt_no_chat
    
    # Override generate() for single generation (used by grader)
    def generate_rft(prompt: str) -> str:
        import torch
        inputs = llm.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(llm.device)
        
        with torch.no_grad():
            outputs = llm.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,  # Increased for RFT reasoning
                eos_token_id=llm.tokenizer.eos_token_id,
                pad_token_id=llm.tokenizer.eos_token_id,
                do_sample=False
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        return llm.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    llm.generate = generate_rft
    
    # Override batched_generate to use more tokens for reasoning + answers
    def batched_generate_rft(prompts, num_return_sequences=None, temperature=0):
        # RFT generates reasoning + answer, needs more tokens
        # Some of the code has been generated with help of Gemini
        import torch
        from tqdm import tqdm
        
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(range(0, len(prompts), micro_batch_size), 
                          desc=f"LLM Running on Micro Batches {micro_batch_size}"):
                batch_prompts = prompts[idx : idx + micro_batch_size]
                batch_results = batched_generate_rft(batch_prompts, num_return_sequences, temperature)
                results.extend(batch_results)
            return results
        
        original_padding_side = llm.tokenizer.padding_side
        llm.tokenizer.padding_side = "left"
        
        try:
            inputs = llm.tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(llm.device)
            
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": 150,  # Increased for RFT reasoning + answers
                "eos_token_id": llm.tokenizer.eos_token_id,
                "pad_token_id": llm.tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                })
            else:
                generation_kwargs["do_sample"] = False
                
            if num_return_sequences is not None:
                generation_kwargs["num_return_sequences"] = num_return_sequences
            
            with torch.no_grad():
                outputs = llm.model.generate(**generation_kwargs)
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            
            decoded_outputs = llm.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            if num_return_sequences is None:
                return decoded_outputs
            else:
                result = []
                for i in range(len(prompts)):
                    start_idx = i * num_return_sequences
                    end_idx = start_idx + num_return_sequences
                    result.append(decoded_outputs[start_idx:end_idx])
                return result
        finally:
            llm.tokenizer.padding_side = original_padding_side
    
    llm.batched_generate = batched_generate_rft

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    rft_data_path: str = "data/rft.json",
    **kwargs,
):
    """Train RFT model on reasoning chains."""
    import os
    import json
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from pathlib import Path
    
    os.environ["WANDB_DISABLED"] = "true"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RFT TRAINING")
    print("=" * 60)
    
    # Load base model
    llm = BaseLLM()
    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    
    # LoRA config - increased capacity for reasoning
    lora_config = LoraConfig(
        r=16,  # Increased rank for better reasoning capacity
        lora_alpha=64,  # 4x rank
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05  # Lower dropout for better learning
    )
    
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()
    
    # Load RFT dataset
    print(f"\nLoading: {rft_data_path}")
    with open(rft_data_path, 'r') as f:
        rft_data = json.load(f)
    print(f"✓ {len(rft_data)} examples\n")
    
    # Custom tokenization for RFT
    class RFTDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            question, answer, reasoning = self.data[idx]
            
            # Combine question + reasoning + EOS
            full_text = f"{question} {reasoning}{self.tokenizer.eos_token}"
            
            # Tokenize with appropriate length for reasoning
            encoded = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=200,  # Balanced for reasoning + answer
                return_tensors=None
            )
            
            # Get question length for masking
            question_encoded = self.tokenizer(question, return_tensors=None)
            question_len = len(question_encoded["input_ids"])
            
            # Create labels: mask question, keep reasoning
            labels = [-100] * question_len + encoded["input_ids"][question_len:]
            
            # Mask padding
            for i in range(len(labels)):
                if encoded["attention_mask"][i] == 0:
                    labels[i] = -100
            
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels
            }
    
    train_dataset = RFTDataset(llm.tokenizer, rft_data)
    
    # Training args - optimized for reasoning chains
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=5,  # More epochs for reasoning
        per_device_train_batch_size=24,  # Balanced batch size
        learning_rate=5e-4,  # Higher learning rate like SFT
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1,
    )
    
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=llm.tokenizer,
    )
    
    print("Training RFT model...")
    print(f"  • Dataset size: {len(train_dataset)}")
    print(f"  • Epochs: {training_args.num_train_epochs}")
    print(f"  • Batch size: {training_args.per_device_train_batch_size}")
    print(f"  • Learning rate: {training_args.learning_rate}")
    print(f"  • LoRA rank: {lora_config.r}")
    print()
    
    trainer.train()
    
    # Save the final model
    trainer.save_model(str(output_path))
    llm.tokenizer.save_pretrained(str(output_path))
    print(f"\n✅ Model saved to {output_path}")
    
    # Test on validation set
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    test_model(str(output_path))


def test_model(ckpt_path: str):
    """Test RFT model."""
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel
    import torch
    
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    
    #  Override format_prompt to NOT use chat template (same as load())
    def format_prompt_no_chat(question: str) -> str:
        return question
    
    llm.format_prompt = format_prompt_no_chat
    
    # Override generate() to use 150 tokens (same as load())
    def generate_rft(prompt: str) -> str:
        inputs = llm.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(llm.device)
        
        with torch.no_grad():
            outputs = llm.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                eos_token_id=llm.tokenizer.eos_token_id,
                pad_token_id=llm.tokenizer.eos_token_id,
                do_sample=False
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        return llm.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    llm.generate = generate_rft
    
    #  Also override batched_generate (used by benchmark)
    def batched_generate_rft_test(prompts, num_return_sequences=None, temperature=0):
        from tqdm import tqdm
        
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(range(0, len(prompts), micro_batch_size), 
                          desc=f"LLM Running on Micro Batches {micro_batch_size}"):
                batch_prompts = prompts[idx : idx + micro_batch_size]
                batch_results = batched_generate_rft_test(batch_prompts, num_return_sequences, temperature)
                results.extend(batch_results)
            return results
        
        original_padding_side = llm.tokenizer.padding_side
        llm.tokenizer.padding_side = "left"
        
        try:
            inputs = llm.tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(llm.device)
            
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": 150,
                "eos_token_id": llm.tokenizer.eos_token_id,
                "pad_token_id": llm.tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                })
            else:
                generation_kwargs["do_sample"] = False
                
            if num_return_sequences is not None:
                generation_kwargs["num_return_sequences"] = num_return_sequences
            
            with torch.no_grad():
                outputs = llm.model.generate(**generation_kwargs)
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            
            decoded_outputs = llm.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            if num_return_sequences is None:
                return decoded_outputs
            else:
                result = []
                for i in range(len(prompts)):
                    start_idx = i * num_return_sequences
                    end_idx = start_idx + num_return_sequences
                    result.append(decoded_outputs[start_idx:end_idx])
                return result
        finally:
            llm.tokenizer.padding_side = original_padding_side
    
    llm.batched_generate = batched_generate_rft_test

    benchmark_result = benchmark(llm, testset, 100)
    print(f"accuracy={benchmark_result.accuracy:.2f}  answer_rate={benchmark_result.answer_rate:.2f}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})