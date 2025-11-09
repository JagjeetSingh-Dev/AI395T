from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, str(model_path)).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    README says: NO chat template, just "question <answer>X</answer>"
    """
    # Simple rounding to 2 decimals - what was working before
    rounded_answer = round(float(answer), 2)
    
    return {
        "question": prompt,
        "answer": f"<answer>{rounded_answer}</answer>"
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "homework/sft_model",
    **kwargs,
):
    """
    SFT: Train model to complete questions with <answer>X</answer>
    NO chat template per README
    """
    import os
    import torch
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from pathlib import Path
    
    os.environ["WANDB_DISABLED"] = "true"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=== SFT Training ===")
    
    # Load base model
    llm = BaseLLM()
    if llm.tokenizer.pad_token is None:
        llm.tokenizer.pad_token = llm.tokenizer.eos_token
    
    # LoRA config - keep exactly as before (this was working)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1
    )
    
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()
    llm.model.print_trainable_parameters()
    
    # Dataset
    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
    
    # Training args - CAREFUL tuning for stability
    training_args = TrainingArguments(
        output_dir=str(output_path),
        logging_dir=str(output_path),
        num_train_epochs=4,  # What was working
        per_device_train_batch_size=32,
        learning_rate=5e-4,  # Back to what worked
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # NO FP16 - it caused NaN gradients!
    )
    
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=llm.tokenizer,
    )
    
    print("Training...")
    trainer.train()
    trainer.save_model(str(output_path))
    print(f"✅ Saved to {output_path}")
    
    test_model(str(output_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})