import torch
from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Create a more concise system message and example
        messages = [
            {
                "role": "system", 
                "content": "Solve unit conversions step-by-step. End with <answer>number</answer>."
            },
            {
                "role": "user",
                "content": "Convert 5 feet to meters."
            },
            {
                "role": "assistant", 
                "content": "1 foot = 0.3048 meters\n5 × 0.3048 = 1.524\n<answer>1.524</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def generate(self, prompt: str) -> str:
        """
        Override generate to use optimized tokens for Chain of Thought reasoning
        """
        # Format the prompt using chat template
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize the input with proper attention mask
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate with optimized parameters for speed
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=80,  # Reduced from 150 to 80 for speed
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        # Extract only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        # Decode and return
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Override batched_generate with optimized parameters for speed
        """
        from tqdm import tqdm
        
        # Handle micro-batching for large inputs
        micro_batch_size = 16  # Reduced from 32 to 16 for speed
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size), 
                desc=f"LLM Running on Micro Batches {micro_batch_size}"
            ):
                batch_prompts = prompts[idx : idx + micro_batch_size]
                batch_results = self.batched_generate(batch_prompts, num_return_sequences, temperature)
                results.extend(batch_results)
            return results
        
        # Format all prompts using chat template
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        # Store original padding side
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        try:
            # Tokenize with proper attention masks
            inputs = self.tokenizer(
                formatted_prompts,
                padding=True, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Set up generation parameters optimized for speed
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": 80,  # Reduced from 150 to 80 for speed
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Configure sampling
            if temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                })
            else:
                generation_kwargs["do_sample"] = False
                
            # Handle multiple return sequences
            if num_return_sequences is not None:
                generation_kwargs["num_return_sequences"] = num_return_sequences
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Extract only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            
            # Decode the outputs
            decoded_outputs = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Return in the correct format
            if num_return_sequences is None:
                return decoded_outputs
            else:
                # Group by original prompts
                result = []
                for i in range(len(prompts)):
                    start_idx = i * num_return_sequences
                    end_idx = start_idx + num_return_sequences
                    result.append(decoded_outputs[start_idx:end_idx])
                return result
                
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})